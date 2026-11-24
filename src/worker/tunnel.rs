//! The worker end of the control plane's tunnel.
//!
//! The worker opens a WebSocket to the heartbeat's `tunnel_url` with its join
//! key and subprotocol `worker-tunnel.v1`. Binary messages carry one ordered
//! byte stream (text messages are ignored); over it runs yamux with the
//! worker as server and the router opening every stream. Each stream is one
//! HTTP/1.1 connection (keep-alive and upgrades allowed) served by BLAST's
//! own API router. Requests on the tunnel are trusted: only the router can
//! write to it.

use std::{
    io,
    pin::Pin,
    task::{ready, Context, Poll},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use axum::Router;
use futures::{Sink, Stream};
use hyper_util::{rt::TokioIo, service::TowerToHyperService};
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::{
    self,
    client::IntoClientRequest,
    http::{header, HeaderValue},
};
use tokio_util::compat::FuturesAsyncReadCompatExt;
use tracing::{debug, info, warn};

pub const SUBPROTOCOL: &str = "worker-tunnel.v1";

const FIRST_RETRY: Duration = Duration::from_secs(1);
const MAX_RETRY: Duration = Duration::from_secs(30);

/// Keeps the tunnel open to whatever URL the latest heartbeat named,
/// reconnecting with capped exponential backoff.
pub async fn run(mut url_rx: watch::Receiver<Option<String>>, api_key: String, router: Router) {
    let mut retry = FIRST_RETRY;
    loop {
        let Ok(url) = url_rx
            .wait_for(Option::is_some)
            .await
            .map(|url| url.clone().unwrap_or_default())
        else {
            return;
        };
        let connected_at = Instant::now();
        match connect(&url, &api_key).await {
            Ok(ws) => {
                info!(%url, "tunnel connected");
                tokio::select! {
                    result = serve(ws, router.clone()) => match result {
                        Ok(()) => info!("tunnel closed"),
                        Err(e) => warn!(err = %format!("{e:#}"), "tunnel failed"),
                    },
                    changed = url_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                        info!("tunnel membership changed; closing connection");
                    }
                }
            }
            Err(e) => {
                warn!(%url, err = %format!("{e:#}"), retry_in = ?retry, "tunnel connect failed");
            }
        }
        // A connection that held for a while was healthy: start over at the
        // shortest delay rather than where the last outage left off.
        if connected_at.elapsed() > MAX_RETRY {
            retry = FIRST_RETRY;
        }
        if url_rx.borrow().is_some() {
            tokio::time::sleep(retry).await;
        }
        retry = (retry * 2).min(MAX_RETRY);
    }
}

pub async fn connect(
    url: &str,
    api_key: &str,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
> {
    let mut request = url.into_client_request().context("tunnel url")?;
    let headers = request.headers_mut();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .context("api key is not a valid header")?,
    );
    headers.insert(
        header::SEC_WEBSOCKET_PROTOCOL,
        HeaderValue::from_static(SUBPROTOCOL),
    );
    let (ws, _response) = tokio_tungstenite::connect_async(request)
        .await
        .context("websocket handshake")?;
    Ok(ws)
}

/// Runs yamux (as server) over `ws` until the connection ends, serving every
/// inbound stream as an HTTP/1.1 connection on `router`.
pub async fn serve<S, M, E>(ws: S, router: Router) -> Result<()>
where
    S: Stream<Item = Result<M, E>> + Sink<M, Error = E> + Unpin + Send + 'static,
    M: WsMessage,
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let mut conn =
        yamux::Connection::new(WsIo::new(ws), yamux::Config::default(), yamux::Mode::Server);
    loop {
        match futures::future::poll_fn(|cx| conn.poll_next_inbound(cx)).await {
            Some(Ok(stream)) => {
                let router = router.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream.compat());
                    let service = TowerToHyperService::new(router);
                    if let Err(e) = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .with_upgrades()
                        .await
                    {
                        debug!(err = %e, "tunnel stream ended with an error");
                    }
                });
            }
            Some(Err(e)) => return Err(e).context("yamux"),
            None => return Ok(()),
        }
    }
}

/// What the byte-stream adapter needs from a WebSocket message type, so the
/// same adapter runs over a tungstenite client and an axum server socket.
pub trait WsMessage: Sized {
    fn binary(data: Vec<u8>) -> Self;
    fn into_frame(self) -> Frame;
}

pub enum Frame {
    Data(Vec<u8>),
    Close,
    /// Text, ping, and pong: not part of the byte stream. The WebSocket
    /// library answers pings itself.
    Ignored,
}

impl WsMessage for tungstenite::Message {
    fn binary(data: Vec<u8>) -> Self {
        Self::Binary(data)
    }

    fn into_frame(self) -> Frame {
        match self {
            Self::Binary(data) => Frame::Data(data),
            Self::Close(_) => Frame::Close,
            _ => Frame::Ignored,
        }
    }
}

/// A WebSocket viewed as one ordered byte stream: binary message payloads
/// concatenated on read, each write sent as one binary message.
pub struct WsIo<S, M> {
    ws: S,
    binary: fn(Vec<u8>) -> M,
    read_buf: Vec<u8>,
    read_pos: usize,
    eof: bool,
}

impl<S, M: WsMessage> WsIo<S, M> {
    pub fn new(ws: S) -> Self {
        Self {
            ws,
            binary: M::binary,
            read_buf: Vec::new(),
            read_pos: 0,
            eof: false,
        }
    }
}

fn io_err<E: Into<Box<dyn std::error::Error + Send + Sync>>>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, e)
}

impl<S, M, E> futures::io::AsyncRead for WsIo<S, M>
where
    S: Stream<Item = Result<M, E>> + Unpin,
    M: WsMessage,
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        loop {
            if this.read_pos < this.read_buf.len() {
                let n = buf.len().min(this.read_buf.len() - this.read_pos);
                buf[..n].copy_from_slice(&this.read_buf[this.read_pos..this.read_pos + n]);
                this.read_pos += n;
                return Poll::Ready(Ok(n));
            }
            if this.eof {
                return Poll::Ready(Ok(0));
            }
            match ready!(Pin::new(&mut this.ws).poll_next(cx)) {
                Some(Ok(msg)) => match msg.into_frame() {
                    Frame::Data(data) => {
                        this.read_buf = data;
                        this.read_pos = 0;
                    }
                    Frame::Close => this.eof = true,
                    Frame::Ignored => {}
                },
                Some(Err(e)) => return Poll::Ready(Err(io_err(e))),
                None => this.eof = true,
            }
        }
    }
}

impl<S, M> futures::io::AsyncWrite for WsIo<S, M>
where
    S: Sink<M> + Unpin,
    M: WsMessage,
    S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        ready!(Pin::new(&mut this.ws).poll_ready(cx)).map_err(io_err)?;
        Pin::new(&mut this.ws)
            .start_send((this.binary)(buf.to_vec()))
            .map_err(io_err)?;
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().ws)
            .poll_flush(cx)
            .map_err(io_err)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().ws)
            .poll_close(cx)
            .map_err(io_err)
    }
}
