//! A minimal S3 client: the handful of object operations snapshot storage
//! needs, signed with AWS Signature Version 4. Works against AWS S3
//! (virtual-hosted addressing) and S3-compatible servers such as `MinIO`
//! (path-style addressing under a configured endpoint).

use std::{fmt::Write as _, path::Path, time::Duration};

use anyhow::{bail, Context, Result};
use hmac::{Hmac, Mac};
use reqwest::{Method, StatusCode, Url};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Files larger than this go up as a multipart upload in parts of this size,
/// so no single request is limited by S3's 5 GiB object PUT cap and no more
/// than one part is held in memory.
const PART_BYTES: u64 = 64 * 1024 * 1024;

/// Short-lived credentials the control plane hands out with each heartbeat.
/// An empty session token (static keys, as `MinIO` takes) is not sent.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Credentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: String,
    pub expires_at: String,
}

/// A bucket, the key prefix this worker may use in it, and the credentials
/// scoped to that prefix.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ObjectStorage {
    pub bucket: String,
    pub region: String,
    pub prefix: String,
    pub credentials: Credentials,
}

#[derive(Clone)]
pub struct Bucket {
    http: reqwest::Client,
    storage: ObjectStorage,
    /// Path-style base URL of an S3-compatible server; `None` is AWS.
    endpoint: Option<String>,
}

impl Bucket {
    pub fn new(storage: ObjectStorage, endpoint: Option<String>) -> Self {
        Self {
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                // A stalled transfer fails instead of holding a restore (and
                // the request waiting on it) forever.
                .read_timeout(Duration::from_secs(60))
                .build()
                .expect("reqwest client"),
            storage,
            endpoint: endpoint.map(|e| e.trim_end_matches('/').to_owned()),
        }
    }

    pub const fn storage(&self) -> &ObjectStorage {
        &self.storage
    }

    /// The full object key for `key` under this worker's prefix.
    pub fn key(&self, key: &str) -> String {
        let prefix = self.storage.prefix.trim_matches('/');
        if prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{prefix}/{key}")
        }
    }

    fn url(&self, key: &str) -> Result<Url> {
        let path = uri_encode(key, false);
        let ObjectStorage { bucket, region, .. } = &self.storage;
        let url = self.endpoint.as_ref().map_or_else(
            || format!("https://{bucket}.s3.{region}.amazonaws.com/{path}"),
            |endpoint| format!("{endpoint}/{bucket}/{path}"),
        );
        Url::parse(&url).context("object url")
    }

    /// Sends one signed request. `key` is a full object key ("" for the
    /// bucket itself).
    async fn send(
        &self,
        method: Method,
        key: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Result<reqwest::Response> {
        let mut url = self.url(key)?;
        let canonical_query = canonical_query(query);
        if !canonical_query.is_empty() {
            url.set_query(Some(&canonical_query));
        }
        let headers = sign(
            &method,
            &url,
            &canonical_query,
            &body,
            &self.storage,
            &humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string(),
        );
        let mut request = self.http.request(method.clone(), url).body(body);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("{method} {key}"))?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let text = response.text().await.unwrap_or_default();
        Err(S3Error { status, body: text }).with_context(|| format!("{method} {key}"))
    }

    /// The object's bytes, or `None` when it does not exist.
    pub async fn get_bytes(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self
            .send(Method::GET, &self.key(key), &[], Vec::new())
            .await
        {
            Ok(response) => Ok(Some(response.bytes().await?.to_vec())),
            Err(e) if is_not_found(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub async fn put_bytes(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
        self.send(Method::PUT, &self.key(key), &[], bytes).await?;
        Ok(())
    }

    /// Uploads the file at `src` to `key`, in parts when it is large.
    pub async fn put_file(&self, key: &str, src: &Path) -> Result<()> {
        let key = self.key(key);
        let mut file = tokio::fs::File::open(src)
            .await
            .with_context(|| format!("open {}", src.display()))?;
        let size = file.metadata().await?.len();
        if size <= PART_BYTES {
            let mut bytes = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
            file.read_to_end(&mut bytes).await?;
            self.send(Method::PUT, &key, &[], bytes).await?;
            return Ok(());
        }
        let created = self
            .send(Method::POST, &key, &[("uploads", "")], Vec::new())
            .await?;
        let upload_id = xml_values(&created.text().await?, "UploadId")
            .into_iter()
            .next()
            .context("CreateMultipartUpload answered without an UploadId")?;
        match self.put_parts(&key, &upload_id, &mut file, size).await {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = self
                    .send(
                        Method::DELETE,
                        &key,
                        &[("uploadId", &upload_id)],
                        Vec::new(),
                    )
                    .await;
                Err(e)
            }
        }
    }

    async fn put_parts(
        &self,
        key: &str,
        upload_id: &str,
        file: &mut tokio::fs::File,
        size: u64,
    ) -> Result<()> {
        let mut complete = String::from("<CompleteMultipartUpload>");
        let mut offset = 0;
        let mut part_number = 1u32;
        while offset < size {
            let len = PART_BYTES.min(size - offset);
            let mut part = vec![0; usize::try_from(len)?];
            file.read_exact(&mut part).await?;
            let number = part_number.to_string();
            let response = self
                .send(
                    Method::PUT,
                    key,
                    &[("partNumber", &number), ("uploadId", upload_id)],
                    part,
                )
                .await?;
            let etag = response
                .headers()
                .get("etag")
                .and_then(|v| v.to_str().ok())
                .context("UploadPart answered without an ETag")?;
            let _ = write!(
                complete,
                "<Part><PartNumber>{part_number}</PartNumber><ETag>{}</ETag></Part>",
                xml_escape(etag)
            );
            offset += len;
            part_number += 1;
        }
        complete.push_str("</CompleteMultipartUpload>");
        let response = self
            .send(
                Method::POST,
                key,
                &[("uploadId", upload_id)],
                complete.into_bytes(),
            )
            .await?;
        // CompleteMultipartUpload can fail after a 200, with the error in the body.
        let text = response.text().await?;
        if text.contains("<Error>") {
            bail!("CompleteMultipartUpload {key}: {text}");
        }
        Ok(())
    }

    /// Downloads `key` to `dest`; `false` when the object does not exist.
    pub async fn get_file(&self, key: &str, dest: &Path) -> Result<bool> {
        let mut response = match self
            .send(Method::GET, &self.key(key), &[], Vec::new())
            .await
        {
            Ok(response) => response,
            Err(e) if is_not_found(&e) => return Ok(false),
            Err(e) => return Err(e),
        };
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::File::create(dest)
            .await
            .with_context(|| format!("create {}", dest.display()))?;
        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        Ok(true)
    }

    /// Every key under `prefix` (relative to this worker's prefix), relative
    /// to `prefix`.
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.key(prefix);
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query = vec![("list-type", "2"), ("prefix", full_prefix.as_str())];
            if let Some(token) = &token {
                query.push(("continuation-token", token));
            }
            let text = self
                .send(Method::GET, "", &query, Vec::new())
                .await?
                .text()
                .await?;
            keys.extend(
                xml_values(&text, "Key")
                    .into_iter()
                    .filter_map(|k| k.strip_prefix(&full_prefix).map(str::to_owned)),
            );
            token = xml_values(&text, "NextContinuationToken")
                .into_iter()
                .next();
            if token.is_none() {
                return Ok(keys);
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("object storage answered {status}: {body}")]
struct S3Error {
    status: StatusCode,
    body: String,
}

fn is_not_found(e: &anyhow::Error) -> bool {
    e.downcast_ref::<S3Error>()
        .is_some_and(|e| e.status == StatusCode::NOT_FOUND)
}

type HmacSha256 = Hmac<Sha256>;

fn hmac(key: &[u8], data: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

/// The headers that authenticate one request (`SigV4`, header form). `now` is
/// an RFC 3339 UTC timestamp.
fn sign(
    method: &Method,
    url: &Url,
    canonical_query: &str,
    body: &[u8],
    storage: &ObjectStorage,
    now: &str,
) -> Vec<(&'static str, String)> {
    let amz_date: String = now.chars().filter(|c| *c != '-' && *c != ':').collect();
    let date = &amz_date[..8];
    let host_name = url.host_str().unwrap_or_default();
    let host = url.port().map_or_else(
        || host_name.to_owned(),
        |port| format!("{host_name}:{port}"),
    );
    let payload_hash = hex::encode(Sha256::digest(body));
    let creds = &storage.credentials;

    let mut headers = vec![
        ("host", host),
        ("x-amz-content-sha256", payload_hash.clone()),
        ("x-amz-date", amz_date.clone()),
    ];
    if !creds.session_token.is_empty() {
        headers.push(("x-amz-security-token", creds.session_token.clone()));
    }
    let signed_headers = headers
        .iter()
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers = headers.iter().fold(String::new(), |mut out, (n, v)| {
        let _ = writeln!(out, "{n}:{}", v.trim());
        out
    });
    let canonical_request = format!(
        "{method}\n{}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        url.path()
    );
    let scope = format!("{date}/{}/s3/aws4_request", storage.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );
    let key = [date, storage.region.as_str(), "s3", "aws4_request"]
        .iter()
        .fold(
            format!("AWS4{}", creds.secret_access_key).into_bytes(),
            |key, part| hmac(&key, part),
        );
    let signature = hex::encode(hmac(&key, &string_to_sign));

    headers.retain(|(name, _)| *name != "host");
    headers.push((
        "authorization",
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            creds.access_key_id
        ),
    ));
    headers
}

/// `SigV4`'s URI encoding: every byte but the unreserved characters, with `/`
/// kept as is unless `encode_slash` (paths keep it, query values do not).
fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(b));
            }
            b'/' if !encode_slash => out.push('/'),
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

fn canonical_query(query: &[(&str, &str)]) -> String {
    let mut pairs: Vec<(String, String)> = query
        .iter()
        .map(|(k, v)| (uri_encode(k, true), uri_encode(v, true)))
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// The text of every `<tag>` element in an S3 XML response, unescaped. S3's
/// responses are flat enough that this is all the parsing they need.
fn xml_values(xml: &str, tag: &str) -> Vec<String> {
    let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
    let mut values = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        let Some(end) = after.find(&close) else { break };
        values.push(xml_unescape(&after[..end]));
        rest = &after[end + close.len()..];
    }
    values
}

fn xml_unescape(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn storage() -> ObjectStorage {
        ObjectStorage {
            bucket: "examplebucket".into(),
            region: "us-east-1".into(),
            prefix: String::new(),
            credentials: Credentials {
                access_key_id: "AKIAIOSFODNN7EXAMPLE".into(),
                secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
                session_token: String::new(),
                expires_at: "2099-01-01T00:00:00Z".into(),
            },
        }
    }

    fn signature(url: &str, query: &[(&str, &str)]) -> String {
        let url = Url::parse(url).unwrap();
        let now = "2013-05-24T00:00:00Z";
        let headers = sign(
            &Method::GET,
            &url,
            &canonical_query(query),
            b"",
            &storage(),
            now,
        );
        let auth = headers
            .into_iter()
            .find(|(n, _)| *n == "authorization")
            .unwrap()
            .1;
        assert!(
            auth.starts_with(
                "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
                 SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature="
            ),
            "{auth}"
        );
        auth.rsplit('=').next().unwrap().to_owned()
    }

    /// Worked examples from AWS's `SigV4` documentation for S3.
    #[test]
    fn signs_like_the_aws_reference() {
        assert_eq!(
            signature(
                "https://examplebucket.s3.amazonaws.com/",
                &[("lifecycle", "")]
            ),
            "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"
        );
        assert_eq!(
            signature(
                "https://examplebucket.s3.amazonaws.com/",
                &[("max-keys", "2"), ("prefix", "J")]
            ),
            "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
        );
    }

    #[test]
    fn session_token_is_signed() {
        let mut s = storage();
        s.credentials.session_token = "tok".into();
        let url = Url::parse("http://127.0.0.1:9000/b/k").unwrap();
        let headers = sign(&Method::PUT, &url, "", b"x", &s, "2026-01-01T00:00:00Z");
        assert!(headers.contains(&("x-amz-security-token", "tok".into())));
        let auth = &headers
            .iter()
            .find(|(n, _)| *n == "authorization")
            .unwrap()
            .1;
        assert!(auth
            .contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token"));
    }

    #[test]
    fn query_and_path_encoding() {
        assert_eq!(
            canonical_query(&[("prefix", "p/vms/"), ("list-type", "2"), ("uploads", "")]),
            "list-type=2&prefix=p%2Fvms%2F&uploads="
        );
        assert_eq!(uri_encode("a b/c+d", false), "a%20b/c%2Bd");
    }

    #[test]
    fn keys_live_under_the_prefix() {
        let mut s = storage();
        s.prefix = "pools/abc/".into();
        let bucket = Bucket::new(s.clone(), Some("http://127.0.0.1:9000/".into()));
        assert_eq!(bucket.key("vms/v/x"), "pools/abc/vms/v/x");
        assert_eq!(
            bucket.url("pools/abc/vms/v/x").unwrap().as_str(),
            "http://127.0.0.1:9000/examplebucket/pools/abc/vms/v/x"
        );
        s.prefix = String::new();
        let aws = Bucket::new(s, None);
        assert_eq!(
            aws.url("vms/v").unwrap().as_str(),
            "https://examplebucket.s3.us-east-1.amazonaws.com/vms/v"
        );
    }

    #[test]
    fn xml_values_are_unescaped() {
        let xml =
            "<R><Key>a&amp;b</Key><Key>c</Key><NextContinuationToken>t</NextContinuationToken></R>";
        assert_eq!(xml_values(xml, "Key"), vec!["a&b", "c"]);
        assert_eq!(xml_values(xml, "NextContinuationToken"), vec!["t"]);
    }
}
