import React from 'react'
import Head from 'next/head'

// Sharp-corner, 50ms-hover link "button", the same treatment arker.ai uses
// for its secondary nav/footer actions
// (cloudflare/src/components/landing/nav-client.tsx:187,198: bg-surface,
// gap-2, px-4 py-2.5, text-base font-semibold, hover:bg-surface-hover,
// transition-colors duration-[50ms], verified literal class-for-class
// against that file, not approximated).
function LinkButton({
  href,
  internal,
  children,
}: {
  href: string
  /** Same-site route (e.g. /docs); skip target="_blank"/rel. */
  internal?: boolean
  children: React.ReactNode
}) {
  return (
    <a
      href={href}
      {...(internal ? {} : { target: '_blank', rel: 'noopener noreferrer' })}
      className="inline-flex items-center gap-2 bg-surface px-4 py-2.5 text-base font-semibold text-ink-secondary transition-colors duration-50 hover:bg-surface-hover hover:text-ink"
    >
      {children}
    </a>
  )
}

function DocsIcon() {
  return (
    <svg
      className="size-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
    >
      <path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H15l5 5v11.5a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19.5v-15Z" />
      <path d="M14 3v5.5a.5.5 0 0 0 .5.5H20" />
      <path d="M8 13h8M8 17h5" />
    </svg>
  )
}

function DiscordIcon() {
  return (
    <svg className="size-4" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.317 4.492c-1.53-.69-3.17-1.2-4.885-1.49a.075.075 0 0 0-.079.036c-.21.369-.444.85-.608 1.23a18.566 18.566 0 0 0-5.487 0 12.36 12.36 0 0 0-.617-1.23A.077.077 0 0 0 8.562 3c-1.714.29-3.354.8-4.885 1.491a.07.07 0 0 0-.032.027C.533 9.093-.32 13.555.099 17.961a.08.08 0 0 0 .031.055 20.03 20.03 0 0 0 5.993 2.98.078.078 0 0 0 .084-.026 13.83 13.83 0 0 0 1.226-1.963.074.074 0 0 0-.041-.104 13.175 13.175 0 0 1-1.872-.878.075.075 0 0 1-.008-.125c.126-.093.252-.19.372-.287a.075.075 0 0 1 .078-.01c3.927 1.764 8.18 1.764 12.061 0a.075.075 0 0 1 .079.009c.12.098.245.195.372.288a.075.075 0 0 1-.006.125c-.598.344-1.22.635-1.873.877a.075.075 0 0 0-.041.105c.36.687.772 1.341 1.225 1.962a.077.077 0 0 0 .084.028 19.963 19.963 0 0 0 6.002-2.981.076.076 0 0 0 .032-.054c.5-5.094-.838-9.52-3.549-13.442a.06.06 0 0 0-.031-.028zM8.02 15.278c-1.182 0-2.157-1.069-2.157-2.38 0-1.312.956-2.38 2.157-2.38 1.21 0 2.176 1.077 2.157 2.38 0 1.312-.956 2.38-2.157 2.38zm7.975 0c-1.183 0-2.157-1.069-2.157-2.38 0-1.312.955-2.38 2.157-2.38 1.21 0 2.176 1.077 2.157 2.38 0 1.312-.946 2.38-2.157 2.38z" />
    </svg>
  )
}

function GitHubIcon() {
  return (
    <svg className="size-4" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  )
}

const Home: React.FC = () => {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-page px-6 text-ink">
      <Head>
        <title>BLAST</title>
        <meta name="description" content="Open-source VMs-as-a-service" />
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/blast_icon_only.svg" type="image/svg+xml" />
        <meta name="theme-color" content="#121212" />
      </Head>

      <main className="flex flex-col items-center text-center">
        {/* The actual BLAST mark (website/public/blast_icon_only.svg),
            centered, no wordmark alongside it. */}
        <img
          src="/blast_icon_only.svg"
          alt="BLAST"
          className="mb-8 h-28 w-28 sm:h-36 sm:w-36"
        />
        <h1 className="sr-only">BLAST</h1>

        <p className="mb-12 max-w-xl text-base text-ink-secondary sm:text-lg">
          Open-source VMs-as-a-service
        </p>

        <div className="flex flex-wrap items-center justify-center gap-3">
          <LinkButton href="/docs/" internal>
            <DocsIcon />
            Docs
          </LinkButton>
          <LinkButton href="https://discord.gg/NqrkJwYYh4">
            <DiscordIcon />
            Discord
          </LinkButton>
          <LinkButton href="https://github.com/stanford-mast/blast">
            <GitHubIcon />
            GitHub
          </LinkButton>
        </div>
      </main>
    </div>
  )
}

export default Home
