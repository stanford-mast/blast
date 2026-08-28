import type { AppProps } from 'next/app'
import { Inter, Sometype_Mono } from 'next/font/google'
import localFont from 'next/font/local'
import '../styles/globals.css'

// Same three font families as arker.ai (cloudflare/src/app/layout.tsx:12-32):
// Inter for body copy, Sometype Mono for anything code-ish, and Zalando Sans
// SemiExpanded (self-hosted, same file) as the display face for the wordmark.
const inter = Inter({
  variable: '--font-inter',
  subsets: ['latin'],
  display: 'swap',
})
const sometypeMono = Sometype_Mono({
  variable: '--font-sometype-mono',
  subsets: ['latin'],
  display: 'swap',
})
const zalandoSemiExpanded = localFont({
  src: '../public/fonts/zalando-sans-semi-expanded-latin.woff2',
  variable: '--font-zalando-semi-expanded',
  weight: '200 900',
  display: 'swap',
})

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div
      className={`${inter.variable} ${sometypeMono.variable} ${zalandoSemiExpanded.variable} font-sans`}
    >
      <Component {...pageProps} />
    </div>
  )
}
