import React from 'react'
import Head from 'next/head'
import Image from 'next/image'

const Home: React.FC = () => {
  return (
    <div className="bg-dark text-white min-h-screen">
      <Head>
        <title>BLAST - Browser Language Agent System for Tasks</title>
        <link rel="icon" href="/assets/blast_icon_only.svg" type="image/svg+xml" />
      </Head>

      <header>
        <nav className="py-4">
          <div className="container mx-auto px-6">
            <div className="flex items-center justify-between">
              <a href="/" className="flex items-center space-x-2">
                <Image src="/assets/blast_icon_only.svg" alt="BLAST" width={40} height={40} />
                <span className="text-xl font-bold">BLAST</span>
              </a>
              <div className="space-x-6">
                <a href="https://docs.blastproject.org" className="hover:text-blast-yellow">Docs</a>
                <a href="https://github.com/stanford-mast/blast" className="hover:text-blast-yellow">GitHub</a>
                <a href="https://x.com/realcalebwin" className="hover:text-blast-yellow">Twitter</a>
              </div>
            </div>
          </div>
        </nav>
      </header>

      <main>
        <section className="py-20 text-center">
          <div className="container mx-auto px-6">
            <Image src="/assets/blast_icon_only.svg" alt="BLAST" width={200} height={200} className="mx-auto mb-8" />
            <h1 className="text-6xl font-bold mb-4">BLAST</h1>
            <p className="text-2xl mb-8 text-gray-300">Browser Language Agent System for Tasks</p>
            <div className="space-x-4">
              <a href="https://docs.blastproject.org/get-started/quickstart" 
                 className="bg-blast-yellow text-black px-8 py-3 rounded-lg font-medium hover:opacity-90">
                Get Started
              </a>
              <a href="https://vimeo.com/1079613095/7e90cc78f7?ts=0&share=copy"
                 className="border border-blast-yellow text-blast-yellow px-8 py-3 rounded-lg font-medium hover:bg-blast-yellow hover:text-black">
                Watch Demo
              </a>
            </div>
          </div>
        </section>

        <section className="py-20 bg-black bg-opacity-30">
          <div className="container mx-auto px-6">
            <h2 className="text-4xl font-bold mb-12 text-center">Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              <div className="p-6 rounded-lg bg-black bg-opacity-50">
                <h3 className="text-xl font-bold mb-2 text-blast-yellow">OpenAI-Compatible API</h3>
                <p>Drop-in replacement for OpenAI&apos;s API</p>
              </div>
              <div className="p-6 rounded-lg bg-black bg-opacity-50">
                <h3 className="text-xl font-bold mb-2 text-blast-yellow">High Performance</h3>
                <p>Built for concurrency and parallel processing</p>
              </div>
              <div className="p-6 rounded-lg bg-black bg-opacity-50">
                <h3 className="text-xl font-bold mb-2 text-blast-yellow">Streaming Support</h3>
                <p>Real-time streaming of AI responses</p>
              </div>
              <div className="p-6 rounded-lg bg-black bg-opacity-50">
                <h3 className="text-xl font-bold mb-2 text-blast-yellow">Smart Caching</h3>
                <p>Intelligent caching system for improved performance</p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-20">
          <div className="container mx-auto px-6">
            <h2 className="text-4xl font-bold mb-12 text-center">Quick Start</h2>
            <div className="max-w-3xl mx-auto space-y-8">
              <div>
                <p className="mb-2">Install BLAST:</p>
                <pre className="bg-black bg-opacity-50 p-4 rounded-lg">
                  <code>pip install blastai</code>
                </pre>
              </div>
              <div>
                <p className="mb-2">Start the server:</p>
                <pre className="bg-black bg-opacity-50 p-4 rounded-lg">
                  <code>blastai serve</code>
                </pre>
              </div>
              <div>
                <p className="mb-2">Use it like the OpenAI API:</p>
                <pre className="bg-black bg-opacity-50 p-4 rounded-lg overflow-x-auto">
                  <code>{`from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://127.0.0.1:8000"
)

stream = client.responses.create(
    model="not-needed",
    input="Search for Python docs",
    stream=True
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)`}</code>
                </pre>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="py-8 border-t border-gray-800">
        <div className="container mx-auto px-6">
          <div className="flex flex-col items-center space-y-4">
            <div className="space-x-6">
              <a href="https://docs.blastproject.org" className="hover:text-blast-yellow">Documentation</a>
              <a href="https://github.com/stanford-mast/blast" className="hover:text-blast-yellow">GitHub</a>
              <a href="https://x.com/realcalebwin" className="hover:text-blast-yellow">Twitter</a>
            </div>
            <p className="text-gray-500">© 2024 BLAST Project. MIT License.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Home