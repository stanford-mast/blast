import { useEffect, useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [serverStatus, setServerStatus] = useState<'checking' | 'ready' | 'error'>('checking')
  const [errorMessage, setErrorMessage] = useState<string>('')

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/health')
        if (response.data.ready) {
          setServerStatus('ready')
        } else {
          setServerStatus('error')
          setErrorMessage('Server is not ready')
        }
      } catch (error) {
        setServerStatus('error')
        setErrorMessage('Could not connect to server')
      }
    }

    checkHealth()
  }, [])

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">BlastAI Dashboard</h1>
        
        <div className="mb-8 p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-2">Server Status</h2>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              serverStatus === 'ready' ? 'bg-green-500' :
              serverStatus === 'error' ? 'bg-red-500' :
              'bg-yellow-500'
            }`} />
            <span className="capitalize">{serverStatus}</span>
          </div>
          {errorMessage && (
            <p className="text-red-500 mt-2">{errorMessage}</p>
          )}
        </div>
      </div>
    </main>
  )
}