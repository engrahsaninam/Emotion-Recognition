'use client'

import { useState } from 'react'
import FileUploader from '@/components/FileUploader'
import ResultsDisplay from '@/components/ResultsDisplay'

interface AnalysisResult {
  primary_emotion: string
  confidence: number
  emotions: Record<string, number>
}

export default function AudioAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    if (!file) return
    setIsAnalyzing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/api/v1/analyze/audio', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Analysis failed')
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError('Could not connect to API. Showing demo results.')
      setResults({
        primary_emotion: 'happy',
        confidence: 87.5,
        emotions: { happy: 87.5, neutral: 8.2, surprised: 2.8, sad: 1.0, angry: 0.3, fear: 0.2 },
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen py-12">
      <div className="max-w-6xl mx-auto px-6">
        {/* Header */}
        <div className="mb-10">
          <div className="badge mb-4">
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
            Wav2Vec2
          </div>
          <h1 className="text-3xl font-bold mb-2">Audio Analysis</h1>
          <p className="text-zinc-400">Extract emotions from voice recordings</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Upload */}
          <div className="space-y-6">
            <FileUploader
              accept=".wav,.mp3,.flac,.ogg,.m4a"
              onFileSelect={setFile}
              label="Drop your audio file"
            />
            
            <button
              onClick={handleAnalyze}
              disabled={!file || isAnalyzing}
              className="btn-primary w-full py-4"
            >
              {isAnalyzing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analyzing...
                </span>
              ) : 'Analyze Audio'}
            </button>
            
            {error && (
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <p className="text-yellow-400 text-sm">{error}</p>
              </div>
            )}
            
            {/* Capabilities */}
            <div className="card p-5">
              <p className="font-medium mb-3 text-sm">Capabilities</p>
              <ul className="space-y-2 text-sm text-zinc-400">
                {[
                  'Speech emotion recognition',
                  'Multi-language support',
                  'Real-time processing',
                  'Confidence scoring',
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="w-1 h-1 bg-blue-400 rounded-full" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Right: Results */}
          <div>
            {results ? (
              <div className="animate-in">
                <ResultsDisplay results={results} type="audio" />
              </div>
            ) : (
              <div className="card h-full flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-card rounded-full flex items-center justify-center border border-border">
                    <svg className="w-7 h-7 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </div>
                  <p className="text-zinc-500 mb-1">No results yet</p>
                  <p className="text-zinc-600 text-sm">Upload an audio file to start</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
