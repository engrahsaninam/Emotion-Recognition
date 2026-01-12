'use client'

import { useState } from 'react'
import FileUploader from '@/components/FileUploader'
import ResultsDisplay from '@/components/ResultsDisplay'

interface AnalysisResult {
  primary_emotion: string
  confidence: number
  emotions: Record<string, number>
}

export default function VideoAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'file' | 'url'>('file')

  const handleAnalyze = async () => {
    if (!file && !youtubeUrl) return
    setIsAnalyzing(true)
    setError(null)

    try {
      const formData = new FormData()
      if (file) formData.append('file', file)
      else formData.append('youtube_url', youtubeUrl)

      const response = await fetch('http://localhost:8000/api/v1/analyze/video', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Analysis failed')
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError('Could not connect to API. Showing demo results.')
      setResults({
        primary_emotion: 'neutral',
        confidence: 72.3,
        emotions: { neutral: 72.3, happy: 15.8, sad: 5.2, surprised: 4.1, angry: 1.8, fear: 0.8 },
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
          <div className="badge mb-4" style={{ background: 'rgba(168, 85, 247, 0.1)', borderColor: 'rgba(168, 85, 247, 0.2)', color: '#a855f7' }}>
            <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
            DeepFace + MediaPipe
          </div>
          <h1 className="text-3xl font-bold mb-2">Video Analysis</h1>
          <p className="text-zinc-400">Detect facial emotions from video content</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Upload */}
          <div className="space-y-6">
            {/* Mode Toggle */}
            <div className="tab-group">
              <button onClick={() => setMode('file')} className={`tab ${mode === 'file' ? 'active' : ''}`}>
                File Upload
              </button>
              <button onClick={() => setMode('url')} className={`tab ${mode === 'url' ? 'active' : ''}`}>
                YouTube URL
              </button>
            </div>
            
            {mode === 'file' ? (
              <FileUploader
                accept=".mp4,.webm,.avi,.mov,.mkv"
                onFileSelect={setFile}
                label="Drop your video file"
              />
            ) : (
              <div>
                <input
                  type="url"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="https://youtube.com/watch?v=..."
                  className="input"
                />
                <p className="text-zinc-600 text-sm mt-2">Paste any YouTube video URL</p>
              </div>
            )}
            
            <button
              onClick={handleAnalyze}
              disabled={(!file && !youtubeUrl) || isAnalyzing}
              className="btn-primary w-full py-4"
              style={{ background: (file || youtubeUrl) && !isAnalyzing ? '#a855f7' : undefined }}
            >
              {isAnalyzing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analyzing...
                </span>
              ) : 'Analyze Video'}
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
                  'Facial emotion detection',
                  'Multi-face tracking',
                  'YouTube video support',
                  'Frame-by-frame analysis',
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="w-1 h-1 bg-purple-400 rounded-full" />
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
                <ResultsDisplay results={results} type="video" />
              </div>
            ) : (
              <div className="card h-full flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-card rounded-full flex items-center justify-center border border-border">
                    <svg className="w-7 h-7 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <p className="text-zinc-500 mb-1">No results yet</p>
                  <p className="text-zinc-600 text-sm">Upload a video or paste a URL</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
