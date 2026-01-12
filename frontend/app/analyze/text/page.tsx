'use client'

import { useState } from 'react'
import FileUploader from '@/components/FileUploader'
import ResultsDisplay from '@/components/ResultsDisplay'

interface AnalysisResult {
  primary_emotion: string
  confidence: number
  emotions: Record<string, number>
  sentiment?: string
  sentiment_score?: number
  entities?: Array<{ text: string; label: string }>
}

export default function TextAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [textInput, setTextInput] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'text' | 'file'>('text')

  const handleAnalyze = async () => {
    if (!file && !textInput) return
    setIsAnalyzing(true)
    setError(null)

    try {
      const formData = new FormData()
      if (file) formData.append('file', file)
      else formData.append('text', textInput)

      const response = await fetch('http://localhost:8000/api/v1/analyze/text', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Analysis failed')
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError('Could not connect to API. Showing demo results.')
      setResults({
        primary_emotion: 'positive',
        confidence: 94.2,
        emotions: { positive: 94.2, neutral: 4.1, negative: 1.7 },
        sentiment: 'positive',
        sentiment_score: 0.89,
        entities: [
          { text: 'Apple Inc.', label: 'ORG' },
          { text: '$150B', label: 'MONEY' },
        ],
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
          <div className="badge mb-4" style={{ background: 'rgba(34, 197, 94, 0.1)', borderColor: 'rgba(34, 197, 94, 0.2)', color: '#22c55e' }}>
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full" />
            FinBERT + SpaCy
          </div>
          <h1 className="text-3xl font-bold mb-2">Text Analysis</h1>
          <p className="text-zinc-400">Sentiment analysis and entity recognition</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Input */}
          <div className="space-y-6">
            {/* Mode Toggle */}
            <div className="tab-group">
              <button onClick={() => setMode('text')} className={`tab ${mode === 'text' ? 'active' : ''}`}>
                Direct Input
              </button>
              <button onClick={() => setMode('file')} className={`tab ${mode === 'file' ? 'active' : ''}`}>
                Upload File
              </button>
            </div>
            
            {mode === 'text' ? (
              <div>
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter your text for sentiment analysis...

Example: Apple Inc. reported record revenue of $150B in Q4 2024, exceeding expectations."
                  className="input h-48 resize-none"
                  rows={8}
                />
                <p className="text-zinc-600 text-sm mt-2 text-right">{textInput.length} chars</p>
              </div>
            ) : (
              <FileUploader
                accept=".pdf,.txt,.doc,.docx"
                onFileSelect={setFile}
                label="Drop your document"
              />
            )}
            
            <button
              onClick={handleAnalyze}
              disabled={(!file && !textInput) || isAnalyzing}
              className="btn-primary w-full py-4"
              style={{ background: (file || textInput) && !isAnalyzing ? '#22c55e' : undefined }}
            >
              {isAnalyzing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analyzing...
                </span>
              ) : 'Analyze Text'}
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
                  'Financial sentiment (FinBERT)',
                  'Named Entity Recognition',
                  'PDF/document processing',
                  'Aspect extraction',
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="w-1 h-1 bg-green-400 rounded-full" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Right: Results */}
          <div>
            {results ? (
              <div className="animate-in space-y-6">
                <ResultsDisplay results={results} type="text" />
                
                {/* Named Entities */}
                {results.entities && results.entities.length > 0 && (
                  <div className="card p-5">
                    <p className="font-medium mb-3 text-sm">Named Entities</p>
                    <div className="flex flex-wrap gap-2">
                      {results.entities.map((entity, i) => (
                        <span key={i} className="px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-md text-sm">
                          <span className="text-green-400">{entity.text}</span>
                          <span className="text-zinc-500 ml-2 text-xs">{entity.label}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="card h-full flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-card rounded-full flex items-center justify-center border border-border">
                    <svg className="w-7 h-7 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <p className="text-zinc-500 mb-1">No results yet</p>
                  <p className="text-zinc-600 text-sm">Enter text or upload a document</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
