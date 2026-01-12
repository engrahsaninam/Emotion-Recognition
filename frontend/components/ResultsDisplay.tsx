'use client'

import EmotionChart from './EmotionChart'

interface AnalysisResult {
  primary_emotion: string
  confidence: number
  emotions: Record<string, number>
  sentiment?: string
  sentiment_score?: number
}

interface ResultsDisplayProps {
  results: AnalysisResult
  type: 'audio' | 'video' | 'text'
}

const EMOTION_COLORS: Record<string, string> = {
  happy: '#22c55e',
  sad: '#3b82f6',
  angry: '#ef4444',
  fear: '#a855f7',
  surprise: '#f59e0b',
  neutral: '#71717a',
  positive: '#22c55e',
  negative: '#ef4444',
}

export default function ResultsDisplay({ results, type }: ResultsDisplayProps) {
  const color = EMOTION_COLORS[results.primary_emotion.toLowerCase()] || '#3b82f6'

  return (
    <div className="space-y-6">
      {/* Primary Result */}
      <div className="card p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <p className="text-zinc-500 text-sm mb-1">Primary Emotion</p>
            <h3 className="text-3xl font-bold capitalize" style={{ color }}>
              {results.primary_emotion}
            </h3>
          </div>
          
          {/* Confidence Circle */}
          <div className="relative w-16 h-16">
            <svg className="w-16 h-16 -rotate-90" viewBox="0 0 64 64">
              <circle
                cx="32" cy="32" r="28"
                fill="none"
                stroke="#27272a"
                strokeWidth="4"
              />
              <circle
                cx="32" cy="32" r="28"
                fill="none"
                stroke={color}
                strokeWidth="4"
                strokeDasharray={`${results.confidence * 1.76} 176`}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-sm font-bold font-mono">{results.confidence.toFixed(0)}%</span>
            </div>
          </div>
        </div>
        
        {/* Sentiment (for text) */}
        {results.sentiment && (
          <div className="flex items-center gap-3 p-3 bg-dark rounded-lg mb-6">
            <span className={`text-sm font-medium capitalize ${
              results.sentiment === 'positive' ? 'text-green-400' : 
              results.sentiment === 'negative' ? 'text-red-400' : 'text-zinc-400'
            }`}>
              {results.sentiment}
            </span>
            {results.sentiment_score !== undefined && (
              <span className="text-zinc-500 text-sm font-mono">
                Score: {results.sentiment_score.toFixed(2)}
              </span>
            )}
          </div>
        )}
        
        {/* Breakdown */}
        <div>
          <p className="text-zinc-500 text-sm mb-4">Breakdown</p>
          <EmotionChart emotions={results.emotions} primaryEmotion={results.primary_emotion} />
        </div>
      </div>
      
      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="card p-4 text-center">
          <p className="text-xl font-bold font-mono" style={{ color }}>
            {Object.keys(results.emotions).length}
          </p>
          <p className="text-zinc-500 text-xs">Emotions</p>
        </div>
        <div className="card p-4 text-center">
          <p className="text-xl font-bold font-mono">{results.confidence.toFixed(0)}%</p>
          <p className="text-zinc-500 text-xs">Confidence</p>
        </div>
        <div className="card p-4 text-center">
          <p className="text-xl font-bold font-mono text-green-400">✓</p>
          <p className="text-zinc-500 text-xs">Complete</p>
        </div>
      </div>
    </div>
  )
}
