'use client'

import { useMemo } from 'react'

interface EmotionChartProps {
  emotions: Record<string, number>
  primaryEmotion: string
}

const EMOTION_COLORS: Record<string, string> = {
  happy: '#22c55e',
  joy: '#22c55e',
  sad: '#3b82f6',
  sadness: '#3b82f6',
  angry: '#ef4444',
  anger: '#ef4444',
  fear: '#a855f7',
  surprise: '#f59e0b',
  surprised: '#f59e0b',
  disgust: '#14b8a6',
  neutral: '#71717a',
  positive: '#22c55e',
  negative: '#ef4444',
}

export default function EmotionChart({ emotions, primaryEmotion }: EmotionChartProps) {
  const sortedEmotions = useMemo(() => {
    return Object.entries(emotions)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 6)
  }, [emotions])

  const maxValue = Math.max(...Object.values(emotions), 1)

  const getColor = (emotion: string) => {
    return EMOTION_COLORS[emotion.toLowerCase()] || '#71717a'
  }

  return (
    <div className="space-y-4">
      {sortedEmotions.map(([emotion, value]) => {
        const percentage = (value / maxValue) * 100
        const color = getColor(emotion)
        const isPrimary = emotion.toLowerCase() === primaryEmotion.toLowerCase()
        
        return (
          <div key={emotion}>
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <span 
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className={`text-sm capitalize ${isPrimary ? 'font-medium' : 'text-zinc-400'}`}>
                  {emotion}
                </span>
                {isPrimary && (
                  <span className="text-[10px] font-medium text-accent bg-accent/10 px-2 py-0.5 rounded-full">
                    Primary
                  </span>
                )}
              </div>
              <span className="text-sm font-mono text-zinc-500">
                {typeof value === 'number' ? value.toFixed(1) : value}%
              </span>
            </div>
            <div className="h-1.5 bg-card rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${percentage}%`, backgroundColor: color }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
