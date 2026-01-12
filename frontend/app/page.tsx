'use client'

import Link from 'next/link'

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative pt-24 pb-20 bg-gradient-subtle">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl">
            <div className="badge mb-6">
              <span className="w-1.5 h-1.5 bg-accent rounded-full" />
              AI-Powered Analysis
            </div>
            
            <h1 className="text-4xl md:text-5xl font-bold leading-tight mb-6">
              Understand emotions from
              <span className="text-accent"> any source</span>
            </h1>
            
            <p className="text-lg text-zinc-400 mb-8 leading-relaxed">
              Extract emotional insights from audio, video, and text using 
              state-of-the-art transformer models. Built for developers.
            </p>
            
            <div className="flex gap-3">
              <Link href="/analyze/audio" className="btn-primary">
                Start Analyzing
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </Link>
              <Link href="#features" className="btn-secondary">
                Learn More
              </Link>
            </div>
          </div>
          
          {/* Stats */}
          <div className="flex gap-12 mt-16 pt-8 border-t border-border">
            {[
              { value: '99%', label: 'Accuracy' },
              { value: '<2s', label: 'Response' },
              { value: '3', label: 'Modalities' },
            ].map((stat, i) => (
              <div key={i}>
                <p className="text-2xl font-bold font-mono text-accent">{stat.value}</p>
                <p className="text-zinc-500 text-sm">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="mb-12">
            <p className="text-accent text-sm font-medium mb-2">Capabilities</p>
            <h2 className="text-2xl font-bold">Three modalities, one platform</h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6">
            {/* Audio */}
            <Link href="/analyze/audio" className="card card-hover p-6 group">
              <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <h3 className="font-semibold mb-2 group-hover:text-accent transition-colors">Audio Analysis</h3>
              <p className="text-zinc-500 text-sm leading-relaxed mb-4">
                Speech emotion recognition with Wav2Vec2. Detect emotions from voice recordings.
              </p>
              <div className="flex gap-2">
                {['WAV', 'MP3', 'FLAC'].map(f => (
                  <span key={f} className="text-xs text-zinc-600 font-mono">{f}</span>
                ))}
              </div>
            </Link>

            {/* Video */}
            <Link href="/analyze/video" className="card card-hover p-6 group">
              <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="font-semibold mb-2 group-hover:text-accent transition-colors">Video Analysis</h3>
              <p className="text-zinc-500 text-sm leading-relaxed mb-4">
                Facial emotion detection with DeepFace. Frame-by-frame analysis.
              </p>
              <div className="flex gap-2">
                {['MP4', 'YouTube', 'WebM'].map(f => (
                  <span key={f} className="text-xs text-zinc-600 font-mono">{f}</span>
                ))}
              </div>
            </Link>

            {/* Text */}
            <Link href="/analyze/text" className="card card-hover p-6 group">
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="font-semibold mb-2 group-hover:text-accent transition-colors">Text Analysis</h3>
              <p className="text-zinc-500 text-sm leading-relaxed mb-4">
                Sentiment analysis with FinBERT. Named entity recognition included.
              </p>
              <div className="flex gap-2">
                {['PDF', 'TXT', 'Direct'].map(f => (
                  <span key={f} className="text-xs text-zinc-600 font-mono">{f}</span>
                ))}
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* API Section */}
      <section className="py-20 border-t border-border">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <p className="text-accent text-sm font-medium mb-2">For Developers</p>
              <h2 className="text-2xl font-bold mb-4">Simple REST API</h2>
              <p className="text-zinc-400 mb-6">
                Integrate emotion analysis into your apps with our straightforward API. 
                JWT auth, API keys, and async processing.
              </p>
              <Link href="#" className="btn-secondary inline-flex">
                View Documentation
              </Link>
            </div>
            
            {/* Code Block */}
            <div className="card p-4 font-mono text-sm overflow-hidden">
              <div className="flex gap-1.5 mb-4">
                <span className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
                <span className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
                <span className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
              </div>
              <pre className="text-zinc-400 text-xs leading-relaxed overflow-x-auto">
{`curl -X POST \\
  https://api.recognition.io/v1/audio \\
  -H "X-API-Key: your_key" \\
  -F "file=@audio.wav"

# Response
{
  "emotion": "happy",
  "confidence": 94.2
}`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <h2 className="text-2xl font-bold mb-4">Ready to get started?</h2>
          <p className="text-zinc-500 mb-6">Try it free with 100 API calls. No credit card required.</p>
          <Link href="/analyze/audio" className="btn-primary inline-flex">
            Start Free
          </Link>
        </div>
      </section>
    </div>
  )
}
