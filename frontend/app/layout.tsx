import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Recognition — Emotion AI',
  description: 'Multi-modal emotion and sentiment analysis powered by AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-dark text-white min-h-screen">
        {/* Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-dark/90 backdrop-blur-md border-b border-border">
          <div className="max-w-6xl mx-auto px-6">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <a href="/" className="flex items-center gap-3">
                <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                  <span className="font-bold text-sm">R</span>
                </div>
                <span className="font-semibold">Recognition</span>
              </a>
              
              {/* Nav Links */}
              <div className="flex items-center gap-1">
                <a href="/analyze/audio" className="nav-link">Audio</a>
                <a href="/analyze/video" className="nav-link">Video</a>
                <a href="/analyze/text" className="nav-link">Text</a>
                <span className="w-px h-5 bg-border mx-3" />
                <a href="/analyze/audio" className="btn-primary text-sm py-2 px-4">
                  Get Started
                </a>
              </div>
            </div>
          </div>
        </nav>
        
        {/* Main Content */}
        <main className="pt-16">
          {children}
        </main>
        
        {/* Footer */}
        <footer className="border-t border-border mt-24 py-12">
          <div className="max-w-6xl mx-auto px-6">
            <div className="flex flex-col md:flex-row justify-between gap-8">
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-6 h-6 bg-accent rounded flex items-center justify-center">
                    <span className="font-bold text-xs">R</span>
                  </div>
                  <span className="font-semibold text-sm">Recognition</span>
                </div>
                <p className="text-zinc-500 text-sm max-w-xs">
                  Emotion analysis platform for audio, video, and text.
                </p>
              </div>
              <div className="flex gap-12 text-sm">
                <div>
                  <p className="font-medium mb-3">Platform</p>
                  <div className="space-y-2 text-zinc-500">
                    <a href="/analyze/audio" className="block hover:text-white transition-colors">Audio</a>
                    <a href="/analyze/video" className="block hover:text-white transition-colors">Video</a>
                    <a href="/analyze/text" className="block hover:text-white transition-colors">Text</a>
                  </div>
                </div>
                <div>
                  <p className="font-medium mb-3">Developers</p>
                  <div className="space-y-2 text-zinc-500">
                    <a href="#" className="block hover:text-white transition-colors">API Docs</a>
                    <a href="#" className="block hover:text-white transition-colors">Pricing</a>
                  </div>
                </div>
              </div>
            </div>
            <div className="border-t border-border mt-8 pt-6 text-zinc-600 text-sm">
              © 2025 Recognition
            </div>
          </div>
        </footer>
      </body>
    </html>
  )
}
