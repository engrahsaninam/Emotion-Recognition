'use client'

import { useState, useCallback, useRef } from 'react'

interface FileUploaderProps {
  accept: string
  onFileSelect: (file: File) => void
  label: string
}

export default function FileUploader({ accept, onFileSelect, label }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      setSelectedFile(file)
      onFileSelect(file)
    }
  }, [onFileSelect])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      onFileSelect(file)
    }
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`upload-zone text-center ${isDragging ? 'active' : ''} ${selectedFile ? 'success' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
      />
      
      {selectedFile ? (
        <>
          <div className="w-12 h-12 mx-auto mb-4 bg-green-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <p className="font-medium mb-1">{selectedFile.name}</p>
          <p className="text-zinc-500 text-sm font-mono">{formatSize(selectedFile.size)}</p>
          <p className="text-accent text-sm mt-3">Click to change</p>
        </>
      ) : (
        <>
          <div className="w-12 h-12 mx-auto mb-4 bg-card rounded-xl flex items-center justify-center border border-border">
            <svg className="w-6 h-6 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <p className="font-medium mb-1">{label}</p>
          <p className="text-zinc-500 text-sm">Drag & drop or click to browse</p>
          <div className="flex justify-center gap-2 mt-4">
            {accept.split(',').slice(0, 4).map((type, i) => (
              <span key={i} className="text-xs text-zinc-600 font-mono">
                {type.replace('.', '').toUpperCase()}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
