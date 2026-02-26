"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Mic, MicOff, Send, Upload, FileText, X, Loader2, Volume2, Square } from "lucide-react"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  audioData?: string | null
}

interface UploadedFile {
  name: string
  content: string
}

const DEMO_TOPICS = [
  { topic: "us-politics", userMessage: "Tell me about international news" },
  { topic: "japan-news", userMessage: "What's happening in Japan?" },
]

export function PodcastInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentPlayingId, setCurrentPlayingId] = useState<string | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [demoIndex, setDemoIndex] = useState(0)
  const [isInitialized, setIsInitialized] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioQueueRef = useRef<string[]>([])

  useEffect(() => {
    if (!isInitialized) {
      setIsInitialized(true)
      loadDemoResponse(0)
    }
  }, [isInitialized])

  const loadDemoResponse = async (index: number) => {
    if (index >= DEMO_TOPICS.length) return

    const demo = DEMO_TOPICS[index]
    setIsLoading(true)

    // Add user message
    const userMessage: Message = {
      id: `demo-user-${index}`,
      role: "user",
      content: demo.userMessage,
    }
    setMessages((prev) => [...prev, userMessage])

    try {
      const formData = new FormData()
      formData.append("inputType", "text")
      formData.append("text", demo.userMessage)
      formData.append("topic", demo.topic)
      formData.append("context", "")
      formData.append("messages", JSON.stringify([]))

      const response = await fetch("/api/podcast", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("API request failed")

      const data = await response.json()

      const assistantMessage: Message = {
        id: `demo-assistant-${index}`,
        role: "assistant",
        content: data.aiResponse,
        audioData: data.audioData,
      }
      setMessages((prev) => [...prev, assistantMessage])

      if (data.audioData) {
        playAudioWithCallback(data.audioData, assistantMessage.id, () => {
          // After audio finishes, load next demo
          const nextIndex = index + 1
          setDemoIndex(nextIndex)
          if (nextIndex < DEMO_TOPICS.length) {
            setTimeout(() => loadDemoResponse(nextIndex), 1000)
          }
        })
      }
    } catch (error) {
      console.error("Demo load failed:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const playAudioWithCallback = (audioData: string, messageId: string, onEnd?: () => void) => {
    if (audioRef.current) {
      audioRef.current.pause()
    }

    const audio = new Audio(`data:audio/wav;base64,${audioData}`)
    audioRef.current = audio
    setIsPlaying(true)
    setCurrentPlayingId(messageId)

    audio.onended = () => {
      setIsPlaying(false)
      setCurrentPlayingId(null)
      onEnd?.()
    }

    audio.onerror = () => {
      setIsPlaying(false)
      setCurrentPlayingId(null)
      onEnd?.()
    }

    audio.play().catch((err) => {
      console.error("Auto-play failed:", err)
      setIsPlaying(false)
      setCurrentPlayingId(null)
      onEnd?.()
    })
  }

  const sendToBackend = async (input: { type: "text"; text: string } | { type: "audio"; audioBlob: Blob }) => {
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append("inputType", input.type)
      formData.append("context", uploadedFiles.map((f) => f.content).join("\n\n"))
      formData.append("messages", JSON.stringify(messages))

      if (input.type === "audio") {
        formData.append("audio", input.audioBlob, "recording.webm")
      } else {
        formData.append("text", input.text)
      }

      if (input.type === "text") {
        const userMessage: Message = {
          id: Date.now().toString(),
          role: "user",
          content: input.text,
        }
        setMessages((prev) => [...prev, userMessage])
      }

      const response = await fetch("/api/podcast", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("API request failed")
      }

      const data = await response.json()

      if (input.type === "audio") {
        const userMessage: Message = {
          id: Date.now().toString(),
          role: "user",
          content: data.userInput,
        }
        setMessages((prev) => [...prev, userMessage])
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.aiResponse,
        audioData: data.audioData,
      }
      setMessages((prev) => [...prev, assistantMessage])

      if (data.audioData) {
        playAudioWithCallback(data.audioData, assistantMessage.id)
      }
    } catch (error) {
      console.error("Request failed:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const playAudio = (audioData: string, messageId: string) => {
    playAudioWithCallback(audioData, messageId)
  }

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setIsPlaying(false)
      setCurrentPlayingId(null)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" })
        stream.getTracks().forEach((track) => track.stop())
        await sendToBackend({ type: "audio", audioBlob })
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error("Failed to start recording:", error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return
    const text = input
    setInput("")
    await sendToBackend({ type: "text", text })
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    for (const file of Array.from(files)) {
      const text = await file.text()
      setUploadedFiles((prev) => [...prev, { name: file.name, content: text }])
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const removeFile = (fileName: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.name !== fileName))
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 flex flex-col h-screen">
      <div className="text-center mb-10">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-balance">Podcast +</h1>
        <p className="text-lg md:text-xl text-muted-foreground mt-3">Speak or type to start a conversation</p>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="mb-6 flex flex-wrap gap-2">
          {uploadedFiles.map((file) => (
            <div key={file.name} className="flex items-center gap-2 px-4 py-2 bg-secondary rounded-full text-base">
              <FileText className="h-4 w-4" />
              <span className="max-w-40 truncate">{file.name}</span>
              <button
                onClick={() => removeFile(file.name)}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-5 mb-8">
        {messages.length === 0 && !isLoading && (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <p className="text-lg">Loading demo content...</p>
          </div>
        )}
        {messages.map((message) => (
          <Card
            key={message.id}
            className={cn(
              "p-5",
              message.role === "user" ? "bg-primary text-primary-foreground ml-16" : "bg-muted mr-16",
            )}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <p className="text-sm font-semibold mb-2 uppercase tracking-wide opacity-70">
                  {message.role === "user" ? "You" : "Podcast"}
                </p>
                <p className="text-lg leading-relaxed whitespace-pre-wrap">{message.content}</p>
              </div>
              {message.role === "assistant" && message.audioData && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() =>
                    currentPlayingId === message.id ? stopAudio() : playAudio(message.audioData!, message.id)
                  }
                  className="shrink-0"
                >
                  {currentPlayingId === message.id && isPlaying ? (
                    <Square className="h-5 w-5" />
                  ) : (
                    <Volume2 className="h-5 w-5" />
                  )}
                </Button>
              )}
            </div>
          </Card>
        ))}
        {isLoading && (
          <Card className="p-5 bg-muted mr-16">
            <p className="text-sm font-semibold mb-2 uppercase tracking-wide opacity-70">Podcast</p>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="text-lg">Generating response...</span>
            </div>
          </Card>
        )}
      </div>

      <div className="space-y-4">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="min-h-28 resize-none text-lg"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault()
              handleSubmit()
            }
          }}
        />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant={isRecording ? "destructive" : "outline"}
              size="lg"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isLoading}
              className="gap-2"
            >
              {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
              <span className="hidden sm:inline">{isRecording ? "Stop" : "Voice"}</span>
            </Button>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".txt,.md,.json,.csv"
              multiple
              className="hidden"
            />
            <Button variant="outline" size="lg" onClick={() => fileInputRef.current?.click()} className="gap-2">
              <Upload className="h-5 w-5" />
              <span className="hidden sm:inline">Upload</span>
            </Button>

            {isRecording && <span className="text-base text-destructive animate-pulse font-medium">Recording...</span>}
          </div>

          <Button size="lg" onClick={handleSubmit} disabled={!input.trim() || isLoading} className="px-6">
            {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
            <span className="ml-2 text-base">Send</span>
          </Button>
        </div>
      </div>
    </div>
  )
}
