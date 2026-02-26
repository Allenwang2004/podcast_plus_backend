import { generateText } from "ai"

const DEMO_MODE = true

const DEMO_RESPONSES: Record<string, string> = {
  "us-politics": `Breaking news from Washington D.C. today. The Senate has passed a major infrastructure bill with bipartisan support, allocating $1.2 trillion for roads, bridges, and broadband expansion. President Biden praised the legislation as a "historic investment in America's future." Meanwhile, the House is preparing for heated debates on the upcoming budget proposal, with key negotiations expected to continue through the weekend. Political analysts suggest this could be a pivotal moment for the administration's domestic agenda.`,
  "japan-news": `Good evening, here's the latest from Japan. The Bank of Japan announced it will maintain its ultra-low interest rate policy, keeping rates at negative 0.1 percent. In Tokyo, the government unveiled a new initiative to boost semiconductor manufacturing, with plans to invest over 2 trillion yen in domestic chip production facilities. Meanwhile, Japan's tourism sector continues its strong recovery, with visitor numbers reaching pre-pandemic levels for the first time. The cherry blossom forecast has also been released, predicting early blooming in southern regions this spring.`,
}

export async function POST(req: Request) {
  const formData = await req.formData()

  const inputType = formData.get("inputType") as string
  const context = formData.get("context") as string | null
  const messagesJson = formData.get("messages") as string
  const messages = messagesJson ? JSON.parse(messagesJson) : []
  const topic = formData.get("topic") as string | null

  let userInput = ""

  if (inputType === "audio") {
    const audioFile = formData.get("audio") as Blob
    if (!audioFile) {
      return Response.json({ error: "No audio file provided" }, { status: 400 })
    }

    if (DEMO_MODE) {
      userInput = "This is a demo voice input transcription. In production, this would be your actual spoken words."
    } else {
      const arrayBuffer = await audioFile.arrayBuffer()
      const base64Audio = Buffer.from(arrayBuffer).toString("base64")

      try {
        const { text } = await generateText({
          model: "openai/whisper-1",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "file",
                  data: base64Audio,
                  mimeType: "audio/webm",
                },
              ],
            },
          ],
        })
        userInput = text
      } catch (error) {
        console.error("Transcription error:", error)
        return Response.json({ error: "Transcription failed" }, { status: 500 })
      }
    }
  } else {
    userInput = formData.get("text") as string
    if (!userInput) {
      return Response.json({ error: "No text input provided" }, { status: 400 })
    }
  }

  let aiResponse = ""

  if (DEMO_MODE) {
    if (topic && DEMO_RESPONSES[topic]) {
      aiResponse = DEMO_RESPONSES[topic]
    } else {
      aiResponse = DEMO_RESPONSES["us-politics"]
    }

    if (context && context.trim()) {
      aiResponse = "Based on the documents you provided, here's the latest update. " + aiResponse
    }
  } else {
    const systemPrompt = `You are a professional Podcast host assistant. Your responses should be natural and conversational, like a real podcast conversation.

${context ? `Here are reference materials provided by the user:\n\n${context}` : ""}

Response guidelines:
- Keep the conversation natural and flowing
- Ask follow-up questions to deepen discussion
- Provide insightful perspectives
- Summarize key points when appropriate
- Keep responses concise (2-4 sentences) for natural audio playback`

    try {
      const result = await generateText({
        model: "anthropic/claude-sonnet-4-20250514",
        system: systemPrompt,
        messages: [
          ...messages.map((m: { role: string; content: string }) => ({
            role: m.role,
            content: m.content,
          })),
          { role: "user", content: userInput },
        ],
      })
      aiResponse = result.text
    } catch (error) {
      console.error("AI generation error:", error)
      return Response.json({ error: "AI response generation failed" }, { status: 500 })
    }
  }

  if (DEMO_MODE) {
    const demoAudio = generateDemoAudio(topic === "japan-news" ? "japan" : "us")

    return Response.json({
      userInput,
      aiResponse,
      audioData: demoAudio,
      audioMimeType: "audio/wav",
    })
  }

  try {
    const ttsResponse = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "tts-1",
        input: aiResponse,
        voice: "alloy",
        response_format: "mp3",
      }),
    })

    if (!ttsResponse.ok) {
      throw new Error("TTS API failed")
    }

    const audioBuffer = await ttsResponse.arrayBuffer()
    const audioBase64 = Buffer.from(audioBuffer).toString("base64")

    return Response.json({
      userInput,
      aiResponse,
      audioData: audioBase64,
      audioMimeType: "audio/mp3",
    })
  } catch (error) {
    console.error("TTS error:", error)
    return Response.json({
      userInput,
      aiResponse,
      audioData: null,
      audioMimeType: null,
    })
  }
}

function generateDemoAudio(region: "us" | "japan"): string {
  const sampleRate = 22050
  const duration = 2.5
  const numSamples = Math.floor(sampleRate * duration)

  const buffer = new ArrayBuffer(44 + numSamples * 2)
  const view = new DataView(buffer)

  writeString(view, 0, "RIFF")
  view.setUint32(4, 36 + numSamples * 2, true)
  writeString(view, 8, "WAVE")
  writeString(view, 12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(view, 36, "data")
  view.setUint32(40, numSamples * 2, true)

  const frequencies = region === "japan" ? [440, 554.37, 659.25] : [392, 493.88, 587.33]

  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate
    let sample = 0

    for (const freq of frequencies) {
      sample += Math.sin(2 * Math.PI * freq * t) * 0.25
    }

    const fadeIn = Math.min(1, t * 4)
    const fadeOut = Math.max(0, 1 - (t - duration + 0.5) / 0.5)
    const envelope = fadeIn * fadeOut
    sample *= envelope

    const intSample = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)))
    view.setInt16(44 + i * 2, intSample, true)
  }

  const bytes = new Uint8Array(buffer)
  let binary = ""
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i))
  }
}
