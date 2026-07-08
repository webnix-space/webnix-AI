# Webnix AI

**A fully offline, on-device AI assistant for Android.** No cloud. No API keys. No network required. Every model, every computation, and every piece of your data stays on your phone.

Built for the **TETHER Hackathon — QVAC (Local AI) track**.

---

## Overview

Most AI assistants require a live internet connection and send your data to a server you don't control. Webnix AI proves that's not necessary — it runs a real, useful AI assistant experience (chat, file Q&A, private notes) entirely on-device using the [QVAC SDK](https://docs.qvac.ai), with local LLM and embedding models doing all the work on the phone itself.

Built and tested on mid-range Android hardware, not just flagship devices — to show private, offline AI is achievable outside high-end chipsets.

## Features

- **Offline chat** — conversational Q&A powered entirely by an on-device LLM. No network call is ever made during inference.
- **Private file indexing** — pick a file from your device (TXT, MD, JS, PY, CSV, JSON), and it's chunked, embedded, and indexed locally for retrieval-augmented answers.
- **Private notes** — write and save notes; they're indexed the same way as files and searchable in conversation.
- **Grounded answers** — when a query matches indexed content, the assistant answers from your files/notes instead of general knowledge, and labels the source in the UI.
- **Conversation archive** — starting a new chat archives the previous conversation instead of deleting it. Past conversations are browsable and restorable from the History panel.
- **Persistent local storage** — chat, files, notes, and history all persist across app restarts via on-device storage. Nothing is synced anywhere.

## Tech Stack

| Layer | Choice |
|---|---|
| App framework | Expo + React Native |
| On-device AI runtime | [QVAC SDK](https://docs.qvac.ai) |
| Language model | Llama 3.2 1B Instruct (Q4_0 quantized) |
| Embedding model | GTE-Large (FP16) |
| Local persistence | `@react-native-async-storage/async-storage` |
| File access | `expo-document-picker`, `expo-file-system` |
| Build/deploy | EAS Build (custom dev client — see note below) |

## Architecture

```
User query
   │
   ▼
[embedId ready?] ──yes──► ragSearch(query) against indexed files/notes
   │                              │
   no                       match found (score > threshold)?
   │                              │
   ▼                        yes ──┴── no
General knowledge         Context injected      General knowledge
   system prompt          into system prompt        system prompt
   │                              │                      │
   └──────────────┬───────────────┴──────────────────────┘
                  ▼
        completion(modelId: llmId, history, stream: true)
                  │
                  ▼
        Streamed token-by-token response
        rendered in chat UI, <think> tags stripped
```

File/note ingestion pipeline:
```
File/Note text → chunked (400 chars, 80 overlap) → ragIngest(embedId, workspace, chunks) → stored in local RAG workspace
```

## Getting Started

> **Important:** this app uses a native module (`@qvac/sdk/expo-plugin`) and **will not run in Expo Go**. You need a custom development client or a full EAS build.

```bash
git clone <this-repo>
cd webnix-ai
npm install --legacy-peer-deps
```

**Development build (device required):**
```bash
eas build --profile development --platform android
```
Install the resulting APK on your device, then:
```bash
npx expo start --dev-client
```

**Production/preview build:**
```bash
eas build --profile preview --platform android
```

On first launch, the app downloads and loads the LLM and embedding models on-device. This takes a few minutes depending on connection speed for the initial download only — all inference afterward is fully offline.

## Usage

- **Ask AI tab** — chat with the assistant. Suggested prompts appear on first launch.
- **Files tab** — index a file from your device for the assistant to reference.
- **Notes tab** — write and save a note; it's indexed automatically.
- **History** — accessible from the header; browse and restore archived conversations.

## Performance

Measured on-device (CPU only, no GPU acceleration): roughly **8–9 tokens/sec** for the Llama 3.2 1B model on mid-range Snapdragon-class hardware.

## Known Limitations

- Binary/PDF files are not text-extracted — only plain-text formats (TXT, MD, JS, PY, CSV, JSON) are indexed with real content today.
- Inference speed is CPU-bound; no GPU/NPU acceleration path is wired up yet.
- Single active conversation thread at a time (past threads are archived, not run concurrently).

## Hackathon

Submitted to the **TETHER Hackathon**, QVAC (Local AI) track. Built solo, developed primarily on Android hardware.

## License

MIT (or update to match your actual license choice before publishing).
