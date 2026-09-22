import { PLUGIN_LLM, PLUGIN_EMBEDDING } from "@qvac/sdk";

// Only the plugins App.js actually calls: llamacpp-completion (chat)
// and embedding (GTE_LARGE_FP16, used for RAG ingest/search).
// Everything else the SDK ships — TTS, Whisper/Parakeet, AudioGen,
// diffusion, OCR, VLA, classification, translation — gets dropped
// from the mobile worker bundle instead of being bare-packed by default.
export default {
  plugins: [PLUGIN_LLM, PLUGIN_EMBEDDING],
};
