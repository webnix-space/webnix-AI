import { useState, useEffect, useRef, useCallback } from "react";
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  ScrollView, ActivityIndicator, SafeAreaView, StatusBar,
  Alert, Platform, Dimensions, Share, KeyboardAvoidingView,
  Keyboard, Modal, Switch
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as DocumentPicker from "expo-document-picker";
import * as FileSystem from "expo-file-system/legacy";
import * as Device from "expo-device";
import { useAudioRecorder, RecordingPresets, AudioModule, useAudioRecorderState } from "expo-audio";

// ─── QVAC SDK 0.13.3 ─────────────────────────────────────────
let loadModel, completion, unloadModel, ragIngest, ragSearch,
    ragCloseWorkspace, GTE_LARGE_FP16, LLAMA_3_2_1B_INST_Q4_0,
    QWEN3_600M_INST_Q4, transcribe;
let QVAC_OK = true;

try {
  const sdk = require("@qvac/sdk");
  loadModel             = sdk.loadModel;
  completion            = sdk.completion;
  unloadModel           = sdk.unloadModel;
  ragIngest             = sdk.ragIngest;
  ragSearch             = sdk.ragSearch;
  ragCloseWorkspace     = sdk.ragCloseWorkspace;
  GTE_LARGE_FP16        = sdk.GTE_LARGE_FP16;
  LLAMA_3_2_1B_INST_Q4_0 = sdk.LLAMA_3_2_1B_INST_Q4_0;
  QWEN3_600M_INST_Q4    = sdk.QWEN3_600M_INST_Q4;
  transcribe            = sdk.transcribe;
  console.log("[QVAC] exports:", Object.keys(sdk).join(", "));
} catch (e) {
  console.error("[QVAC] load failed:", e);
  QVAC_OK = false;
}

// ─── Model definitions ────────────────────────────────────────
const MODELS = [
  {
    id:       "llama",
    label:    "Llama 3.2 1B",
    desc:     "Balanced • ~800MB",
    src:      LLAMA_3_2_1B_INST_Q4_0 ||
              "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  },
  {
    id:       "qwen",
    label:    "Qwen3 0.6B",
    desc:     "Fastest • ~400MB • Low RAM",
    src:      QWEN3_600M_INST_Q4 ||
              "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0_6b-q4_k_m.gguf",
  },
];

const RAG_WORKSPACE = "webnix-rag";
const STORAGE_KEYS = {
  CHAT:    "webnix_chat_v4",
  FILES:   "webnix_files_v3",
  NOTES:   "webnix_notes_v3",
  HISTORY: "webnix_chat_history_v1",
  MODEL:   "webnix_model_v1",
};

const { width } = Dimensions.get("window");
const TABS = { ASK: "ask", FILES: "files", NOTES: "notes" };
const IS_IOS = Platform.OS === "ios";
const ANDROID_STATUSBAR_HEIGHT = Platform.OS === "android"
  ? (StatusBar.currentHeight || 24) : 0;

// ─── MCP Tools ────────────────────────────────────────────────
const MCP_TOOLS = [
  {
    name: "calculator",
    description: "Evaluate a math expression. Use for any arithmetic, percentage, or unit calculation.",
    parameters: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Math expression to evaluate e.g. '(15 * 8) / 3 + 12'" }
      },
      required: ["expression"]
    }
  },
  {
    name: "get_datetime",
    description: "Get the current date, time, day of week, or timezone.",
    parameters: {
      type: "object",
      properties: {
        format: { type: "string", description: "What to return: 'full', 'date', 'time', 'day'" }
      },
      required: ["format"]
    }
  },
  {
    name: "create_note",
    description: "Create and save a note with a title and content.",
    parameters: {
      type: "object",
      properties: {
        title:   { type: "string", description: "Note title" },
        content: { type: "string", description: "Note content" }
      },
      required: ["title", "content"]
    }
  },
  {
    name: "word_counter",
    description: "Count words, characters, or sentences in a given text.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "Text to analyze" }
      },
      required: ["text"]
    }
  }
];

// ─── Tool executor ────────────────────────────────────────────
const executeTool = async (name, args, saveNoteFn) => {
  try {
    switch (name) {
      case "calculator": {
        const expr = String(args.expression || "").replace(/[^0-9+\-*/().%\s]/g, "");
        if (!expr) return "Invalid expression";
        // eslint-disable-next-line no-eval
        const result = Function(`"use strict"; return (${expr})`)();
        return `${args.expression} = ${result}`;
      }
      case "get_datetime": {
        const now = new Date();
        const fmt = args.format || "full";
        if (fmt === "date") return now.toLocaleDateString();
        if (fmt === "time") return now.toLocaleTimeString();
        if (fmt === "day")  return now.toLocaleDateString(undefined, { weekday: "long" });
        return now.toLocaleString();
      }
      case "create_note": {
        await saveNoteFn(args.title, args.content);
        return `Note "${args.title}" saved and indexed successfully.`;
      }
      case "word_counter": {
        const text = args.text || "";
        const words = text.trim().split(/\s+/).filter(Boolean).length;
        const chars = text.length;
        const sentences = text.split(/[.!?]+/).filter(Boolean).length;
        return `Words: ${words} | Characters: ${chars} | Sentences: ${sentences}`;
      }
      default:
        return `Unknown tool: ${name}`;
    }
  } catch (e) {
    return `Tool error: ${e.message}`;
  }
};

// ─── Helpers ──────────────────────────────────────────────────
const chunkText = (text, size = 400, overlap = 80) => {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    chunks.push(text.slice(i, i + size));
    i += size - overlap;
  }
  return chunks;
};

const readFile = async (uri, mimeType, name) => {
  try {
    if (
      mimeType === "text/plain" ||
      ["txt","md","csv","json","js","py","ts","html","css","xml"].some(e => name.endsWith("." + e))
    ) {
      return await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.UTF8 });
    }
    return `File: ${name}\nType: ${mimeType}\nNote: Binary file — AI reasons from filename and type only.`;
  } catch (e) {
    throw new Error("Cannot read: " + e.message);
  }
};

const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

const save = async (key, val) => {
  try { await AsyncStorage.setItem(key, JSON.stringify(val)); } catch {}
};
const load = async (key, fallback) => {
  try {
    const v = await AsyncStorage.getItem(key);
    return v ? JSON.parse(v) : fallback;
  } catch { return fallback; }
};

// ═════════════════════════════════════════════════════════════
export default function App() {
  // Model state
  const [selectedModelId, setSelectedModelId] = useState("llama");
  const [llmId,      setLlmId]      = useState(null);
  const [embedId,    setEmbedId]    = useState(null);
  const [whisperModelId, setWhisperModelId] = useState(null);
  const [bootState,  setBootState]  = useState("idle");
  const [progress,   setProgress]   = useState(0);
  const [bootMsg,    setBootMsg]    = useState("");

  // UI
  const [tab,        setTab]        = useState(TABS.ASK);
  const [logs,       setLogs]       = useState([]);
  const [showSettings, setShowSettings] = useState(false);

  // Persisted state
  const [files,      setFiles]      = useState([]);
  const [notes,      setNotes]      = useState([]);
  const [chat,       setChat]       = useState([]);
  const [history,    setHistory]    = useState([]);
  const [showHistory,setShowHistory]= useState(false);

  // Files UI
  const [indexing,   setIndexing]   = useState(false);
  const [curFile,    setCurFile]    = useState(null);
  const [viewingFile, setViewingFile] = useState(null);

  // Notes UI
  const [noteTitle,  setNoteTitle]  = useState("");
  const [noteBody,   setNoteBody]   = useState("");
  const [savingNote, setSavingNote] = useState(false);

  // Chat UI
  const [query,      setQuery]      = useState("");
  const [thinking,   setThinking]   = useState(false);
  const [tps,        setTps]        = useState(null);

  // Voice — expo-audio hook-based recorder (expo-av is incompatible with SDK 56)
  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(audioRecorder);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const isRecording = recorderState.isRecording;

  const scrollRef = useRef(null);
  const inputRef  = useRef(null);
  const llmIdRef   = useRef(null);
  const embedIdRef = useRef(null);
  useEffect(() => { llmIdRef.current = llmId; },   [llmId]);
  useEffect(() => { embedIdRef.current = embedId; }, [embedId]);

  const log = useCallback((msg) => {
    const line = `${new Date().toLocaleTimeString()} ${msg}`;
    setLogs(p => [...p.slice(-40), line]);
    console.log("[WebnixAI]", msg);
  }, []);

  // ── Load persisted data ────────────────────────────────────
  useEffect(() => {
    (async () => {
      const [c, f, n, h, m] = await Promise.all([
        load(STORAGE_KEYS.CHAT,    []),
        load(STORAGE_KEYS.FILES,   []),
        load(STORAGE_KEYS.NOTES,   []),
        load(STORAGE_KEYS.HISTORY, []),
        load(STORAGE_KEYS.MODEL,   "llama"),
      ]);
      setChat(c);
      setFiles(f);
      setNotes(n);
      setHistory(h);
      setSelectedModelId(m);
      log(`Restored: ${c.length} msgs, ${f.length} files, ${n.length} notes`);
    })();
  }, []);

  // ── Auto-scroll ────────────────────────────────────────────
  useEffect(() => {
    if (chat.length > 0)
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
  }, [chat]);

  // ── Boot ───────────────────────────────────────────────────
  useEffect(() => {
    if (!QVAC_OK) { setBootState("error"); setBootMsg("QVAC SDK missing"); return; }
    boot("llama");
    return () => {
      if (llmIdRef.current)   unloadModel({ modelId: llmIdRef.current,   clearStorage: false }).catch(() => {});
      if (embedIdRef.current) unloadModel({ modelId: embedIdRef.current, clearStorage: false }).catch(() => {});
      ragCloseWorkspace?.({ workspace: RAG_WORKSPACE, deleteOnClose: false }).catch(() => {});
    };
  }, []);

  const boot = async (modelId) => {
    setBootState("loading");
    setProgress(0);
    try {
      const model = MODELS.find(m => m.id === modelId) || MODELS[0];

      setBootMsg(`Loading ${model.label}...`);
      log(`Loading ${model.label}...`);
      const lId = await loadModel({
        modelSrc:    model.src,
        modelType:   "llamacpp-completion",
        modelConfig: { device: "cpu", ctx_size: 2048 },
        onProgress:  (p) => setProgress(Math.round((p.percentage || p * 100 || 0) * 0.6)),
      });
      setLlmId(lId);
      llmIdRef.current = lId;
      log(`${model.label} ready`);

      setBootMsg("Loading embedding model...");
      log("Loading GTE-Large...");
      const eId = await loadModel({
        modelSrc:   GTE_LARGE_FP16,
        onProgress: (p) => setProgress(60 + Math.round((p.percentage || p * 100 || 0) * 0.4)),
      });
      setEmbedId(eId);
      embedIdRef.current = eId;
      log("Embedder ready");

      setBootState("ready");
      setProgress(100);
      setBootMsg("");
      log(`All models active — ${model.label}`);
    } catch (e) {
      log("Boot failed: " + (e?.message || e));
      setBootState("error");
      setBootMsg(String(e?.message || e));
    }
  };

  // ── Switch model ───────────────────────────────────────────
  const switchModel = async (newModelId) => {
    if (newModelId === selectedModelId) { setShowSettings(false); return; }
    Alert.alert(
      "Switch Model",
      `Switch to ${MODELS.find(m => m.id === newModelId)?.label}? Current chat will be saved.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Switch",
          onPress: async () => {
            setShowSettings(false);
            if (chat.length > 0) newChat(false);
            if (llmIdRef.current) {
              await unloadModel({ modelId: llmIdRef.current, clearStorage: false }).catch(() => {});
              setLlmId(null); llmIdRef.current = null;
            }
            setSelectedModelId(newModelId);
            save(STORAGE_KEYS.MODEL, newModelId);
            await boot(newModelId);
          }
        }
      ]
    );
  };

  // ── Persist helpers ────────────────────────────────────────
  const updateChat = useCallback((updated) => {
    setChat(updated);
    const clean = updated
      .filter(m => !m.streaming && m.content && m.content.length > 0)
      .map(({ streaming, ...rest }) => rest);
    save(STORAGE_KEYS.CHAT, clean);
  }, []);

  const updateFiles = useCallback((updated) => {
    setFiles(updated);
    save(STORAGE_KEYS.FILES, updated);
  }, []);

  const updateNotes = useCallback((updated) => {
    setNotes(updated);
    save(STORAGE_KEYS.NOTES, updated);
  }, []);

  // ── New chat / history ─────────────────────────────────────
  const newChat = (confirm = true) => {
    const doNew = () => {
      if (chat.length > 0) {
        const firstMsg = chat.find(m => m.role === "user");
        const title = firstMsg
          ? firstMsg.content.slice(0, 50) + (firstMsg.content.length > 50 ? "..." : "")
          : "Conversation";
        const archived = {
          id: Date.now().toString(),
          title,
          messages: chat.filter(m => m.content),
          savedAt: new Date().toLocaleString(),
        };
        const updated = [archived, ...history].slice(0, 20);
        setHistory(updated);
        save(STORAGE_KEYS.HISTORY, updated);
      }
      updateChat([]);
      setTps(null);
      log("New chat started");
    };

    if (!confirm) { doNew(); return; }
    if (chat.length === 0) return;
    Alert.alert("New Chat", "Archive current conversation and start fresh?", [
      { text: "Cancel", style: "cancel" },
      { text: "Archive & New", onPress: doNew },
    ]);
  };

  const restoreChat = (archived) => {
    updateChat(archived.messages);
    setShowHistory(false);
    log(`Restored: ${archived.title}`);
  };

  const deleteArchived = (id) => {
    Alert.alert("Delete", "Remove this conversation?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete", style: "destructive",
        onPress: () => {
          const updated = history.filter(h => h.id !== id);
          setHistory(updated);
          save(STORAGE_KEYS.HISTORY, updated);
        }
      }
    ]);
  };

  // ── Save note (called by tool + UI) ───────────────────────
  const saveNoteInternal = async (title, content) => {
    if (!embedId) throw new Error("Embedder not ready");
    const chunks = chunkText(`Title: ${title}\n\n${content}`);
    await ragIngest({ modelId: embedId, workspace: RAG_WORKSPACE, documents: chunks, chunk: false });
    const newNote = { title, preview: content.slice(0, 80), at: new Date().toLocaleTimeString() };
    updateNotes([...notes, newNote]);
    log(`✓ Note: ${title}`);
    return newNote;
  };

  // ── Index file ─────────────────────────────────────────────
  const pickFile = async () => {
    if (!embedId) return Alert.alert("Not Ready", "Models still loading.");
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: ["text/*", "application/pdf", "application/json", "*/*"],
        copyToCacheDirectory: true,
      });
      if (res.canceled) return;

      const file = res.assets[0];
      setCurFile(file.name); setIndexing(true);
      log("Indexing: " + file.name);

      const text   = await readFile(file.uri, file.mimeType, file.name);
      const chunks = chunkText(text);
      const result = await ragIngest({ modelId: embedId, workspace: RAG_WORKSPACE, documents: chunks, chunk: false });

      const newFile = {
        name:    file.name,
        chunks:  result.processed?.length || chunks.length,
        at:      new Date().toLocaleTimeString(),
        preview: text.slice(0, 2000), // stored for in-app viewing, not re-sent to model
        size:    text.length,
      };
      updateFiles([...files, newFile]);
      log(`✓ Indexed: ${file.name}`);
      Alert.alert("Done", `"${file.name}" ready to query.`);
    } catch (e) {
      log("Index failed: " + (e?.message || e));
      Alert.alert("Error", String(e?.message || e));
    } finally {
      setIndexing(false); setCurFile(null);
    }
  };

  // ── Save note UI ───────────────────────────────────────────
  const saveNote = async () => {
    if (!noteBody.trim()) return Alert.alert("Empty", "Write something.");
    if (!embedId)         return Alert.alert("Not Ready", "Models loading.");
    setSavingNote(true);
    const title = noteTitle.trim() || `Note ${notes.length + 1}`;
    try {
      await saveNoteInternal(title, noteBody.trim());
      setNoteTitle(""); setNoteBody("");
      Alert.alert("Saved", `"${title}" indexed.`);
    } catch (e) {
      log("Save failed: " + (e?.message || e));
      Alert.alert("Error", String(e?.message || e));
    } finally { setSavingNote(false); }
  };

  // ── Voice recording (expo-audio) ───────────────────────────
  const startRecording = async () => {
    try {
      const { granted } = await AudioModule.requestRecordingPermissionsAsync();
      if (!granted) return Alert.alert("Permission denied", "Microphone access is required.");

      await AudioModule.setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });

      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();
      log("Recording started");
    } catch (e) {
      log("Record failed: " + (e?.message || e));
      Alert.alert("Error", "Could not start recording: " + (e?.message || e));
    }
  };

  const stopRecording = async () => {
    setIsTranscribing(true);
    try {
      await audioRecorder.stop();
      const uri = audioRecorder.uri;
      log("Recording stopped, transcribing...");

      if (transcribe && llmId && uri) {
        // Use QVAC Whisper transcription
        const result = await transcribe({
          modelId:   llmId,
          audioPath: uri,
          language:  "en",
        });
        const text = result?.text?.trim() || "";
        if (text) {
          setQuery(prev => (prev ? prev + " " : "") + text);
          log(`Transcribed: ${text.slice(0, 50)}`);
        } else {
          Alert.alert("Nothing detected", "Could not transcribe audio. Try speaking more clearly.");
        }
      } else {
        // Fallback — inform user to type
        Alert.alert("Transcription unavailable", "Voice transcription requires Whisper model. Type your message instead.");
      }
    } catch (e) {
      log("Transcribe failed: " + (e?.message || e));
      Alert.alert("Error", "Transcription failed: " + (e?.message || e));
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleMicPress = () => {
    if (isRecording) stopRecording();
    else startRecording();
  };

  // ── Ask AI with MCP tool support ──────────────────────────
  const ask = async () => {
    const q = query.trim();
    if (!q || !llmId) return;
    Keyboard.dismiss();

    const ts = new Date().toLocaleTimeString();
    const userMsg = { role: "user", content: q, ts };
    const withUser = [...chat, userMsg];
    const aiMsg   = { role: "assistant", content: "", ts, streaming: true, source: "general knowledge" };
    const withAi  = [...withUser, aiMsg];

    setChat(withAi);
    setQuery(""); setThinking(true); setTps(null);

    try {
      // RAG retrieval
      let context = "", source = "general knowledge";
      if (embedId && (files.length > 0 || notes.length > 0)) {
        try {
          const results = await ragSearch({ modelId: embedId, workspace: RAG_WORKSPACE, query: q, topK: 4 });
          if (results?.length && results[0].score > 0.35) {
            context = results.map((r, i) => `[${i+1}] ${r.content}`).join("\n\n");
            source  = "your files & notes";
            log(`RAG: ${results.length} matches (score: ${results[0].score.toFixed(2)})`);
          }
        } catch (e) { log("RAG skip: " + (e?.message || e)); }
      }

      const model = MODELS.find(m => m.id === selectedModelId) || MODELS[0];
      const systemPrompt = context
        ? `You are Webnix AI, a private personal assistant running on this device (${model.label}). Use the indexed content below to answer accurately. Be concise.\n\nIndexed content:\n${context}`
        : `You are Webnix AI, a private personal assistant running entirely on this device (${model.label}). You have access to tools for calculations, datetime, notes, and text analysis. Be concise and helpful.`;

      const historyMsgs = chat
        .filter(m => !m.streaming && m.content)
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }));

      const history = [
        { role: "user", content: systemPrompt },
        ...historyMsgs,
        { role: "user", content: q },
      ];

      const run = completion({
        modelId: llmId,
        history,
        stream:  true,
        tools:   MCP_TOOLS,
      });

      let raw = "", tokens = 0;
      const t0 = Date.now();

      for await (const token of run.tokenStream) {
        raw += token;
        tokens++;
        const hasOpenThink = raw.includes("<think>") && !raw.includes("</think>");
        const clean = hasOpenThink ? "" : stripThink(raw);
        const elapsed = (Date.now() - t0) / 1000;
        if (elapsed > 0.3) setTps((tokens / elapsed).toFixed(1));
        setChat(prev => {
          const next = [...prev];
          next[next.length - 1] = { ...next[next.length - 1], content: clean, source };
          return next;
        });
      }

      // Handle tool calls if any
      const toolCalls = await run.toolCalls?.catch(() => []) || [];
      let finalContent = stripThink(raw);

      if (toolCalls.length > 0) {
        log(`Tool calls: ${toolCalls.map(t => t.name).join(", ")}`);
        const toolResults = [];
        for (const call of toolCalls) {
          const result = await executeTool(call.name, call.arguments || call.input || {}, saveNoteInternal);
          toolResults.push(`[${call.name}]: ${result}`);
          log(`Tool ${call.name}: ${result}`);
        }
        // Append tool results to answer
        if (toolResults.length > 0) {
          finalContent = finalContent
            ? finalContent + "\n\n" + toolResults.join("\n")
            : toolResults.join("\n");
        }
      }

      const elapsed = (Date.now() - t0) / 1000;
      const finalTps = elapsed > 0 ? (tokens / elapsed).toFixed(1) : "?";
      setTps(finalTps);

      const done = [...withAi];
      done[done.length - 1] = { role: "assistant", content: finalContent, ts, streaming: false, source };
      updateChat(done);
      log(`Done — ${tokens} tok, ${finalTps} tok/s`);
    } catch (e) {
      log("Ask failed: " + (e?.message || e));
      const err = [...withAi];
      err[err.length - 1] = { role: "assistant", content: "Error: " + (e?.message || e), ts, streaming: false };
      updateChat(err);
    } finally { setThinking(false); }
  };

  // ── Share chat ─────────────────────────────────────────────
  const shareChat = async () => {
    if (!chat.length) return;
    const text = chat
      .filter(m => m.content)
      .map(m => `[${m.ts}] ${m.role === "user" ? "You" : "Webnix AI"}: ${m.content}`)
      .join("\n\n");
    try { await Share.share({ message: text, title: "Webnix AI Chat" }); }
    catch (e) { Alert.alert("Error", String(e?.message || e)); }
  };

  // ── Boot screens ───────────────────────────────────────────
  const currentModel = MODELS.find(m => m.id === selectedModelId) || MODELS[0];

  if (bootState === "idle" || bootState === "loading") {
    return (
      <View style={s.boot}>
        <StatusBar barStyle="light-content" backgroundColor="#050505" />
        <Text style={s.bootIcon}>◈</Text>
        <Text style={s.bootTitle}>Webnix AI</Text>
        <Text style={s.bootSub}>Private Intelligence. On Device.</Text>
        <View style={s.progressTrack}>
          <View style={[s.progressFill, { width: `${progress}%` }]} />
        </View>
        <Text style={s.bootMsg}>{bootMsg || "Initializing..."}</Text>
        <Text style={s.bootPct}>{progress > 0 ? `${progress}%` : ""}</Text>
        <Text style={s.bootModelLabel}>{currentModel.label}</Text>
        <Text style={s.bootDevice}>{Device.modelName || "Device"} · Local Silicon</Text>
      </View>
    );
  }

  if (bootState === "error") {
    return (
      <View style={s.boot}>
        <StatusBar barStyle="light-content" backgroundColor="#050505" />
        <Text style={[s.bootIcon, { color: "#ff1744" }]}>✕</Text>
        <Text style={s.bootTitle}>Failed to Start</Text>
        <Text style={s.bootMsg}>{bootMsg}</Text>
        <TouchableOpacity style={s.retryBtn} onPress={() => boot(selectedModelId)}>
          <Text style={s.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Main UI ────────────────────────────────────────────────
  return (
    <SafeAreaView style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor="#050505" />

      {/* Header */}
      <View style={s.header}>
        <View style={s.headerLeft}>
          <Text style={s.headerTitle}>◈ Webnix AI</Text>
          <Text style={s.headerSub}>{currentModel.label} · On Device</Text>
        </View>
        <View style={s.headerRight}>
          <View style={s.badge}>
            <View style={s.dot} />
            <Text style={s.badgeText}>Online</Text>
          </View>
          <TouchableOpacity style={s.iconBtn} onPress={() => setShowSettings(true)}>
            <Text style={s.iconBtnText}>⚙</Text>
          </TouchableOpacity>
          {tab === TABS.ASK && (
            <TouchableOpacity style={s.iconBtn} onPress={() => setShowHistory(true)}>
              <Text style={s.iconBtnText}>◷</Text>
            </TouchableOpacity>
          )}
          {tab === TABS.ASK && chat.length > 0 && (
            <TouchableOpacity style={s.newChatBtn} onPress={() => newChat(true)}>
              <Text style={s.newChatText}>+ New</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      {/* Tabs */}
      <View style={s.tabBar}>
        {[
          { key: TABS.ASK,   label: "Ask AI" },
          { key: TABS.FILES, label: `Files${files.length > 0 ? ` (${files.length})` : ""}` },
          { key: TABS.NOTES, label: `Notes${notes.length > 0 ? ` (${notes.length})` : ""}` },
        ].map(t => (
          <TouchableOpacity
            key={t.key}
            style={[s.tab, tab === t.key && s.tabActive]}
            onPress={() => setTab(t.key)}
          >
            <Text style={[s.tabText, tab === t.key && s.tabTextActive]}>{t.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* ── ASK TAB ── */}
      {tab === TABS.ASK && (
        <KeyboardAvoidingView style={{ flex: 1 }} behavior={IS_IOS ? "padding" : "height"}>
          <ScrollView
            ref={scrollRef}
            style={s.chatScroll}
            contentContainerStyle={s.chatInner}
            keyboardShouldPersistTaps="handled"
            onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: false })}
          >
            {chat.length === 0 && (
              <View style={s.emptyChat}>
                <Text style={s.emptyChatIcon}>◈</Text>
                <Text style={s.emptyChatTitle}>Webnix AI</Text>
                <Text style={s.emptyChatSub}>
                  {files.length === 0 && notes.length === 0
                    ? "Ask anything. Index files or notes to ground answers in your content."
                    : `${files.length} file(s) + ${notes.length} note(s) indexed and ready.`}
                </Text>
                <View style={s.chips}>
                  {[
                    "What can you do?",
                    "Calculate 15% of 2400",
                    "What time is it?",
                    "Summarize my files",
                  ].map(c => (
                    <TouchableOpacity key={c} style={s.chip} onPress={() => setQuery(c)}>
                      <Text style={s.chipText}>{c}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {/* MCP Tools hint */}
                <View style={s.toolsHint}>
                  <Text style={s.toolsHintTitle}>Built-in Tools</Text>
                  <Text style={s.toolsHintText}>🧮 Calculator  🕐 DateTime  📝 Note Creator  📊 Word Counter</Text>
                </View>
              </View>
            )}

            {chat.map((m, i) => (
              <View key={i} style={[s.bubble, m.role === "user" ? s.bubbleUser : s.bubbleAI]}>
                {m.role === "assistant" && (
                  <Text style={s.bubbleLabel}>
                    WEBNIX AI{m.source && m.source !== "general knowledge" ? ` · ${m.source.toUpperCase()}` : ""}{m.streaming ? " ▌" : ""}
                  </Text>
                )}
                <Text style={[s.bubbleText, m.role === "user" && s.bubbleTextUser]}>
                  {m.content || (m.streaming ? "..." : "")}
                </Text>
                <Text style={[s.bubbleTs, m.role === "user" && { color: "rgba(0,0,0,0.35)" }]}>{m.ts}</Text>
              </View>
            ))}

            {tps && !thinking && (
              <View style={s.tpsBadge}>
                <Text style={s.tpsText}>{tps} tok/s · {currentModel.label}</Text>
              </View>
            )}
          </ScrollView>

          {chat.length > 0 && (
            <View style={s.chatActions}>
              <TouchableOpacity style={s.actionBtn} onPress={shareChat}>
                <Text style={s.actionBtnText}>↓ Save Chat</Text>
              </TouchableOpacity>
              <TouchableOpacity style={s.actionBtn} onPress={() => newChat(true)}>
                <Text style={s.actionBtnText}>+ New Chat</Text>
              </TouchableOpacity>
            </View>
          )}

          <View style={s.inputRow}>
            <TextInput
              ref={inputRef}
              style={s.input}
              placeholder="Ask Webnix AI..."
              placeholderTextColor="#444"
              value={query}
              onChangeText={setQuery}
              multiline
              maxHeight={90}
              returnKeyType="send"
              onSubmitEditing={ask}
              blurOnSubmit={false}
            />
            <TouchableOpacity
              style={[s.micBtn, isRecording && s.micBtnActive, isTranscribing && s.btnOff]}
              onPress={handleMicPress}
              disabled={isTranscribing}
            >
              {isTranscribing
                ? <ActivityIndicator size="small" color="#00FFCC" />
                : <Text style={s.micIcon}>{isRecording ? "⏹" : "🎤"}</Text>
              }
            </TouchableOpacity>
            <TouchableOpacity
              style={[s.sendBtn, (thinking || !query.trim()) && s.sendBtnOff]}
              onPress={ask}
              disabled={thinking || !query.trim()}
            >
              {thinking
                ? <ActivityIndicator size="small" color="#000" />
                : <Text style={s.sendIcon}>↑</Text>
              }
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      )}

      {/* ── FILES TAB ── */}
      {tab === TABS.FILES && (
        <ScrollView style={s.scroll} contentContainerStyle={s.scrollInner}>
          <TouchableOpacity
            style={[s.primaryBtn, indexing && s.btnOff]}
            onPress={pickFile}
            disabled={indexing}
          >
            {indexing ? (
              <View style={s.row}>
                <ActivityIndicator size="small" color="#000" />
                <Text style={[s.primaryBtnText, { marginLeft: 8 }]}>
                  Indexing {curFile ? `"${curFile.slice(0, 20)}..."` : "..."}
                </Text>
              </View>
            ) : (
              <Text style={s.primaryBtnText}>+ Pick File to Index</Text>
            )}
          </TouchableOpacity>
          <Text style={s.hint}>TXT, MD, JS, PY, CSV, JSON, PDF — processed on device.</Text>

          {files.length === 0 ? (
            <View style={s.empty}>
              <Text style={s.emptyIcon2}>▣</Text>
              <Text style={s.emptyTitle2}>No files indexed</Text>
              <Text style={s.emptyDesc}>Pick a file — AI reads and indexes it for Q&A.</Text>
            </View>
          ) : files.map((f, i) => (
            <TouchableOpacity key={i} style={s.card} onPress={() => setViewingFile(f)}>
              <View style={s.cardLeft}>
                <Text style={s.cardIcon}>◻</Text>
                <View style={{ flex: 1 }}>
                  <Text style={s.cardName} numberOfLines={1}>{f.name}</Text>
                  <Text style={s.cardMeta}>{f.chunks} chunks · {f.at} · tap to view</Text>
                </View>
              </View>
              <View style={s.cardBadge}><Text style={s.cardBadgeText}>Indexed</Text></View>
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}

      {/* ── FILE PREVIEW MODAL ── */}
      <Modal visible={!!viewingFile} animationType="slide" transparent onRequestClose={() => setViewingFile(null)}>
        <View style={s.modalOverlay}>
          <View style={s.modalSheet}>
            <View style={s.modalHeader}>
              <Text style={s.modalTitle} numberOfLines={1}>{viewingFile?.name}</Text>
              <TouchableOpacity onPress={() => setViewingFile(null)}>
                <Text style={s.modalClose}>Close</Text>
              </TouchableOpacity>
            </View>
            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16 }}>
              <View style={s.infoCard}>
                <Text style={s.infoText}>Indexed: {viewingFile?.at}</Text>
                <Text style={s.infoText}>Chunks: {viewingFile?.chunks}</Text>
                <Text style={s.infoText}>Size: {viewingFile?.size ? `${viewingFile.size} chars` : "unknown"}</Text>
              </View>
              <Text style={[s.sectionLabel, { marginTop: 16 }]}>
                CONTENT {viewingFile?.preview?.length >= 2000 ? "(first 2000 chars)" : ""}
              </Text>
              <View style={s.previewBox}>
                <Text style={s.previewText}>
                  {viewingFile?.preview || "No preview available for this file type."}
                </Text>
              </View>
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* ── NOTES TAB ── */}
      {tab === TABS.NOTES && (
        <KeyboardAvoidingView style={{ flex: 1 }} behavior={IS_IOS ? "padding" : "height"}>
          <ScrollView style={s.scroll} contentContainerStyle={s.scrollInner} keyboardShouldPersistTaps="handled">
            <TextInput
              style={s.titleInput}
              placeholder="Note title (optional)"
              placeholderTextColor="#444"
              value={noteTitle}
              onChangeText={setNoteTitle}
            />
            <TextInput
              style={s.bodyInput}
              placeholder="Write your note — thoughts, research, meeting notes. AI indexes and searches it."
              placeholderTextColor="#444"
              value={noteBody}
              onChangeText={setNoteBody}
              multiline
              textAlignVertical="top"
            />
            <TouchableOpacity
              style={[s.primaryBtn, savingNote && s.btnOff]}
              onPress={saveNote}
              disabled={savingNote}
            >
              {savingNote ? (
                <View style={s.row}>
                  <ActivityIndicator size="small" color="#000" />
                  <Text style={[s.primaryBtnText, { marginLeft: 8 }]}>Saving...</Text>
                </View>
              ) : (
                <Text style={s.primaryBtnText}>Save & Index Note</Text>
              )}
            </TouchableOpacity>

            {notes.length === 0 ? (
              <View style={s.empty}>
                <Text style={s.emptyIcon2}>◎</Text>
                <Text style={s.emptyTitle2}>No notes yet</Text>
                <Text style={s.emptyDesc}>Save a note — AI will find it when you ask.</Text>
              </View>
            ) : notes.map((n, i) => (
              <View key={i} style={s.card}>
                <View style={s.cardLeft}>
                  <Text style={s.cardIcon}>◎</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={s.cardName}>{n.title}</Text>
                    <Text style={s.cardMeta} numberOfLines={2}>{n.preview}</Text>
                  </View>
                </View>
                <Text style={s.cardMeta}>{n.at}</Text>
              </View>
            ))}

            {logs.length > 0 && (
              <View style={[s.logCard, { marginTop: 20 }]}>
                <Text style={s.logLabel}>ACTIVITY</Text>
                {logs.slice(-8).map((l, i) => (
                  <Text key={i} style={s.logLine}>{l}</Text>
                ))}
              </View>
            )}
          </ScrollView>
        </KeyboardAvoidingView>
      )}

      {/* ── SETTINGS MODAL ── */}
      <Modal visible={showSettings} animationType="slide" transparent>
        <View style={s.modalOverlay}>
          <View style={s.modalSheet}>
            <View style={s.modalHeader}>
              <Text style={s.modalTitle}>Settings</Text>
              <TouchableOpacity onPress={() => setShowSettings(false)}>
                <Text style={s.modalClose}>Done</Text>
              </TouchableOpacity>
            </View>

            <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: 16 }}>
              {/* Model picker */}
              <Text style={s.sectionLabel}>AI MODEL</Text>
              {MODELS.map(m => (
                <TouchableOpacity
                  key={m.id}
                  style={[s.modelCard, selectedModelId === m.id && s.modelCardActive]}
                  onPress={() => switchModel(m.id)}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={[s.modelName, selectedModelId === m.id && { color: "#000" }]}>{m.label}</Text>
                    <Text style={[s.modelDesc, selectedModelId === m.id && { color: "#000" }]}>{m.desc}</Text>
                  </View>
                  {selectedModelId === m.id && (
                    <Text style={s.modelCheck}>✓</Text>
                  )}
                </TouchableOpacity>
              ))}

              {/* MCP Tools */}
              <Text style={[s.sectionLabel, { marginTop: 24 }]}>MCP TOOLS (ALWAYS ON)</Text>
              {MCP_TOOLS.map(t => (
                <View key={t.name} style={s.toolCard}>
                  <View style={{ flex: 1 }}>
                    <Text style={s.toolName}>{t.name}</Text>
                    <Text style={s.toolDesc}>{t.description}</Text>
                  </View>
                  <Text style={s.toolActive}>✓</Text>
                </View>
              ))}

              {/* Device info */}
              <Text style={[s.sectionLabel, { marginTop: 24 }]}>DEVICE</Text>
              <View style={s.infoCard}>
                <Text style={s.infoText}>Device: {Device.modelName || "Unknown"}</Text>
                <Text style={s.infoText}>Platform: Android {Device.osVersion}</Text>
                <Text style={s.infoText}>SDK: @qvac/sdk 0.13.3</Text>
                <Text style={s.infoText}>Files: {files.length} indexed</Text>
                <Text style={s.infoText}>Notes: {notes.length} indexed</Text>
                <Text style={s.infoText}>Chats archived: {history.length}</Text>
              </View>

              {/* Danger zone */}
              <Text style={[s.sectionLabel, { marginTop: 24, color: "#ff5252" }]}>DATA</Text>
              <TouchableOpacity
                style={s.dangerBtn}
                onPress={() => Alert.alert("Clear All Data", "This will delete all files, notes, and chat history. Cannot be undone.", [
                  { text: "Cancel", style: "cancel" },
                  { text: "Clear", style: "destructive", onPress: async () => {
                    await Promise.all(Object.values(STORAGE_KEYS).map(k => AsyncStorage.removeItem(k)));
                    setChat([]); setFiles([]); setNotes([]); setHistory([]);
                    setShowSettings(false);
                    log("All data cleared");
                  }}
                ])}
              >
                <Text style={s.dangerBtnText}>Clear All Data</Text>
              </TouchableOpacity>
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* ── HISTORY OVERLAY ── */}
      {showHistory && (
        <View style={s.historyOverlay}>
          <View style={s.historyHeader}>
            <Text style={s.historyTitle}>Archived Conversations</Text>
            <TouchableOpacity onPress={() => setShowHistory(false)}>
              <Text style={s.historyClose}>Close</Text>
            </TouchableOpacity>
          </View>
          <ScrollView style={s.historyList} contentContainerStyle={{ padding: 16 }}>
            {history.length === 0 ? (
              <View style={s.empty}>
                <Text style={s.emptyIcon2}>◷</Text>
                <Text style={s.emptyTitle2}>No archived chats</Text>
                <Text style={s.emptyDesc}>Start a new chat to archive the current one here.</Text>
              </View>
            ) : history.map((h) => (
              <TouchableOpacity key={h.id} style={s.card} onPress={() => restoreChat(h)}>
                <View style={s.cardLeft}>
                  <Text style={s.cardIcon}>◷</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={s.cardName} numberOfLines={1}>{h.title}</Text>
                    <Text style={s.cardMeta}>{h.messages.length} messages · {h.savedAt}</Text>
                  </View>
                </View>
                <TouchableOpacity onPress={() => deleteArchived(h.id)} style={{ padding: 6 }}>
                  <Text style={{ color: "#ff5252", fontSize: 12, fontWeight: "700" }}>Delete</Text>
                </TouchableOpacity>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      )}
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────
const C = {
  bg: "#050505", surface: "#0d0d0d", border: "#1a1a1a",
  accent: "#00FFCC", text: "#e8e8e8", mid: "#555", dim: "#222",
};

const s = StyleSheet.create({
  boot:           { flex: 1, backgroundColor: C.bg, alignItems: "center", justifyContent: "center", padding: 32 },
  bootIcon:       { fontSize: 52, color: C.accent, marginBottom: 14 },
  bootTitle:      { fontSize: 28, fontWeight: "700", color: C.text, letterSpacing: 1 },
  bootSub:        { fontSize: 13, color: C.mid, marginTop: 6, letterSpacing: 0.5, marginBottom: 36 },
  progressTrack:  { width: "100%", height: 2, backgroundColor: C.border, borderRadius: 2, overflow: "hidden", marginBottom: 10 },
  progressFill:   { height: 2, backgroundColor: C.accent },
  bootMsg:        { fontSize: 12, color: C.mid, fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace", marginBottom: 4, textAlign: "center" },
  bootPct:        { fontSize: 20, color: C.accent, fontWeight: "700", height: 28 },
  bootModelLabel: { fontSize: 12, color: C.mid, marginTop: 4 },
  bootDevice:     { fontSize: 11, color: C.dim, position: "absolute", bottom: 36 },
  retryBtn:       { marginTop: 24, backgroundColor: C.accent, paddingHorizontal: 32, paddingVertical: 12, borderRadius: 8 },
  retryText:      { color: "#000", fontWeight: "700", fontSize: 14 },

  root:           { flex: 1, backgroundColor: C.bg },
  header:         { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 16, paddingTop: 14 + ANDROID_STATUSBAR_HEIGHT, paddingBottom: 10, borderBottomWidth: 1, borderBottomColor: C.border },
  headerLeft:     { flex: 1 },
  headerRight:    { flexDirection: "row", alignItems: "center", gap: 6 },
  headerTitle:    { fontSize: 16, fontWeight: "700", color: C.text, letterSpacing: 0.5 },
  headerSub:      { fontSize: 10, color: C.mid, marginTop: 1 },
  badge:          { flexDirection: "row", alignItems: "center", backgroundColor: "#00FFCC12", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 16, borderWidth: 1, borderColor: "#00FFCC25" },
  dot:            { width: 5, height: 5, borderRadius: 3, backgroundColor: C.accent, marginRight: 5 },
  badgeText:      { fontSize: 10, color: C.accent, fontWeight: "600" },
  iconBtn:        { width: 30, height: 30, alignItems: "center", justifyContent: "center" },
  iconBtnText:    { fontSize: 16, color: C.mid },
  newChatBtn:     { backgroundColor: "#0a1f18", borderWidth: 1, borderColor: "#0d3d2a", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 14 },
  newChatText:    { color: C.accent, fontSize: 11, fontWeight: "700" },

  tabBar:         { flexDirection: "row", borderBottomWidth: 1, borderBottomColor: C.border },
  tab:            { flex: 1, paddingVertical: 11, alignItems: "center" },
  tabActive:      { borderBottomWidth: 2, borderBottomColor: C.accent },
  tabText:        { fontSize: 12, color: C.mid, fontWeight: "500" },
  tabTextActive:  { color: C.accent, fontWeight: "700" },

  chatScroll:     { flex: 1 },
  chatInner:      { padding: 16, paddingBottom: 8 },
  emptyChat:      { alignItems: "center", paddingVertical: 40, paddingHorizontal: 24 },
  emptyChatIcon:  { fontSize: 40, color: C.dim, marginBottom: 14 },
  emptyChatTitle: { fontSize: 20, fontWeight: "700", color: C.mid, marginBottom: 6 },
  emptyChatSub:   { fontSize: 13, color: C.dim, textAlign: "center", lineHeight: 20 },
  chips:          { marginTop: 20, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", gap: 8 },
  chip:           { backgroundColor: "#0a1f18", borderWidth: 1, borderColor: "#0d3d2a", borderRadius: 20, paddingHorizontal: 14, paddingVertical: 7 },
  chipText:       { color: C.accent, fontSize: 12 },
  toolsHint:      { marginTop: 20, backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 14, width: "100%" },
  toolsHintTitle: { fontSize: 10, color: C.mid, fontWeight: "700", letterSpacing: 1, marginBottom: 6, textTransform: "uppercase" },
  toolsHintText:  { fontSize: 12, color: C.mid, lineHeight: 20 },

  bubble:         { marginBottom: 12, padding: 13, borderRadius: 18, maxWidth: "86%" },
  bubbleUser:     { alignSelf: "flex-end", backgroundColor: C.accent, borderBottomRightRadius: 4 },
  bubbleAI:       { alignSelf: "flex-start", backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderBottomLeftRadius: 4 },
  bubbleLabel:    { fontSize: 9, color: C.accent, fontWeight: "700", marginBottom: 5, letterSpacing: 1.2 },
  bubbleText:     { fontSize: 15, color: C.text, lineHeight: 22 },
  bubbleTextUser: { color: "#000" },
  bubbleTs:       { fontSize: 9, color: "rgba(255,255,255,0.2)", marginTop: 4, textAlign: "right" },

  tpsBadge:       { alignSelf: "center", backgroundColor: "#00FFCC10", borderWidth: 1, borderColor: "#00FFCC20", borderRadius: 12, paddingHorizontal: 12, paddingVertical: 4, marginBottom: 8 },
  tpsText:        { fontSize: 11, color: C.accent, fontWeight: "700", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace" },

  chatActions:    { flexDirection: "row", paddingHorizontal: 12, paddingVertical: 5, gap: 8 },
  actionBtn:      { flex: 1, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderRadius: 8, paddingVertical: 7, alignItems: "center" },
  actionBtnText:  { fontSize: 11, color: C.mid, fontWeight: "600" },

  inputRow:       { flexDirection: "row", paddingHorizontal: 12, paddingVertical: 8, alignItems: "flex-end", borderTopWidth: 1, borderTopColor: C.border, gap: 6, backgroundColor: C.bg },
  input:          { flex: 1, backgroundColor: C.surface, borderRadius: 22, borderWidth: 1, borderColor: C.border, color: C.text, paddingHorizontal: 14, paddingVertical: 9, fontSize: 14, maxHeight: 90 },
  micBtn:         { width: 38, height: 38, borderRadius: 19, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, alignItems: "center", justifyContent: "center" },
  micBtnActive:   { backgroundColor: "#2d0a0a", borderColor: "#ff5252" },
  micIcon:        { fontSize: 16 },
  sendBtn:        { backgroundColor: C.accent, width: 42, height: 42, borderRadius: 21, alignItems: "center", justifyContent: "center" },
  sendBtnOff:     { backgroundColor: "#0a1f18" },
  sendIcon:       { fontSize: 20, color: "#000", fontWeight: "700" },
  btnOff:         { opacity: 0.5 },
  row:            { flexDirection: "row", alignItems: "center" },

  scroll:         { flex: 1 },
  scrollInner:    { padding: 16, paddingBottom: 40 },
  primaryBtn:     { backgroundColor: C.accent, paddingVertical: 14, borderRadius: 10, alignItems: "center", marginBottom: 10 },
  primaryBtnText: { color: "#000", fontWeight: "700", fontSize: 15 },
  hint:           { fontSize: 12, color: C.mid, marginBottom: 16, lineHeight: 18 },
  empty:          { alignItems: "center", paddingVertical: 48 },
  emptyIcon2:     { fontSize: 36, color: C.dim, marginBottom: 12 },
  emptyTitle2:    { fontSize: 15, fontWeight: "600", color: C.mid, marginBottom: 6 },
  emptyDesc:      { fontSize: 13, color: C.dim, textAlign: "center", lineHeight: 20 },

  card:           { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 13, marginBottom: 10, flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardLeft:       { flexDirection: "row", alignItems: "center", flex: 1 },
  cardIcon:       { fontSize: 18, color: C.accent, marginRight: 10 },
  cardName:       { fontSize: 14, fontWeight: "600", color: C.text, maxWidth: width * 0.5 },
  cardMeta:       { fontSize: 11, color: C.mid, marginTop: 2 },
  cardBadge:      { backgroundColor: "#00FFCC12", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  cardBadgeText:  { fontSize: 11, color: C.accent, fontWeight: "600" },

  titleInput:     { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 15, fontWeight: "600", marginBottom: 8 },
  bodyInput:      { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 14, minHeight: 140, marginBottom: 12, lineHeight: 22 },

  logCard:        { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 12 },
  logLabel:       { fontSize: 10, color: C.mid, fontWeight: "700", marginBottom: 8, letterSpacing: 1 },
  logLine:        { fontSize: 10, color: "#333", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace", marginBottom: 3, lineHeight: 15 },

  // Settings modal
  modalOverlay:   { flex: 1, backgroundColor: "rgba(0,0,0,0.7)", justifyContent: "flex-end" },
  modalSheet:     { backgroundColor: "#0a0a0a", borderTopLeftRadius: 20, borderTopRightRadius: 20, maxHeight: "85%", borderWidth: 1, borderColor: C.border },
  modalHeader:    { flexDirection: "row", justifyContent: "space-between", alignItems: "center", padding: 20, borderBottomWidth: 1, borderBottomColor: C.border },
  modalTitle:     { fontSize: 16, fontWeight: "700", color: C.text },
  modalClose:     { color: C.accent, fontSize: 14, fontWeight: "700" },
  sectionLabel:   { fontSize: 10, color: C.mid, fontWeight: "700", letterSpacing: 1.5, marginBottom: 10 },
  modelCard:      { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 14, marginBottom: 8, flexDirection: "row", alignItems: "center" },
  modelCardActive:{ backgroundColor: C.accent, borderColor: C.accent },
  modelName:      { fontSize: 14, fontWeight: "700", color: C.text, marginBottom: 2 },
  modelDesc:      { fontSize: 11, color: C.mid },
  modelCheck:     { fontSize: 18, color: "#000", fontWeight: "700" },
  toolCard:       { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 12, marginBottom: 8, flexDirection: "row", alignItems: "center" },
  toolName:       { fontSize: 13, fontWeight: "600", color: C.text, marginBottom: 2 },
  toolDesc:       { fontSize: 11, color: C.mid, lineHeight: 16 },
  toolActive:     { fontSize: 14, color: C.accent, fontWeight: "700", marginLeft: 8 },
  infoCard:       { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 14 },
  previewBox:     { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 14, marginBottom: 40 },
  previewText:    { fontSize: 12, color: C.text, lineHeight: 19, fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace" },
  infoText:       { fontSize: 12, color: C.mid, marginBottom: 4, lineHeight: 18 },
  dangerBtn:      { backgroundColor: "#1a0505", borderWidth: 1, borderColor: "#ff5252", borderRadius: 8, padding: 14, alignItems: "center", marginBottom: 40 },
  dangerBtnText:  { color: "#ff5252", fontWeight: "700", fontSize: 14 },

  // History overlay
  historyOverlay: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: C.bg, zIndex: 100 },
  historyHeader:  { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 16, paddingTop: 14 + ANDROID_STATUSBAR_HEIGHT, paddingBottom: 14, borderBottomWidth: 1, borderBottomColor: C.border },
  historyTitle:   { fontSize: 16, fontWeight: "700", color: C.text },
  historyClose:   { color: C.accent, fontSize: 13, fontWeight: "700" },
  historyList:    { flex: 1 },
});
