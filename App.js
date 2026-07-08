import { useState, useEffect, useRef, useCallback } from "react";
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  ScrollView, ActivityIndicator, SafeAreaView, StatusBar,
  Alert, Platform, Dimensions, Share, KeyboardAvoidingView,
  Keyboard
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as DocumentPicker from "expo-document-picker";
import * as FileSystem from "expo-file-system/legacy";
import * as Device from "expo-device";

// ─── QVAC SDK 0.13.3 ─────────────────────────────────────────
let loadModel, completion, unloadModel, ragIngest, ragSearch, ragCloseWorkspace, GTE_LARGE_FP16, LLAMA_3_2_1B_INST_Q4_0;
let QVAC_OK = true;

try {
  const sdk = require("@qvac/sdk");
  loadModel         = sdk.loadModel;
  completion        = sdk.completion;
  unloadModel       = sdk.unloadModel;
  ragIngest         = sdk.ragIngest;
  ragSearch         = sdk.ragSearch;
  ragCloseWorkspace = sdk.ragCloseWorkspace;
  GTE_LARGE_FP16    = sdk.GTE_LARGE_FP16;
  LLAMA_3_2_1B_INST_Q4_0 = sdk.LLAMA_3_2_1B_INST_Q4_0;
  console.log("[QVAC] exports:", Object.keys(sdk).join(", "));
} catch (e) {
  console.error("[QVAC] load failed:", e);
  QVAC_OK = false;
}

const LLM_SRC = LLAMA_3_2_1B_INST_Q4_0 ||
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

const RAG_WORKSPACE = "webnix-rag";
const STORAGE_KEYS = {
  CHAT:    "webnix_chat_v3",
  FILES:   "webnix_files_v3",
  NOTES:   "webnix_notes_v3",
  HISTORY: "webnix_chat_history_v1",
};

const { width } = Dimensions.get("window");
const TABS = { ASK: "ask", FILES: "files", NOTES: "notes" };
const IS_IOS = Platform.OS === "ios";

// FIX: SafeAreaView from 'react-native' is a no-op on Android — it only
// insets on iOS. On Android you must manually pad for the status bar or
// your header draws underneath the clock/signal/battery icons.
const ANDROID_STATUSBAR_HEIGHT = Platform.OS === "android" ? (StatusBar.currentHeight || 24) : 0;

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
  const [llmId,      setLlmId]      = useState(null);
  const [embedId,    setEmbedId]    = useState(null);
  const [bootState,  setBootState]  = useState("idle");
  const [progress,   setProgress]   = useState(0);
  const [bootMsg,    setBootMsg]    = useState("");

  const [tab,        setTab]        = useState(TABS.ASK);
  const [logs,       setLogs]       = useState([]);

  // Persisted state
  const [files,      setFiles]      = useState([]);
  const [notes,      setNotes]      = useState([]);
  const [chat,       setChat]       = useState([]);
  const [history,    setHistory]    = useState([]); // archived past conversations
  const [showHistory,setShowHistory]= useState(false);

  // Files UI
  const [indexing,   setIndexing]   = useState(false);
  const [curFile,    setCurFile]    = useState(null);

  // Notes UI
  const [noteTitle,  setNoteTitle]  = useState("");
  const [noteBody,   setNoteBody]   = useState("");
  const [savingNote, setSavingNote] = useState(false);

  // Chat UI
  const [query,      setQuery]      = useState("");
  const [thinking,   setThinking]   = useState(false);
  const [tps,        setTps]        = useState(null);

  const scrollRef  = useRef(null);
  const inputRef   = useRef(null);

  // FIX: refs mirror llmId/embedId so the unmount cleanup closure (which
  // is frozen at mount time because the effect below has an empty dep
  // array) can actually see the current model IDs instead of always
  // reading null.
  const llmIdRef   = useRef(null);
  const embedIdRef = useRef(null);
  useEffect(() => { llmIdRef.current = llmId; }, [llmId]);
  useEffect(() => { embedIdRef.current = embedId; }, [embedId]);

  const log = useCallback((msg) => {
    const line = `${new Date().toLocaleTimeString()} ${msg}`;
    setLogs(p => [...p.slice(-40), line]);
    console.log("[WebnixAI]", msg);
  }, []);

  // ── Load persisted data on mount ───────────────────────────
  // FIX: removed the duplicate reload that used to fire 500ms after boot()
  // and could stomp fresh state with stale persisted state if the user
  // typed fast. Single source of truth, loaded once.
  useEffect(() => {
    (async () => {
      const [c, f, n, h] = await Promise.all([
        load(STORAGE_KEYS.CHAT,    []),
        load(STORAGE_KEYS.FILES,   []),
        load(STORAGE_KEYS.NOTES,   []),
        load(STORAGE_KEYS.HISTORY, []),
      ]);
      setChat(c);
      setFiles(f);
      setNotes(n);
      setHistory(h);
      log(`Restored: ${c.length} msgs, ${f.length} files, ${n.length} notes, ${h.length} archived chats`);
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
    boot();
    return () => {
      // FIX: read from refs, not stale state captured at mount.
      if (llmIdRef.current)   unloadModel({ modelId: llmIdRef.current,   clearStorage: false }).catch(() => {});
      if (embedIdRef.current) unloadModel({ modelId: embedIdRef.current, clearStorage: false }).catch(() => {});
      ragCloseWorkspace?.({ workspace: RAG_WORKSPACE, deleteOnClose: false }).catch(() => {});
    };
  }, []);

  const boot = async () => {
    setBootState("loading");
    try {
      setBootMsg("Loading language model...");
      log("Loading LLM...");
      const lId = await loadModel({
        modelSrc:    LLM_SRC,
        modelType:   "llamacpp-completion",
        modelConfig: { device: "cpu", ctx_size: 2048 },
        onProgress:  (p) => setProgress(Math.round(p.percentage || p * 100 || 0)),
      });
      setLlmId(lId);
      log("LLM ready");

      setBootMsg("Loading embedding model...");
      log("Loading GTE-Large...");
      const eId = await loadModel({
        modelSrc:   GTE_LARGE_FP16,
        onProgress: (p) => setProgress(Math.round(p.percentage || p * 100 || 0)),
      });
      setEmbedId(eId);
      log("Embedder ready");

      setBootState("ready");
      setBootMsg("");
      log("All models active");
    } catch (e) {
      log("Boot failed: " + (e?.message || e));
      setBootState("error");
      setBootMsg(String(e?.message || e));
    }
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

  const updateHistory = useCallback((updated) => {
    setHistory(updated);
    save(STORAGE_KEYS.HISTORY, updated);
  }, []);

  // ── Index file ─────────────────────────────────────────────
  const pickFile = async () => {
    if (!embedId) return Alert.alert("Not Ready", "Models still loading.");
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: ["text/*", "application/json", "*/*"],
        copyToCacheDirectory: true,
      });
      if (res.canceled) return;

      const file = res.assets[0];
      setCurFile(file.name); setIndexing(true);
      log("Indexing: " + file.name);

      const text   = await readFile(file.uri, file.mimeType, file.name);
      const chunks = chunkText(text);
      log(`${chunks.length} chunks`);

      const result = await ragIngest({
        modelId:   embedId,
        workspace: RAG_WORKSPACE,
        documents: chunks,
        chunk:     false,
      });

      const newFile = {
        name:   file.name,
        chunks: result.processed?.length || chunks.length,
        at:     new Date().toLocaleTimeString(),
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

  // ── Save note ──────────────────────────────────────────────
  const saveNote = async () => {
    if (!noteBody.trim()) return Alert.alert("Empty", "Write something.");
    if (!embedId)         return Alert.alert("Not Ready", "Models loading.");
    setSavingNote(true);
    const title = noteTitle.trim() || `Note ${notes.length + 1}`;
    try {
      const chunks = chunkText(`Title: ${title}\n\n${noteBody.trim()}`);
      await ragIngest({ modelId: embedId, workspace: RAG_WORKSPACE, documents: chunks, chunk: false });
      const newNote = { title, preview: noteBody.slice(0, 80), at: new Date().toLocaleTimeString() };
      updateNotes([...notes, newNote]);
      log("✓ Note: " + title);
      setNoteTitle(""); setNoteBody("");
      Alert.alert("Saved", `"${title}" indexed.`);
    } catch (e) {
      log("Save failed: " + (e?.message || e));
      Alert.alert("Error", String(e?.message || e));
    } finally { setSavingNote(false); }
  };

  // ── Ask AI ─────────────────────────────────────────────────
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

      const systemPrompt = context
        ? `You are Webnix AI, a private personal assistant running entirely on this device. The user has indexed files and notes. Use the context below to answer accurately. Be concise.\n\nIndexed content:\n${context}`
        : "You are Webnix AI, a private personal assistant running entirely on this device. Answer concisely and helpfully.";

      const historyMsgs = chat
        .filter(m => !m.streaming && m.content)
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }));

      const llmHistory = [
        { role: "user", content: systemPrompt },
        ...historyMsgs,
        { role: "user", content: q },
      ];

      const run = completion({ modelId: llmId, history: llmHistory, stream: true });

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

      const elapsed = (Date.now() - t0) / 1000;
      const finalTps = elapsed > 0 ? (tokens / elapsed).toFixed(1) : "?";
      setTps(finalTps);

      const finalContent = stripThink(raw);
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

  // ── New chat ───────────────────────────────────────────────
  // FIX: this used to hard-delete the conversation (updateChat([])) with
  // no way to get it back. Now it archives the current thread into
  // `history` first, then clears. Nothing is destroyed anymore.
  const newChat = () => {
    if (chat.length === 0) return; // nothing to archive, nothing to clear
    Alert.alert("New Chat", "Archive this conversation and start a new one?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Start New", style: "destructive", onPress: () => {
          const completedMsgs = chat.filter(m => !m.streaming && m.content);
          if (completedMsgs.length > 0) {
            const firstUserMsg = completedMsgs.find(m => m.role === "user");
            const entry = {
              id:    Date.now().toString(),
              title: (firstUserMsg?.content || "Conversation").slice(0, 48),
              messages: completedMsgs,
              savedAt: new Date().toLocaleString(),
            };
            updateHistory([entry, ...history]);
          }
          updateChat([]);
          setTps(null);
          log("Chat archived, new conversation started");
        }
      },
    ]);
  };

  // ── Restore an archived chat ────────────────────────────────
  const restoreChat = (entry) => {
    // Archive whatever's currently active before swapping it out, so
    // restoring one thread never silently drops another.
    if (chat.length > 0) {
      const completedMsgs = chat.filter(m => !m.streaming && m.content);
      if (completedMsgs.length > 0) {
        const firstUserMsg = completedMsgs.find(m => m.role === "user");
        const currentEntry = {
          id:    Date.now().toString(),
          title: (firstUserMsg?.content || "Conversation").slice(0, 48),
          messages: completedMsgs,
          savedAt: new Date().toLocaleString(),
        };
        updateHistory([currentEntry, ...history.filter(h => h.id !== entry.id)]);
      } else {
        updateHistory(history.filter(h => h.id !== entry.id));
      }
    } else {
      updateHistory(history.filter(h => h.id !== entry.id));
    }
    updateChat(entry.messages);
    setTps(null);
    setShowHistory(false);
    log(`Restored archived chat: ${entry.title}`);
  };

  const deleteArchived = (id) => {
    Alert.alert("Delete", "Permanently delete this archived conversation?", [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => updateHistory(history.filter(h => h.id !== id)) },
    ]);
  };

  // ── Save / share chat ──────────────────────────────────────
  const saveChat = async () => {
    if (!chat.length) return;
    const text = chat
      .filter(m => m.content)
      .map(m => `[${m.ts}] ${m.role === "user" ? "You" : "Webnix AI"}: ${m.content}`)
      .join("\n\n");
    try { await Share.share({ message: text, title: "Webnix AI Chat" }); }
    catch (e) { Alert.alert("Error", String(e?.message || e)); }
  };

  // ── Boot screens ───────────────────────────────────────────
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
        <TouchableOpacity style={s.retryBtn} onPress={boot}>
          <Text style={s.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Main UI ────────────────────────────────────────────────
  return (
    <SafeAreaView style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor="#050505" />

      {/* Header — Android status bar padding applied via ANDROID_STATUSBAR_HEIGHT in styles */}
      <View style={s.header}>
        <View style={s.headerLeft}>
          <Text style={s.headerTitle}>◈ Webnix AI</Text>
          <Text style={s.headerSub}>All processing on this device</Text>
        </View>
        <View style={s.headerRight}>
          <View style={s.badge}>
            <View style={s.dot} />
            <Text style={s.badgeText}>Online</Text>
          </View>
          {tab === TABS.ASK && (
            <TouchableOpacity style={s.historyBtn} onPress={() => setShowHistory(true)}>
              <Text style={s.historyBtnText}>History{history.length > 0 ? ` (${history.length})` : ""}</Text>
            </TouchableOpacity>
          )}
          {tab === TABS.ASK && chat.length > 0 && (
            <TouchableOpacity style={s.newChatBtn} onPress={newChat}>
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
        <KeyboardAvoidingView
          style={{ flex: 1 }}
          behavior={IS_IOS ? "padding" : "height"}
          keyboardVerticalOffset={IS_IOS ? 0 : 0}
        >
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
                  {["What can you do?", "Summarize my files", "Explain blockchain", "Help me write"].map(c => (
                    <TouchableOpacity key={c} style={s.chip} onPress={() => setQuery(c)}>
                      <Text style={s.chipText}>{c}</Text>
                    </TouchableOpacity>
                  ))}
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
                <Text style={s.tpsText}>{tps} tok/s · Llama 3.2 1B</Text>
              </View>
            )}
          </ScrollView>

          {chat.length > 0 && (
            <View style={s.chatActions}>
              <TouchableOpacity style={s.actionBtn} onPress={saveChat}>
                <Text style={s.actionBtnText}>↓ Save Chat</Text>
              </TouchableOpacity>
              <TouchableOpacity style={s.actionBtn} onPress={newChat}>
                <Text style={s.actionBtnText}>+ New Chat</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Input row */}
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
              style={s.micBtn}
              onPress={() => {
                inputRef.current?.focus();
                Alert.alert("Voice Input", "Tap the mic on your keyboard to dictate. Your device keyboard supports voice input natively.", [{ text: "OK" }]);
              }}
            >
              <Text style={s.micIcon}>🎤</Text>
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
          {/* FIX: PDF removed from picker + hint. readFile() never extracted PDF text —
              it silently fell through to the binary-file branch and fed the model
              nothing but a filename. Claiming PDF support was false. Add real PDF
              text extraction before re-enabling this. */}
          <Text style={s.hint}>TXT, MD, JS, PY, CSV, JSON — processed on device.</Text>

          {files.length === 0 ? (
            <View style={s.empty}>
              <Text style={s.emptyIcon2}>▣</Text>
              <Text style={s.emptyTitle2}>No files indexed</Text>
              <Text style={s.emptyDesc}>Pick a file — AI will read and index it for Q&A.</Text>
            </View>
          ) : files.map((f, i) => (
            <View key={i} style={s.card}>
              <View style={s.cardLeft}>
                <Text style={s.cardIcon}>◻</Text>
                <View style={{ flex: 1 }}>
                  <Text style={s.cardName} numberOfLines={1}>{f.name}</Text>
                  <Text style={s.cardMeta}>{f.chunks} chunks · {f.at}</Text>
                </View>
              </View>
              <View style={s.cardBadge}><Text style={s.cardBadgeText}>Indexed</Text></View>
            </View>
          ))}
        </ScrollView>
      )}

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
                <Text style={s.emptyDesc}>Starting a new chat archives the current one here.</Text>
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
  // Boot
  boot:          { flex: 1, backgroundColor: C.bg, alignItems: "center", justifyContent: "center", padding: 32 },
  bootIcon:      { fontSize: 52, color: C.accent, marginBottom: 14 },
  bootTitle:     { fontSize: 28, fontWeight: "700", color: C.text, letterSpacing: 1 },
  bootSub:       { fontSize: 13, color: C.mid, marginTop: 6, letterSpacing: 0.5, marginBottom: 36 },
  progressTrack: { width: "100%", height: 2, backgroundColor: C.border, borderRadius: 2, overflow: "hidden", marginBottom: 10 },
  progressFill:  { height: 2, backgroundColor: C.accent },
  bootMsg:       { fontSize: 12, color: C.mid, fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace", marginBottom: 4, textAlign: "center" },
  bootPct:       { fontSize: 20, color: C.accent, fontWeight: "700", height: 28 },
  bootDevice:    { fontSize: 11, color: C.dim, position: "absolute", bottom: 36 },
  retryBtn:      { marginTop: 24, backgroundColor: C.accent, paddingHorizontal: 32, paddingVertical: 12, borderRadius: 8 },
  retryText:     { color: "#000", fontWeight: "700", fontSize: 14 },

  // Layout
  root:          { flex: 1, backgroundColor: C.bg },

  // Header — FIX: paddingTop includes ANDROID_STATUSBAR_HEIGHT so content
  // clears the clock/signal/battery icons on Android (SafeAreaView doesn't
  // do this for you outside iOS).
  header:        { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 16, paddingTop: 14 + ANDROID_STATUSBAR_HEIGHT, paddingBottom: 10, borderBottomWidth: 1, borderBottomColor: C.border },
  headerLeft:    { flex: 1 },
  headerRight:   { flexDirection: "row", alignItems: "center", gap: 8 },
  headerTitle:   { fontSize: 16, fontWeight: "700", color: C.text, letterSpacing: 0.5 },
  headerSub:     { fontSize: 10, color: C.mid, marginTop: 1 },
  badge:         { flexDirection: "row", alignItems: "center", backgroundColor: "#00FFCC12", paddingHorizontal: 8, paddingVertical: 4, borderRadius: 16, borderWidth: 1, borderColor: "#00FFCC25" },
  dot:           { width: 5, height: 5, borderRadius: 3, backgroundColor: C.accent, marginRight: 5 },
  badgeText:     { fontSize: 10, color: C.accent, fontWeight: "600" },
  newChatBtn:    { backgroundColor: "#0a1f18", borderWidth: 1, borderColor: "#0d3d2a", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 14 },
  newChatText:   { color: C.accent, fontSize: 11, fontWeight: "700" },
  historyBtn:    { backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, paddingHorizontal: 10, paddingVertical: 4, borderRadius: 14 },
  historyBtnText:{ color: C.mid, fontSize: 11, fontWeight: "700" },

  historyOverlay:{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: C.bg, zIndex: 100 },
  historyHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 16, paddingTop: 14 + ANDROID_STATUSBAR_HEIGHT, paddingBottom: 14, borderBottomWidth: 1, borderBottomColor: C.border },
  historyTitle:  { fontSize: 16, fontWeight: "700", color: C.text },
  historyClose:  { color: C.accent, fontSize: 13, fontWeight: "700" },
  historyList:   { flex: 1 },

  // Tabs
  tabBar:        { flexDirection: "row", borderBottomWidth: 1, borderBottomColor: C.border },
  tab:           { flex: 1, paddingVertical: 11, alignItems: "center" },
  tabActive:     { borderBottomWidth: 2, borderBottomColor: C.accent },
  tabText:       { fontSize: 12, color: C.mid, fontWeight: "500" },
  tabTextActive: { color: C.accent, fontWeight: "700" },

  // Chat
  chatScroll:    { flex: 1 },
  chatInner:     { padding: 16, paddingBottom: 8 },
  emptyChat:     { alignItems: "center", paddingVertical: 48, paddingHorizontal: 24 },
  emptyChatIcon: { fontSize: 40, color: C.dim, marginBottom: 14 },
  emptyChatTitle:{ fontSize: 20, fontWeight: "700", color: C.mid, marginBottom: 6 },
  emptyChatSub:  { fontSize: 13, color: C.dim, textAlign: "center", lineHeight: 20 },
  chips:         { marginTop: 24, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", gap: 8 },
  chip:          { backgroundColor: "#0a1f18", borderWidth: 1, borderColor: "#0d3d2a", borderRadius: 20, paddingHorizontal: 14, paddingVertical: 7 },
  chipText:      { color: C.accent, fontSize: 12 },

  bubble:        { marginBottom: 12, padding: 13, borderRadius: 18, maxWidth: "86%" },
  bubbleUser:    { alignSelf: "flex-end", backgroundColor: C.accent, borderBottomRightRadius: 4 },
  bubbleAI:      { alignSelf: "flex-start", backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderBottomLeftRadius: 4 },
  bubbleLabel:   { fontSize: 9, color: C.accent, fontWeight: "700", marginBottom: 5, letterSpacing: 1.2 },
  bubbleText:    { fontSize: 15, color: C.text, lineHeight: 22 },
  bubbleTextUser:{ color: "#000" },
  bubbleTs:      { fontSize: 9, color: "rgba(255,255,255,0.2)", marginTop: 4, textAlign: "right" },

  tpsBadge:      { alignSelf: "center", backgroundColor: "#00FFCC10", borderWidth: 1, borderColor: "#00FFCC20", borderRadius: 12, paddingHorizontal: 12, paddingVertical: 4, marginBottom: 8 },
  tpsText:       { fontSize: 11, color: C.accent, fontWeight: "700", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace" },

  chatActions:   { flexDirection: "row", paddingHorizontal: 12, paddingVertical: 5, gap: 8 },
  actionBtn:     { flex: 1, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderRadius: 8, paddingVertical: 7, alignItems: "center" },
  actionBtnText: { fontSize: 11, color: C.mid, fontWeight: "600" },

  inputRow:      { flexDirection: "row", paddingHorizontal: 12, paddingVertical: 8, alignItems: "flex-end", borderTopWidth: 1, borderTopColor: C.border, gap: 6, backgroundColor: C.bg },
  input:         { flex: 1, backgroundColor: C.surface, borderRadius: 22, borderWidth: 1, borderColor: C.border, color: C.text, paddingHorizontal: 14, paddingVertical: 9, fontSize: 14, maxHeight: 90 },
  micBtn:        { width: 38, height: 38, borderRadius: 19, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, alignItems: "center", justifyContent: "center" },
  micIcon:       { fontSize: 16 },
  sendBtn:       { backgroundColor: C.accent, width: 42, height: 42, borderRadius: 21, alignItems: "center", justifyContent: "center" },
  sendBtnOff:    { backgroundColor: "#0a1f18" },
  sendIcon:      { fontSize: 20, color: "#000", fontWeight: "700" },

  scroll:        { flex: 1 },
  scrollInner:   { padding: 16, paddingBottom: 40 },
  primaryBtn:    { backgroundColor: C.accent, paddingVertical: 14, borderRadius: 10, alignItems: "center", marginBottom: 10 },
  primaryBtnText:{ color: "#000", fontWeight: "700", fontSize: 15 },
  btnOff:        { opacity: 0.5 },
  row:           { flexDirection: "row", alignItems: "center" },
  hint:          { fontSize: 12, color: C.mid, marginBottom: 16, lineHeight: 18 },
  empty:         { alignItems: "center", paddingVertical: 48 },
  emptyIcon2:    { fontSize: 36, color: C.dim, marginBottom: 12 },
  emptyTitle2:   { fontSize: 15, fontWeight: "600", color: C.mid, marginBottom: 6 },
  emptyDesc:     { fontSize: 13, color: C.dim, textAlign: "center", lineHeight: 20 },

  card:          { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 13, marginBottom: 10, flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardLeft:      { flexDirection: "row", alignItems: "center", flex: 1 },
  cardIcon:      { fontSize: 18, color: C.accent, marginRight: 10 },
  cardName:      { fontSize: 14, fontWeight: "600", color: C.text, maxWidth: width * 0.5 },
  cardMeta:      { fontSize: 11, color: C.mid, marginTop: 2 },
  cardBadge:     { backgroundColor: "#00FFCC12", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  cardBadgeText: { fontSize: 11, color: C.accent, fontWeight: "600" },

  titleInput:    { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 15, fontWeight: "600", marginBottom: 8 },
  bodyInput:     { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 14, minHeight: 140, marginBottom: 12, lineHeight: 22 },

  logCard:       { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 12 },
  logLabel:      { fontSize: 10, color: C.mid, fontWeight: "700", marginBottom: 8, letterSpacing: 1 },
  logLine:       { fontSize: 10, color: "#333", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace", marginBottom: 3, lineHeight: 15 },
});
