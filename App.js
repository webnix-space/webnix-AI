import { useState, useEffect, useRef, useCallback } from "react";
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  ScrollView, ActivityIndicator, SafeAreaView, StatusBar,
  Alert, Platform, Dimensions, Share
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as DocumentPicker from "expo-document-picker";
import * as FileSystem from "expo-file-system/legacy";
import * as Device from "expo-device";

// ─── QVAC SDK 0.14.x ─────────────────────────────────────────
let loadModel, completion, unloadModel, ragIngest, ragSearch,
    ragCloseWorkspace, GTE_LARGE_FP16, LLAMA_3_2_1B_INST_Q4_0,
    RequestValidationFailedError;
let QVAC_OK = true;

try {
  const sdk = require("@qvac/sdk");
  loadModel                  = sdk.loadModel;
  completion                 = sdk.completion;
  unloadModel                = sdk.unloadModel;
  ragIngest                  = sdk.ragIngest;
  ragSearch                  = sdk.ragSearch;
  ragCloseWorkspace          = sdk.ragCloseWorkspace;
  GTE_LARGE_FP16             = sdk.GTE_LARGE_FP16;
  LLAMA_3_2_1B_INST_Q4_0 = sdk.LLAMA_3_2_1B_INST_Q4_0;
  RequestValidationFailedError   = sdk.RequestValidationFailedError;
  console.log("[QVAC] exports:", Object.keys(sdk).join(", "));
} catch (e) {
  console.error("[QVAC] load failed:", e);
  QVAC_OK = false;
}

// Fallback URL if named constant is undefined in bundle
const LLM_SRC = LLAMA_3_2_1B_INST_Q4_0 ||
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

const RAG_WORKSPACE = "webnix-rag";
const CHAT_KEY      = "webnix_chat_v2";
const { width }     = Dimensions.get("window");
const TABS          = { ASK: "ask", FILES: "files", NOTES: "notes" };

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
      name.endsWith(".txt") || name.endsWith(".md") ||
      name.endsWith(".csv") || name.endsWith(".json")
    ) {
      return await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.UTF8 });
    }
    return `File: ${name}\nType: ${mimeType}\nNote: Binary — AI reasons from filename and type.`;
  } catch (e) {
    throw new Error("Cannot read file: " + e.message);
  }
};

const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

const fmtError = (e) => {
  if (RequestValidationFailedError && e instanceof RequestValidationFailedError)
    return e.message;
  return String(e?.message || e);
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

  const [files,      setFiles]      = useState([]);
  const [indexing,   setIndexing]   = useState(false);
  const [curFile,    setCurFile]    = useState(null);

  const [noteTitle,  setNoteTitle]  = useState("");
  const [noteBody,   setNoteBody]   = useState("");
  const [notes,      setNotes]      = useState([]);
  const [savingNote, setSavingNote] = useState(false);

  const [chat,       setChat]       = useState([]);
  const [query,      setQuery]      = useState("");
  const [thinking,   setThinking]   = useState(false);
  const [tps,        setTps]        = useState(null);

  const scrollRef = useRef(null);

  const log = useCallback((msg) => {
    const line = `${new Date().toLocaleTimeString()} ${msg}`;
    setLogs(p => [...p.slice(-40), line]);
    console.log("[WebnixAI]", msg);
  }, []);

  // ── Persist chat ───────────────────────────────────────────
  useEffect(() => {
    AsyncStorage.getItem(CHAT_KEY)
      .then(v => { if (v) setChat(JSON.parse(v)); })
      .catch(() => {});
  }, []);

  const persistChat = useCallback((updated) => {
    setChat(updated);
    AsyncStorage.setItem(CHAT_KEY, JSON.stringify(updated)).catch(() => {});
  }, []);

  // ── Auto-scroll ────────────────────────────────────────────
  useEffect(() => {
    if (chat.length > 0)
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 80);
  }, [chat]);

  // ── Boot ───────────────────────────────────────────────────
  useEffect(() => {
    if (!QVAC_OK) { setBootState("error"); setBootMsg("QVAC SDK missing"); return; }
    boot();
    return () => {
      if (llmId)   unloadModel({ modelId: llmId,   clearStorage: false }).catch(() => {});
      if (embedId) unloadModel({ modelId: embedId, clearStorage: false }).catch(() => {});
      ragCloseWorkspace?.({ workspace: RAG_WORKSPACE, deleteOnClose: false }).catch(() => {});
    };
  }, []);

  const boot = async () => {
    setBootState("loading");
    try {
      // ── LLM ─────────────────────────────────────────────
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

      // ── Embedder ─────────────────────────────────────────
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
      const msg = fmtError(e);
      log("Boot failed: " + msg);
      setBootState("error");
      setBootMsg(msg);
    }
  };

  // ── Index file ─────────────────────────────────────────────
  const pickFile = async () => {
    if (!embedId) return Alert.alert("Not Ready", "Models loading.");
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
      log(`${chunks.length} chunks`);

      const result = await ragIngest({
        modelId:   embedId,
        workspace: RAG_WORKSPACE,
        documents: chunks,
        chunk:     false,
      });

      setFiles(p => [...p, {
        name:   file.name,
        chunks: result.processed?.length || chunks.length,
        at:     new Date().toLocaleTimeString(),
      }]);
      log(`✓ Indexed: ${file.name}`);
      Alert.alert("Done", `"${file.name}" ready to query.`);
    } catch (e) {
      const msg = fmtError(e);
      log("Index failed: " + msg);
      Alert.alert("Error", msg);
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
      setNotes(p => [...p, { title, preview: noteBody.slice(0, 80), at: new Date().toLocaleTimeString() }]);
      log("✓ Note: " + title);
      setNoteTitle(""); setNoteBody("");
      Alert.alert("Saved", `"${title}" indexed.`);
    } catch (e) {
      const msg = fmtError(e);
      log("Save failed: " + msg);
      Alert.alert("Error", msg);
    } finally { setSavingNote(false); }
  };

  // ── Ask AI — 0.14.x result.events API ─────────────────────
  const ask = async () => {
    const q = query.trim();
    if (!q || !llmId) return;

    const userMsg = { role: "user", content: q, ts: new Date().toLocaleTimeString() };
    const withUser = [...chat, userMsg];
    persistChat(withUser);
    setQuery(""); setThinking(true); setTps(null);

    const aiMsg = { role: "assistant", content: "", ts: new Date().toLocaleTimeString(), streaming: true };
    const withAi = [...withUser, aiMsg];
    setChat(withAi);

    try {
      // RAG
      let context = "", source = "general knowledge";
      if (embedId && (files.length > 0 || notes.length > 0)) {
        try {
          const results = await ragSearch({ modelId: embedId, workspace: RAG_WORKSPACE, query: q, topK: 4 });
          if (results?.length && results[0].score > 0.4) {
            context = results.map((r, i) => `[${i+1}] ${r.content}`).join("\n\n");
            source  = "your files & notes";
            log(`RAG: ${results.length} matches`);
          }
        } catch (e) { log("RAG skip: " + fmtError(e)); }
      }

      const systemPrompt = context
        ? `You are Webnix AI, a private assistant on this device. Use context. Be concise.\n\nContext:\n${context}`
        : "You are Webnix AI, a private assistant running entirely on this device. Be concise.";

      const history = [
        { role: "user", content: systemPrompt },
        ...chat.filter(m => !m.streaming).slice(-6).map(m => ({ role: m.role, content: m.content })),
        { role: "user", content: q },
      ];

      // 0.14.x: use result.events for typed stream + real tok/s
      const run = completion({ modelId: llmId, history, stream: true });

      let raw = "", tokens = 0;
      const t0 = Date.now();

      for await (const event of run.events) {
        if (event.type === "contentDelta") {
          raw += event.text;
          tokens++;
          const hasOpenThink = raw.includes("<think>") && !raw.includes("</think>");
          const clean = hasOpenThink ? "" : stripThink(raw);
          const elapsed = (Date.now() - t0) / 1000;
          if (elapsed > 0.3) setTps((tokens / elapsed).toFixed(1));
          setChat(prev => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], content: clean };
            return next;
          });
        }
        if (event.type === "completionStats" && event.stats?.tokensPerSecond) {
          setTps(event.stats.tokensPerSecond.toFixed(1));
        }
      }

      // 0.14.x: result.final for aggregated output
      const final = await run.final;
      const finalContent = stripThink(final.contentText || raw);
      const finalTps = final.stats?.tokensPerSecond?.toFixed(1) || tps;

      if (finalTps) setTps(finalTps);

      const done = [...withAi];
      done[done.length - 1] = {
        role: "assistant", content: finalContent,
        ts: aiMsg.ts, streaming: false, source,
      };
      persistChat(done);
      log(`Done — ${tokens} tok, ${finalTps} tok/s`);
    } catch (e) {
      const msg = fmtError(e);
      log("Ask failed: " + msg);
      const err = [...withAi];
      err[err.length - 1] = { role: "assistant", content: "Error: " + msg, ts: aiMsg.ts, streaming: false };
      persistChat(err);
    } finally { setThinking(false); }
  };

  // ── Chat actions ───────────────────────────────────────────
  const saveChat = async () => {
    if (!chat.length) return;
    const text = chat
      .map(m => `[${m.ts}] ${m.role === "user" ? "You" : "Webnix AI"}: ${m.content}`)
      .join("\n\n");
    try { await Share.share({ message: text, title: "Webnix AI Chat" }); }
    catch (e) { Alert.alert("Error", fmtError(e)); }
  };

  const clearChat = () => { persistChat([]); setTps(null); log("Chat cleared"); };

  // ── Boot screen ────────────────────────────────────────────
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
        <Text style={s.bootProgress}>{progress}%</Text>
        <Text style={s.bootDevice}>{Device.modelName || "Device"} · Local Silicon</Text>
      </View>
    );
  }

  if (bootState === "error") {
    return (
      <View style={s.boot}>
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

      <View style={s.header}>
        <View>
          <Text style={s.headerTitle}>◈ Webnix AI</Text>
          <Text style={s.headerSub}>All processing on this device</Text>
        </View>
        <View style={s.badge}>
          <View style={s.dot} />
          <Text style={s.badgeText}>Online</Text>
        </View>
      </View>

      <View style={s.tabBar}>
        {[
          { key: TABS.ASK,   label: "Ask AI" },
          { key: TABS.FILES, label: "Files" },
          { key: TABS.NOTES, label: "Notes" },
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

      {/* ── ASK ── */}
      {tab === TABS.ASK && (
        <View style={{ flex: 1 }}>
          <ScrollView ref={scrollRef} style={s.chatScroll} contentContainerStyle={s.chatInner}>
            {chat.length === 0 && (
              <View style={s.emptyChat}>
                <Text style={s.emptyChatIcon}>◈</Text>
                <Text style={s.emptyChatTitle}>Webnix AI</Text>
                <Text style={s.emptyChatSub}>
                  {files.length === 0 && notes.length === 0
                    ? "Ask anything. No files indexed — using general knowledge."
                    : `${files.length} file(s) + ${notes.length} note(s) indexed.`}
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
                    WEBNIX AI{m.source ? ` · ${m.source.toUpperCase()}` : ""}{m.streaming ? " ▌" : ""}
                  </Text>
                )}
                <Text style={[s.bubbleText, m.role === "user" && s.bubbleTextUser]}>
                  {m.content || (m.streaming ? "..." : "")}
                </Text>
                <Text style={[s.bubbleTs, m.role === "user" && { color: "rgba(0,0,0,0.3)" }]}>{m.ts}</Text>
              </View>
            ))}

            {tps && !thinking && (
              <View style={s.tpsBadge}>
                <Text style={s.tpsText}>{tps} tok/s</Text>
              </View>
            )}
          </ScrollView>

          {chat.length > 0 && (
            <View style={s.chatActions}>
              <TouchableOpacity style={s.actionBtn} onPress={saveChat}>
                <Text style={s.actionBtnText}>↓ Save Chat</Text>
              </TouchableOpacity>
              <TouchableOpacity style={s.actionBtn} onPress={clearChat}>
                <Text style={s.actionBtnText}>✕ Clear</Text>
              </TouchableOpacity>
            </View>
          )}

          <View style={s.inputRow}>
            <TextInput
              style={s.input}
              placeholder="Ask Webnix AI..."
              placeholderTextColor="#333"
              value={query}
              onChangeText={setQuery}
              multiline
              maxHeight={100}
              onSubmitEditing={ask}
              blurOnSubmit={false}
            />
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
        </View>
      )}

      {/* ── FILES ── */}
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
                  Indexing {curFile ? `"${curFile.slice(0, 22)}..."` : "..."}
                </Text>
              </View>
            ) : (
              <Text style={s.primaryBtnText}>+ Pick File to Index</Text>
            )}
          </TouchableOpacity>
          <Text style={s.hint}>TXT, MD, CSV, JSON, PDF — stays on device.</Text>

          {files.length === 0 ? (
            <View style={s.empty}>
              <Text style={s.emptyIcon2}>▣</Text>
              <Text style={s.emptyTitle2}>No files indexed</Text>
              <Text style={s.emptyDesc}>Pick a file to make it queryable with AI.</Text>
            </View>
          ) : files.map((f, i) => (
            <View key={i} style={s.card}>
              <View style={s.cardLeft}>
                <Text style={s.cardIcon}>◻</Text>
                <View>
                  <Text style={s.cardName} numberOfLines={1}>{f.name}</Text>
                  <Text style={s.cardMeta}>{f.chunks} chunks · {f.at}</Text>
                </View>
              </View>
              <View style={s.cardBadge}><Text style={s.cardBadgeText}>Indexed</Text></View>
            </View>
          ))}
        </ScrollView>
      )}

      {/* ── NOTES ── */}
      {tab === TABS.NOTES && (
        <ScrollView style={s.scroll} contentContainerStyle={s.scrollInner} keyboardShouldPersistTaps="handled">
          <TextInput
            style={s.titleInput}
            placeholder="Note title (optional)"
            placeholderTextColor="#333"
            value={noteTitle}
            onChangeText={setNoteTitle}
          />
          <TextInput
            style={s.bodyInput}
            placeholder="Write your note — thoughts, research, anything. AI indexes and searches it."
            placeholderTextColor="#333"
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
              <Text style={s.emptyDesc}>Save a note to make it searchable.</Text>
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
  boot:          { flex: 1, backgroundColor: C.bg, alignItems: "center", justifyContent: "center", padding: 32 },
  bootIcon:      { fontSize: 52, color: C.accent, marginBottom: 14 },
  bootTitle:     { fontSize: 28, fontWeight: "700", color: C.text, letterSpacing: 1 },
  bootSub:       { fontSize: 13, color: C.mid, marginTop: 6, letterSpacing: 0.5, marginBottom: 36 },
  progressTrack: { width: "100%", height: 2, backgroundColor: C.border, borderRadius: 2, overflow: "hidden", marginBottom: 10 },
  progressFill:  { height: 2, backgroundColor: C.accent },
  bootMsg:       { fontSize: 12, color: C.mid, fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace", marginBottom: 4 },
  bootProgress:  { fontSize: 18, color: C.accent, fontWeight: "700" },
  bootDevice:    { fontSize: 11, color: C.dim, position: "absolute", bottom: 36 },
  retryBtn:      { marginTop: 24, backgroundColor: C.accent, paddingHorizontal: 32, paddingVertical: 12, borderRadius: 8 },
  retryText:     { color: "#000", fontWeight: "700", fontSize: 14 },

  root:          { flex: 1, backgroundColor: C.bg },
  header:        { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 20, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: C.border },
  headerTitle:   { fontSize: 17, fontWeight: "700", color: C.text, letterSpacing: 0.5 },
  headerSub:     { fontSize: 10, color: C.mid, marginTop: 1 },
  badge:         { flexDirection: "row", alignItems: "center", backgroundColor: "#00FFCC10", paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20, borderWidth: 1, borderColor: "#00FFCC25" },
  dot:           { width: 6, height: 6, borderRadius: 3, backgroundColor: C.accent, marginRight: 6 },
  badgeText:     { fontSize: 11, color: C.accent, fontWeight: "600" },

  tabBar:        { flexDirection: "row", borderBottomWidth: 1, borderBottomColor: C.border },
  tab:           { flex: 1, paddingVertical: 12, alignItems: "center" },
  tabActive:     { borderBottomWidth: 2, borderBottomColor: C.accent },
  tabText:       { fontSize: 13, color: C.mid, fontWeight: "500" },
  tabTextActive: { color: C.accent, fontWeight: "700" },

  chatScroll:    { flex: 1 },
  chatInner:     { padding: 16, paddingBottom: 8 },
  emptyChat:     { alignItems: "center", paddingVertical: 52, paddingHorizontal: 20 },
  emptyChatIcon: { fontSize: 44, color: C.dim, marginBottom: 14 },
  emptyChatTitle:{ fontSize: 20, fontWeight: "700", color: C.mid, marginBottom: 6 },
  emptyChatSub:  { fontSize: 13, color: C.dim, textAlign: "center", lineHeight: 20 },
  chips:         { marginTop: 24, flexDirection: "row", flexWrap: "wrap", justifyContent: "center", gap: 8 },
  chip:          { backgroundColor: "#0a1f18", borderWidth: 1, borderColor: "#0d3d2a", borderRadius: 20, paddingHorizontal: 14, paddingVertical: 7 },
  chipText:      { color: C.accent, fontSize: 12 },

  bubble:        { marginBottom: 12, padding: 14, borderRadius: 18, maxWidth: "86%" },
  bubbleUser:    { alignSelf: "flex-end", backgroundColor: C.accent, borderBottomRightRadius: 4 },
  bubbleAI:      { alignSelf: "flex-start", backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderBottomLeftRadius: 4 },
  bubbleLabel:   { fontSize: 9, color: C.accent, fontWeight: "700", marginBottom: 5, letterSpacing: 1.2 },
  bubbleText:    { fontSize: 15, color: C.text, lineHeight: 23 },
  bubbleTextUser:{ color: "#000" },
  bubbleTs:      { fontSize: 9, color: "rgba(255,255,255,0.2)", marginTop: 5, textAlign: "right" },

  tpsBadge:      { alignSelf: "center", backgroundColor: "#00FFCC12", borderWidth: 1, borderColor: "#00FFCC25", borderRadius: 12, paddingHorizontal: 12, paddingVertical: 4, marginBottom: 8 },
  tpsText:       { fontSize: 11, color: C.accent, fontWeight: "700", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace" },

  chatActions:   { flexDirection: "row", paddingHorizontal: 16, paddingVertical: 6, gap: 8 },
  actionBtn:     { flex: 1, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderRadius: 8, paddingVertical: 8, alignItems: "center" },
  actionBtnText: { fontSize: 12, color: C.mid, fontWeight: "600" },

  inputRow:      { flexDirection: "row", padding: 12, paddingTop: 6, alignItems: "flex-end", borderTopWidth: 1, borderTopColor: C.border, gap: 8 },
  input:         { flex: 1, backgroundColor: C.surface, borderRadius: 22, borderWidth: 1, borderColor: C.border, color: C.text, paddingHorizontal: 16, paddingVertical: 10, fontSize: 15, maxHeight: 100 },
  sendBtn:       { backgroundColor: C.accent, width: 44, height: 44, borderRadius: 22, alignItems: "center", justifyContent: "center" },
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

  card:          { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 14, marginBottom: 10, flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardLeft:      { flexDirection: "row", alignItems: "center", flex: 1 },
  cardIcon:      { fontSize: 20, color: C.accent, marginRight: 12 },
  cardName:      { fontSize: 14, fontWeight: "600", color: C.text, maxWidth: width * 0.5 },
  cardMeta:      { fontSize: 11, color: C.mid, marginTop: 2 },
  cardBadge:     { backgroundColor: "#00FFCC15", paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  cardBadgeText: { fontSize: 11, color: C.accent, fontWeight: "600" },

  titleInput:    { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 15, fontWeight: "600", marginBottom: 8 },
  bodyInput:     { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 14, minHeight: 140, marginBottom: 12, lineHeight: 22 },

  logCard:       { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 12 },
  logLabel:      { fontSize: 10, color: C.mid, fontWeight: "700", marginBottom: 8, letterSpacing: 1 },
  logLine:       { fontSize: 10, color: "#333", fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace", marginBottom: 3, lineHeight: 15 },
});
