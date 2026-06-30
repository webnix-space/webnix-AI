import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  ScrollView, ActivityIndicator, Alert, Platform, SafeAreaView,
  StatusBar, Animated, Dimensions
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import * as Device from 'expo-device';

// ═══════════════════════════════════════════════════════════════
// WEBNIX AI — Production Build
// Real QVAC SDK (named exports), SDK56 legacy FS, tps metering
// ═══════════════════════════════════════════════════════════════

let QVAC_AVAILABLE = true;
let loadModel, completion, unloadModel, ragSaveEmbeddings, ragSearch, LLAMA_3_2_1B_INST_Q4_0, GTE_LARGE_FP16, heartbeat;

try {
  const sdk = require('@qvac/sdk');
  loadModel = sdk.loadModel;
  completion = sdk.completion;
  unloadModel = sdk.unloadModel;
  ragSaveEmbeddings = sdk.ragSaveEmbeddings;
  ragSearch = sdk.ragSearch;
  LLAMA_3_2_1B_INST_Q4_0 = sdk.LLAMA_3_2_1B_INST_Q4_0;
  GTE_LARGE_FP16 = sdk.GTE_LARGE_FP16;
  heartbeat = sdk.heartbeat;
} catch (e) {
  console.error('QVAC SDK not found:', e);
  QVAC_AVAILABLE = false;
}

const { width } = Dimensions.get('window');

// ── Helpers ──────────────────────────────────────────────────
const chunkText = (text, size = 400, overlap = 80) => {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    chunks.push(text.slice(i, i + size));
    i += size - overlap;
  }
  return chunks;
};

const extractTextFromFile = async (uri, mimeType, name) => {
  try {
    if (
      mimeType === 'text/plain' ||
      name.endsWith('.txt') ||
      name.endsWith('.md') ||
      name.endsWith('.csv') ||
      name.endsWith('.json')
    ) {
      return await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.UTF8 });
    }
    const raw = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });
    return `File: ${name}\nType: ${mimeType}\nSize: ${raw.length} bytes\nNote: Binary file — AI will reason about this file based on its name and type.`;
  } catch (e) {
    throw new Error(`Could not read file: ${e.message}`);
  }
};

// ── Tab enum ─────────────────────────────────────────────────
const TABS = { FILES: 'files', NOTES: 'notes', ASK: 'ask' };

// ═══════════════════════════════════════════════════════════════
export default function App() {
  // Model state
  const [llmModelId, setLlmModelId] = useState(null);
  const [embedModelId, setEmbedModelId] = useState(null);
  const [loadingState, setLoadingState] = useState('idle'); // idle | loading | ready | error
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadStatus, setLoadStatus] = useState('');

  // UI state
  const [activeTab, setActiveTab] = useState(TABS.FILES);
  const [logs, setLogs] = useState([]);

  // File intelligence state
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [isIndexing, setIsIndexing] = useState(false);
  const [currentFile, setCurrentFile] = useState(null);

  // Notes state
  const [noteText, setNoteText] = useState('');
  const [noteTitle, setNoteTitle] = useState('');
  const [savedNotes, setSavedNotes] = useState([]);
  const [isSavingNote, setIsSavingNote] = useState(false);

  // Ask state
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [isAnswering, setIsAnswering] = useState(false);
  const [answerSource, setAnswerSource] = useState('');
  const [tps, setTps] = useState(null);

  const scrollRef = useRef(null);
  const answerAnim = useRef(new Animated.Value(0)).current;

  const log = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-30), `${new Date().toLocaleTimeString()} ${msg}`]);
  }, []);

  // ── Boot ───────────────────────────────────────────────────
  useEffect(() => {
    if (!QVAC_AVAILABLE) {
      setLoadingState('error');
      setLoadStatus('QVAC SDK unavailable on this build');
      return;
    }
    bootModels();
    return () => {
      if (llmModelId) unloadModel({ modelId: llmModelId }).catch(() => {});
      if (embedModelId) unloadModel({ modelId: embedModelId }).catch(() => {});
    };
  }, []);

  const bootModels = async () => {
    setLoadingState('loading');
    try {
      setLoadStatus('Checking local SDK worker...');
      await heartbeat();
      log('Heartbeat OK');

      setLoadStatus('Loading language model...');
      log('Loading Llama 3.2 1B...');
      const llmId = await loadModel({
        modelSrc: LLAMA_3_2_1B_INST_Q4_0,
        modelType: 'llm',
        onProgress: p => setLoadProgress(Math.round(p * 55)),
      });
      setLlmModelId(llmId);
      log('Language model ready');

      setLoadStatus('Loading embedding model...');
      log('Loading GTE-Large embedder...');
      const embedId = await loadModel({
        modelSrc: GTE_LARGE_FP16,
        modelType: 'embeddings',
        onProgress: p => setLoadProgress(55 + Math.round(p * 45)),
      });
      setEmbedModelId(embedId);
      log('Embedding model ready');

      setLoadingState('ready');
      setLoadProgress(100);
      setLoadStatus('');
      log('All models active — running on device');
    } catch (err) {
      log(`Boot failed: ${err.message}`);
      setLoadingState('error');
      setLoadStatus(`Failed: ${err.message}`);
    }
  };

  // ── File Indexing ──────────────────────────────────────────
  const pickAndIndexFile = async () => {
    if (!embedModelId || !llmModelId) return Alert.alert('Not Ready', 'Models still loading.');

    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['text/*', 'application/pdf', 'application/json', '*/*'],
        copyToCacheDirectory: true,
      });

      if (result.canceled) return;

      const file = result.assets[0];
      setCurrentFile(file.name);
      setIsIndexing(true);
      log(`Indexing: ${file.name}`);

      const text = await extractTextFromFile(file.uri, file.mimeType, file.name);
      if (!text || text.length < 10) throw new Error('File appears empty or unreadable');

      const chunks = chunkText(text, 400, 80);
      log(`Split into ${chunks.length} chunks`);

      await ragSaveEmbeddings({
        modelId: embedModelId,
        documents: chunks,
        chunk: false,
      });

      setIndexedFiles(prev => [...prev, {
        name: file.name,
        size: file.size,
        chunks: chunks.length,
        indexedAt: new Date().toLocaleTimeString(),
      }]);

      log(`✓ ${file.name} indexed (${chunks.length} chunks)`);
      Alert.alert('Indexed', `"${file.name}" is ready to query.`);
    } catch (err) {
      log(`Index failed: ${err.message}`);
      Alert.alert('Error', err.message);
    } finally {
      setIsIndexing(false);
      setCurrentFile(null);
    }
  };

  // ── Save Note ──────────────────────────────────────────────
  const saveNote = async () => {
    if (!noteText.trim()) return Alert.alert('Empty', 'Write something first.');
    if (!embedModelId) return Alert.alert('Not Ready', 'Models still loading.');

    setIsSavingNote(true);
    const title = noteTitle.trim() || `Note ${savedNotes.length + 1}`;
    log(`Saving note: ${title}`);

    try {
      const fullText = `Title: ${title}\n\n${noteText.trim()}`;
      const chunks = chunkText(fullText, 400, 80);

      await ragSaveEmbeddings({
        modelId: embedModelId,
        documents: chunks,
        chunk: false,
      });

      setSavedNotes(prev => [...prev, {
        title,
        preview: noteText.slice(0, 80),
        savedAt: new Date().toLocaleTimeString(),
      }]);

      log(`✓ Note saved: ${title}`);
      setNoteText('');
      setNoteTitle('');
      Alert.alert('Saved', `"${title}" indexed and ready to query.`);
    } catch (err) {
      log(`Save failed: ${err.message}`);
      Alert.alert('Error', err.message);
    } finally {
      setIsSavingNote(false);
    }
  };

  // ── Ask AI ─────────────────────────────────────────────────
  const askAI = async () => {
    if (!query.trim()) return;
    if (!llmModelId) return Alert.alert('Not Ready', 'Language model loading.');

    setAnswer('');
    setAnswerSource('');
    setTps(null);
    setIsAnswering(true);
    log(`Query: ${query.trim()}`);

    Animated.timing(answerAnim, { toValue: 0, duration: 100, useNativeDriver: true }).start();

    try {
      let context = '';
      let source = 'general knowledge';

      if (embedModelId && (indexedFiles.length > 0 || savedNotes.length > 0)) {
        try {
          const results = await ragSearch({
            modelId: embedModelId,
            query: query.trim(),
            topK: 4,
          });
          if (results?.length && results[0].score > 0.5) {
            context = results.map((r, i) => `[${i + 1}] ${r.content || r.text}`).join('\n\n');
            source = 'your files & notes';
            log(`RAG: ${results.length} matches found`);
          }
        } catch (e) {
          log('RAG search skipped');
        }
      }

      const systemPrompt = context
        ? `You are Webnix AI, a private assistant running entirely on this device. Answer using the provided context. Be direct and concise.\n\nContext:\n${context}`
        : `You are Webnix AI, a private assistant running entirely on this device. Answer concisely and accurately.`;

      const history = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: query.trim() },
      ];

      const result = completion({
        modelId: llmModelId,
        history,
        stream: true,
        temperature: 0.6,
        maxTokens: 300,
      });

      setAnswerSource(source);
      Animated.timing(answerAnim, { toValue: 1, duration: 300, useNativeDriver: true }).start();

      let tokenCount = 0;
      const startTime = Date.now();

      for await (const token of result.tokenStream) {
        tokenCount++;
        setAnswer(prev => prev + token);
        const elapsedSec = (Date.now() - startTime) / 1000;
        if (elapsedSec > 0.3) {
          setTps((tokenCount / elapsedSec).toFixed(1));
        }
      }

      const totalElapsed = (Date.now() - startTime) / 1000;
      if (totalElapsed > 0) setTps((tokenCount / totalElapsed).toFixed(1));

      log(`Answer complete — ${tokenCount} tokens, ${(tokenCount / totalElapsed).toFixed(1)} tok/s`);
    } catch (err) {
      log(`Query failed: ${err.message}`);
      setAnswer('Something went wrong. Try again.');
    } finally {
      setIsAnswering(false);
    }
  };

  // ── Loading Screen ─────────────────────────────────────────
  if (loadingState === 'loading' || loadingState === 'idle') {
    return (
      <View style={styles.bootScreen}>
        <StatusBar barStyle="light-content" backgroundColor="#080808" />
        <View style={styles.bootLogo}>
          <Text style={styles.bootIcon}>◈</Text>
          <Text style={styles.bootTitle}>Webnix AI</Text>
          <Text style={styles.bootSub}>Private Intelligence. On Device.</Text>
        </View>
        <View style={styles.bootProgress}>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${loadProgress}%` }]} />
          </View>
          <Text style={styles.bootStatusText}>{loadStatus}</Text>
        </View>
        <Text style={styles.bootDevice}>{Device.modelName || 'Device'} · Local Silicon</Text>
      </View>
    );
  }

  if (loadingState === 'error') {
    return (
      <View style={styles.bootScreen}>
        <Text style={styles.errorIcon}>✕</Text>
        <Text style={styles.errorTitle}>Failed to Start</Text>
        <Text style={styles.errorMsg}>{loadStatus}</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={bootModels}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Main UI ────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#080808" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>◈ Webnix AI</Text>
          <Text style={styles.headerSub}>All processing on this device</Text>
        </View>
        <View style={styles.headerBadge}>
          <View style={styles.onlineDot} />
          <Text style={styles.onlineText}>Online</Text>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabBar}>
        {[
          { key: TABS.FILES, label: 'Files' },
          { key: TABS.NOTES, label: 'Notes' },
          { key: TABS.ASK, label: 'Ask AI' },
        ].map(tab => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => setActiveTab(tab.key)}
          >
            <Text style={[styles.tabText, activeTab === tab.key && styles.tabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Content */}
      <ScrollView
        style={styles.content}
        contentContainerStyle={styles.contentInner}
        keyboardShouldPersistTaps="handled"
        ref={scrollRef}
      >

        {/* ── FILES TAB ── */}
        {activeTab === TABS.FILES && (
          <View>
            <TouchableOpacity
              style={[styles.primaryBtn, isIndexing && styles.btnDisabled]}
              onPress={pickAndIndexFile}
              disabled={isIndexing}
            >
              {isIndexing ? (
                <View style={styles.btnRow}>
                  <ActivityIndicator size="small" color="#080808" />
                  <Text style={[styles.primaryBtnText, { marginLeft: 8 }]}>
                    Indexing {currentFile ? `"${currentFile.slice(0, 20)}..."` : '...'}
                  </Text>
                </View>
              ) : (
                <Text style={styles.primaryBtnText}>+ Pick File to Index</Text>
              )}
            </TouchableOpacity>

            <Text style={styles.sectionHint}>
              Supports TXT, MD, CSV, JSON, PDF. All processing stays on device.
            </Text>

            {indexedFiles.length === 0 ? (
              <View style={styles.emptyState}>
                <Text style={styles.emptyIcon}>▣</Text>
                <Text style={styles.emptyTitle}>No files indexed yet</Text>
                <Text style={styles.emptyDesc}>Pick a file above to make it queryable with AI.</Text>
              </View>
            ) : (
              indexedFiles.map((f, i) => (
                <View key={i} style={styles.fileCard}>
                  <View style={styles.fileCardLeft}>
                    <Text style={styles.fileCardIcon}>◻</Text>
                    <View>
                      <Text style={styles.fileCardName} numberOfLines={1}>{f.name}</Text>
                      <Text style={styles.fileCardMeta}>{f.chunks} chunks · {f.indexedAt}</Text>
                    </View>
                  </View>
                  <View style={styles.fileCardBadge}>
                    <Text style={styles.fileCardBadgeText}>Indexed</Text>
                  </View>
                </View>
              ))
            )}
          </View>
        )}

        {/* ── NOTES TAB ── */}
        {activeTab === TABS.NOTES && (
          <View>
            <TextInput
              style={styles.noteTitleInput}
              placeholder="Note title (optional)"
              placeholderTextColor="#444"
              value={noteTitle}
              onChangeText={setNoteTitle}
            />
            <TextInput
              style={styles.noteBodyInput}
              placeholder="Write your note here — thoughts, research, meeting notes, anything. AI will index and search it."
              placeholderTextColor="#444"
              value={noteText}
              onChangeText={setNoteText}
              multiline
              textAlignVertical="top"
            />
            <TouchableOpacity
              style={[styles.primaryBtn, isSavingNote && styles.btnDisabled]}
              onPress={saveNote}
              disabled={isSavingNote}
            >
              {isSavingNote ? (
                <View style={styles.btnRow}>
                  <ActivityIndicator size="small" color="#080808" />
                  <Text style={[styles.primaryBtnText, { marginLeft: 8 }]}>Saving...</Text>
                </View>
              ) : (
                <Text style={styles.primaryBtnText}>Save & Index Note</Text>
              )}
            </TouchableOpacity>

            {savedNotes.length === 0 ? (
              <View style={styles.emptyState}>
                <Text style={styles.emptyIcon}>◎</Text>
                <Text style={styles.emptyTitle}>No notes yet</Text>
                <Text style={styles.emptyDesc}>Save a note above to make it searchable with AI.</Text>
              </View>
            ) : (
              savedNotes.map((n, i) => (
                <View key={i} style={styles.fileCard}>
                  <View style={styles.fileCardLeft}>
                    <Text style={styles.fileCardIcon}>◎</Text>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.fileCardName}>{n.title}</Text>
                      <Text style={styles.fileCardMeta} numberOfLines={2}>{n.preview}</Text>
                    </View>
                  </View>
                </View>
              ))
            )}
          </View>
        )}

        {/* ── ASK TAB ── */}
        {activeTab === TABS.ASK && (
          <View>
            {indexedFiles.length === 0 && savedNotes.length === 0 && (
              <View style={styles.contextHint}>
                <Text style={styles.contextHintText}>
                  No files or notes indexed yet. AI will answer from general knowledge only.
                </Text>
              </View>
            )}

            <TextInput
              style={styles.queryInput}
              placeholder="Ask anything about your files, notes, or any topic..."
              placeholderTextColor="#444"
              value={query}
              onChangeText={setQuery}
              multiline
              textAlignVertical="top"
            />

            <TouchableOpacity
              style={[styles.askBtn, isAnswering && styles.btnDisabled]}
              onPress={askAI}
              disabled={isAnswering}
            >
              {isAnswering ? (
                <View style={styles.btnRow}>
                  <ActivityIndicator size="small" color="#00FFCC" />
                  <Text style={[styles.askBtnText, { marginLeft: 8 }]}>Thinking on device...</Text>
                </View>
              ) : (
                <Text style={styles.askBtnText}>Ask</Text>
              )}
            </TouchableOpacity>

            {answer ? (
              <Animated.View style={[styles.answerCard, { opacity: answerAnim }]}>
                <View style={styles.answerHeader}>
                  <Text style={styles.answerLabel}>Answer</Text>
                  <View style={styles.answerMetaRow}>
                    {tps ? <Text style={styles.tpsText}>{tps} tok/s</Text> : null}
                    {answerSource ? (
                      <Text style={styles.answerSource}>from {answerSource}</Text>
                    ) : null}
                  </View>
                </View>
                <Text style={styles.answerText}>{answer}</Text>
              </Animated.View>
            ) : null}

            {/* Recent log */}
            {logs.length > 0 && (
              <View style={styles.logCard}>
                <Text style={styles.logLabel}>Activity</Text>
                {logs.slice(-5).map((l, i) => (
                  <Text key={i} style={styles.logLine}>{l}</Text>
                ))}
              </View>
            )}
          </View>
        )}

      </ScrollView>
    </SafeAreaView>
  );
}

// ═══════════════════════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════════════════════
const C = {
  bg: '#080808',
  surface: '#111111',
  border: '#1E1E1E',
  accent: '#00FFCC',
  accentDim: '#00FFCC18',
  text: '#F0F0F0',
  textMid: '#888',
  textDim: '#444',
  danger: '#FF4444',
};

const styles = StyleSheet.create({
  // Boot
  bootScreen: { flex: 1, backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center', padding: 32 },
  bootLogo: { alignItems: 'center', marginBottom: 48 },
  bootIcon: { fontSize: 48, color: C.accent, marginBottom: 12 },
  bootTitle: { fontSize: 28, fontWeight: '700', color: C.text, letterSpacing: 1 },
  bootSub: { fontSize: 13, color: C.textMid, marginTop: 6, letterSpacing: 0.5 },
  bootProgress: { width: '100%', alignItems: 'center', marginBottom: 24 },
  progressTrack: { width: '100%', height: 2, backgroundColor: C.border, borderRadius: 2, marginBottom: 12, overflow: 'hidden' },
  progressFill: { height: 2, backgroundColor: C.accent, borderRadius: 2 },
  bootStatusText: { fontSize: 12, color: C.textMid, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  bootDevice: { fontSize: 11, color: C.textDim, position: 'absolute', bottom: 40 },
  errorIcon: { fontSize: 40, color: C.danger, marginBottom: 12 },
  errorTitle: { fontSize: 20, fontWeight: '700', color: C.text, marginBottom: 8 },
  errorMsg: { fontSize: 13, color: C.textMid, textAlign: 'center', marginBottom: 24 },
  retryBtn: { backgroundColor: C.accent, paddingHorizontal: 32, paddingVertical: 12, borderRadius: 8 },
  retryText: { color: C.bg, fontWeight: '700', fontSize: 14 },

  // Layout
  root: { flex: 1, backgroundColor: C.bg },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: C.border },
  headerTitle: { fontSize: 18, fontWeight: '700', color: C.text, letterSpacing: 0.5 },
  headerSub: { fontSize: 11, color: C.textMid, marginTop: 2 },
  headerBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#00FFCC10', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20, borderWidth: 1, borderColor: '#00FFCC30' },
  onlineDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: C.accent, marginRight: 6 },
  onlineText: { fontSize: 11, color: C.accent, fontWeight: '600' },

  // Tabs
  tabBar: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: C.border },
  tab: { flex: 1, paddingVertical: 13, alignItems: 'center' },
  tabActive: { borderBottomWidth: 2, borderBottomColor: C.accent },
  tabText: { fontSize: 13, color: C.textMid, fontWeight: '500' },
  tabTextActive: { color: C.accent, fontWeight: '700' },

  // Content
  content: { flex: 1 },
  contentInner: { padding: 16, paddingBottom: 40 },

  // Buttons
  primaryBtn: { backgroundColor: C.accent, paddingVertical: 14, borderRadius: 10, alignItems: 'center', marginBottom: 10 },
  primaryBtnText: { color: C.bg, fontWeight: '700', fontSize: 15 },
  askBtn: { backgroundColor: 'transparent', borderWidth: 1.5, borderColor: C.accent, paddingVertical: 14, borderRadius: 10, alignItems: 'center', marginBottom: 16 },
  askBtnText: { color: C.accent, fontWeight: '700', fontSize: 15 },
  btnDisabled: { opacity: 0.5 },
  btnRow: { flexDirection: 'row', alignItems: 'center' },

  // Inputs
  noteTitleInput: { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 15, fontWeight: '600', marginBottom: 8 },
  noteBodyInput: { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 14, minHeight: 140, marginBottom: 12, lineHeight: 22 },
  queryInput: { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, color: C.text, padding: 12, fontSize: 14, minHeight: 80, marginBottom: 12, lineHeight: 22 },

  // Cards
  fileCard: { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 14, marginBottom: 10, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  fileCardLeft: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  fileCardIcon: { fontSize: 20, color: C.accent, marginRight: 12 },
  fileCardName: { fontSize: 14, fontWeight: '600', color: C.text, maxWidth: width * 0.5 },
  fileCardMeta: { fontSize: 11, color: C.textMid, marginTop: 2 },
  fileCardBadge: { backgroundColor: C.accentDim, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  fileCardBadgeText: { fontSize: 11, color: C.accent, fontWeight: '600' },

  // Answer
  answerCard: { backgroundColor: C.accentDim, borderRadius: 10, borderWidth: 1, borderColor: '#00FFCC30', padding: 16, marginBottom: 16 },
  answerHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  answerLabel: { fontSize: 12, fontWeight: '700', color: C.accent, letterSpacing: 0.5, textTransform: 'uppercase' },
  answerMetaRow: { flexDirection: 'row', alignItems: 'center' },
  tpsText: { fontSize: 11, color: C.accent, fontWeight: '700', marginRight: 8, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  answerSource: { fontSize: 11, color: C.textMid },
  answerText: { fontSize: 14, color: C.text, lineHeight: 22 },

  // Hints & empty
  contextHint: { backgroundColor: '#FFCC0010', borderWidth: 1, borderColor: '#FFCC0030', borderRadius: 8, padding: 12, marginBottom: 14 },
  contextHintText: { fontSize: 12, color: '#FFCC00', lineHeight: 18 },
  sectionHint: { fontSize: 12, color: C.textMid, marginBottom: 16, lineHeight: 18 },
  emptyState: { alignItems: 'center', paddingVertical: 48 },
  emptyIcon: { fontSize: 36, color: C.textDim, marginBottom: 12 },
  emptyTitle: { fontSize: 15, fontWeight: '600', color: C.textMid, marginBottom: 6 },
  emptyDesc: { fontSize: 13, color: C.textDim, textAlign: 'center', lineHeight: 20 },

  // Log
  logCard: { backgroundColor: C.surface, borderRadius: 8, borderWidth: 1, borderColor: C.border, padding: 12, marginTop: 8 },
  logLabel: { fontSize: 11, color: C.textMid, fontWeight: '600', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 },
  logLine: { fontSize: 10, color: '#555', fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace', marginBottom: 3, lineHeight: 15 },
});
