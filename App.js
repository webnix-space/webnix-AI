import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  ScrollView, ActivityIndicator, Alert, Platform
} from 'react-native';

// ═══════════════════════════════════════════════════════════════
// SNACK-COMPATIBLE VERSION (Web Preview)
// This uses mock QVAC for Expo Snack browser preview
// For real device demo, use the full App.js with @qvac/sdk imports
// ═══════════════════════════════════════════════════════════════

const isSnackWeb = Platform.OS === 'web';

// Mock QVAC SDK for Snack preview
const mockQVAC = {
  LLAMA_3_2_1B_INST_Q4_0: 'mock-llm-model',
  GTE_LARGE_FP16: 'mock-embed-model',

  loadModel: async ({ modelSrc, modelType, onProgress }) => {
    // Simulate loading progress
    for (let i = 0; i <= 10; i++) {
      await new Promise(r => setTimeout(r, 200));
      onProgress?.(i / 10);
    }
    return `mock-${modelType}-id-${Date.now()}`;
  },

  unloadModel: async ({ modelId }) => {
    console.log('Mock unload:', modelId);
  },

  completion: ({ modelId, history, stream }) => {
    const mockResponse = `[Snack Preview Mode] This is a simulated response. 
In the real app on a physical device, this would stream tokens from 
Llama 3.2 1B running locally via QVAC SDK.

Model: ${modelId}
Platform: ${Platform.OS}`;

    return {
      tokenStream: (async function* () {
        const tokens = mockResponse.split(' ');
        for (const token of tokens) {
          await new Promise(r => setTimeout(r, 50));
          yield token + ' ';
        }
      })()
    };
  },

  ragSaveEmbeddings: async ({ modelId, documents }) => {
    await new Promise(r => setTimeout(r, 300));
    return documents.map((d, i) => ({ id: i, text: d.slice(0, 20) }));
  },

  ragSearch: async ({ modelId, query, topK }) => {
    await new Promise(r => setTimeout(r, 200));
    return [
      { text: `Mock result for: "${query}"`, score: 0.95 },
      { text: 'Simulated vector match from local index', score: 0.87 },
    ];
  },

  config: {
    setLogLevel: () => {}
  }
};

// Use mock or real QVAC based on environment
const QVAC = isSnackWeb ? mockQVAC : null;
// Note: In real device build, import from '@qvac/sdk' instead

export default function App() {
  const [loading, setLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [llmModelId, setLlmModelId] = useState(null);
  const [embedModelId, setEmbedModelId] = useState(null);
  const [logText, setLogText] = useState('');
  const [queryText, setQueryText] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [systemLogs, setSystemLogs] = useState([]);
  const [isInferencing, setIsInferencing] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const scrollViewRef = useRef(null);

  const writeSystemLog = useCallback((message) => {
    const ts = new Date().toISOString().split('T')[1].slice(0, -1);
    setSystemLogs((prev) => [...prev, `[${ts}] ${message}`]);
  }, []);

  useEffect(() => {
    let mounted = true;

    async function initializeNode() {
      try {
        writeSystemLog('Bootstrapping QVAC runtime...');

        if (isSnackWeb) {
          writeSystemLog('⚠️ Snack web preview detected — using mock QVAC');
        }

        // Load LLM
        writeSystemLog('Loading LLM (Llama 3.2 1B Q4_0)...');
        const llmId = await QVAC.loadModel({
          modelSrc: QVAC.LLAMA_3_2_1B_INST_Q4_0,
          modelType: 'llm',
          onProgress: (p) => mounted && setLoadProgress(Math.round(p * 50)),
        });
        if (mounted) {
          setLlmModelId(llmId);
          writeSystemLog(`LLM loaded. ID: ${llmId.slice(0, 8)}...`);
        }

        // Load Embedder
        writeSystemLog('Loading embedding model (GTE-Large FP16)...');
        const embedId = await QVAC.loadModel({
          modelSrc: QVAC.GTE_LARGE_FP16,
          modelType: 'embeddings',
          onProgress: (p) => mounted && setLoadProgress(50 + Math.round(p * 50)),
        });
        if (mounted) {
          setEmbedModelId(embedId);
          writeSystemLog(`Embedder loaded. ID: ${embedId.slice(0, 8)}...`);
          setLoading(false);
          writeSystemLog(isSnackWeb 
            ? 'Node online. (Snack preview mode — mock inference)' 
            : 'Node online. All models active on local silicon.'
          );
        }
      } catch (error) {
        writeSystemLog(`Node bootstrap failed: ${String(error)}`);
        if (mounted) setLoading(false);
      }
    }

    initializeNode();

    return () => {
      mounted = false;
      if (llmModelId) QVAC.unloadModel({ modelId: llmModelId });
      if (embedModelId) QVAC.unloadModel({ modelId: embedModelId });
    };
  }, []);

  const handleSaveLog = async () => {
    if (!logText.trim()) return Alert.alert('Input Required', 'Enter mesh data.');
    if (!embedModelId) return Alert.alert('Not Ready', 'Embedding model loading.');

    setIsIndexing(true);
    writeSystemLog('Quantizing local log frame signatures...');

    try {
      const docs = await QVAC.ragSaveEmbeddings({
        modelId: embedModelId,
        documents: [logText.trim()],
        chunk: false,
      });
      writeSystemLog(`Indexed ${docs.length} vector block(s)`);
      writeSystemLog('Data block securely committed to local vector partition.');
      Alert.alert('Indexed', `Stored ${logText.length} chars to local vectors.`);
      setLogText('');
    } catch (error) {
      writeSystemLog(`Vector Store Exception: ${String(error)}`);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleAskAI = async () => {
    if (!queryText.trim()) return Alert.alert('Input Required', 'Enter a query.');
    if (!llmModelId) return Alert.alert('Not Ready', 'LLM still loading.');

    setAiResponse('');
    setIsInferencing(true);
    writeSystemLog('Scheduling inference thread queue for prompt...');

    try {
      // RAG search
      let context = '';
      if (embedModelId) {
        try {
          const results = await QVAC.ragSearch({
            modelId: embedModelId,
            query: queryText.trim(),
            topK: 3,
          });
          if (results?.length) {
            context = results.map((r, i) => `[Context ${i+1}] ${r.text}`).join('\n');
            writeSystemLog(`RAG retrieved ${results.length} vector match(es)`);
          }
        } catch (e) {
          writeSystemLog('RAG search skipped (empty index)');
        }
      }

      const history = [
        { role: 'system', content: `You are a secure mesh node analyst. ${context ? 'Use context.' : 'General knowledge.'} Be concise.` },
        ...(context ? [{ role: 'user', content: context }] : []),
        { role: 'user', content: queryText.trim() },
      ];

      const result = QVAC.completion({
        modelId: llmModelId,
        history,
        stream: true,
        temperature: 0.7,
        maxTokens: 256,
      });

      writeSystemLog('Inference routine active on device silicon.');

      let tokenCount = 0;
      for await (const token of result.tokenStream) {
        tokenCount++;
        setAiResponse((prev) => prev + token);
      }

      writeSystemLog(`Inference complete. ${tokenCount} tokens generated.`);
    } catch (error) {
      writeSystemLog(`Inference Thread Failure: ${String(error)}`);
    } finally {
      setIsInferencing(false);
    }
  };

  const exportAuditLog = () => {
    const audit = {
      project: 'AirDrop AI Mesh Node',
      mode: isSnackWeb ? 'snack-preview' : 'production',
      platform: Platform.OS,
      timestamp: new Date().toISOString(),
      note: isSnackWeb ? 'Mock data — use real device for actual performance' : 'Real QVAC SDK',
    };
    console.log(JSON.stringify(audit, null, 2));
    Alert.alert('Audit Log', isSnackWeb 
      ? 'Mock audit log exported. Use real device for actual metrics.' 
      : 'Performance audit log exported.'
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#00FFCC" />
        <Text style={styles.loadingText}>Compiling Edge AI Framework...</Text>
        <Text style={styles.progressText}>{loadProgress}%</Text>
        {isSnackWeb && <Text style={styles.snackWarning}>⚠️ Snack Preview Mode</Text>}
      </View>
    );
  }

  return (
    <View style={styles.appWrapper}>
      <Text style={styles.mainTitle}>📡 AirDrop AI Mesh Node</Text>
      {isSnackWeb && (
        <View style={styles.snackBanner}>
          <Text style={styles.snackBannerText}>🔧 Snack Preview — Mock QVAC Mode</Text>
        </View>
      )}

      <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">

        <View style={[styles.featureCard, styles.statusCard]}>
          <Text style={styles.featureTitle}>Node Status</Text>
          <Text style={styles.statusText}>LLM: {llmModelId ? `✅ ${llmModelId.slice(0,8)}...` : '❌ Offline'}</Text>
          <Text style={styles.statusText}>Embedder: {embedModelId ? `✅ ${embedModelId.slice(0,8)}...` : '❌ Offline'}</Text>
          <Text style={styles.statusText}>Platform: {Platform.OS} | Mode: {isSnackWeb ? 'Snack Preview' : 'Local Silicon'}</Text>
        </View>

        <View style={styles.featureCard}>
          <Text style={styles.featureTitle}>Stage Local Node Information Frame</Text>
          <TextInput style={styles.inputArea} placeholder="Enter local operational mesh data..." 
            placeholderTextColor="#555" value={logText} onChangeText={setLogText} multiline />
          <TouchableOpacity style={[styles.actionButton, isIndexing && styles.disabledButton]} 
            onPress={handleSaveLog} disabled={isIndexing}>
            <Text style={styles.actionButtonText}>{isIndexing ? 'Indexing...' : 'Index into Local Vectors'}</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.featureCard}>
          <Text style={styles.featureTitle}>Compute Local Swarm Inference</Text>
          <TextInput style={styles.inputArea} placeholder="Query local peer node intelligence..." 
            placeholderTextColor="#555" value={queryText} onChangeText={setQueryText} />
          <TouchableOpacity style={[styles.actionButton, styles.aiButtonOverride, isInferencing && styles.disabledButton]} 
            onPress={handleAskAI} disabled={isInferencing}>
            <Text style={[styles.actionButtonText, {color: '#000'}]}>{isInferencing ? 'Inferencing...' : 'Query Local Silicon'}</Text>
          </TouchableOpacity>
        </View>

        {aiResponse ? (
          <View style={styles.intelligenceOutputBox}>
            <Text style={styles.intelligenceTitle}>Swarm Intelligence Yield:</Text>
            <Text style={styles.intelligenceContent}>{aiResponse}</Text>
          </View>
        ) : null}

        <View style={[styles.featureCard, {marginBottom: 40}]}>
          <Text style={[styles.featureTitle, {color: '#FFCC00'}]}>Verification Log Outputs (Artifact Bundle)</Text>
          <ScrollView style={styles.terminalConsole} ref={scrollViewRef}
            onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({animated: true})}>
            {systemLogs.map((log, i) => <Text key={i} style={styles.terminalLine}>{log}</Text>)}
          </ScrollView>
          <TouchableOpacity style={[styles.actionButton, {marginTop: 10, backgroundColor: '#333'}]} onPress={exportAuditLog}>
            <Text style={styles.actionButtonText}>Export Audit Log</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  appWrapper: { flex: 1, backgroundColor: '#0A0A0A' },
  scrollContainer: { padding: 16 },
  mainTitle: { fontSize: 20, fontWeight: 'bold', color: '#FFF', textAlign: 'center', marginBottom: 15, marginTop: 45 },
  loadingContainer: { flex: 1, backgroundColor: '#0A0A0A', alignItems: 'center', justifyContent: 'center' },
  loadingText: { color: '#888', marginTop: 12, fontSize: 13, fontFamily: 'monospace' },
  progressText: { color: '#00FFCC', marginTop: 8, fontSize: 16, fontFamily: 'monospace' },
  snackWarning: { color: '#FFCC00', marginTop: 12, fontSize: 11, textAlign: 'center' },
  snackBanner: { backgroundColor: '#FFCC0020', padding: 8, marginHorizontal: 16, marginTop: 8, borderRadius: 6, borderWidth: 1, borderColor: '#FFCC0040' },
  snackBannerText: { color: '#FFCC00', fontSize: 11, textAlign: 'center', fontWeight: '600' },
  featureCard: { backgroundColor: '#141414', padding: 14, borderRadius: 8, marginBottom: 12, borderWidth: 1, borderColor: '#1F1F1F' },
  statusCard: { borderColor: '#00FFCC40' },
  featureTitle: { color: '#EEE', fontSize: 13, fontWeight: '600', marginBottom: 8 },
  statusText: { color: '#AAA', fontSize: 12, fontFamily: 'monospace', marginBottom: 4 },
  inputArea: { backgroundColor: '#1C1C1C', borderRadius: 6, padding: 10, color: '#FFF', minHeight: 45, marginBottom: 10, fontSize: 13, textAlignVertical: 'top' },
  actionButton: { backgroundColor: '#262626', paddingVertical: 11, borderRadius: 6, alignItems: 'center' },
  actionButtonText: { color: '#FFF', fontWeight: 'bold', fontSize: 13 },
  aiButtonOverride: { backgroundColor: '#00FFCC' },
  disabledButton: { opacity: 0.5 },
  intelligenceOutputBox: { backgroundColor: '#00FFCC05', borderColor: '#00FFCC', borderWidth: 1, padding: 14, borderRadius: 8, marginBottom: 12 },
  intelligenceTitle: { color: '#00FFCC', fontWeight: 'bold', fontSize: 12, marginBottom: 4 },
  intelligenceContent: { color: '#FFF', fontSize: 13, lineHeight: 18 },
  terminalConsole: { maxHeight: 140, backgroundColor: '#000', borderRadius: 6, padding: 8, borderWidth: 1, borderColor: '#333' },
  terminalLine: { color: '#00FF00', fontFamily: 'monospace', fontSize: 10, marginBottom: 3 },
});
