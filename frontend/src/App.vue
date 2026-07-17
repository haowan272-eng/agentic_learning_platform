<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive } from "vue";
import {
  ArrowRight,
  BookOpenCheck,
  Bot,
  Brain,
  CheckCircle2,
  Database,
  FileUp,
  Gauge,
  Layers3,
  LoaderCircle,
  LogOut,
  MessageSquareText,
  Network,
  PanelLeft,
  Plus,
  RefreshCw,
  SearchCheck,
  ShieldCheck,
  Sparkles,
  Target,
  UploadCloud,
  XCircle
} from "@lucide/vue";
import {
  answerQuestionStream,
  cancelAgentTask,
  createAgentTask,
  createKnowledgeBase,
  getDocumentProgress,
  listAgentEvents,
  listAgentTasks,
  listDocuments,
  listKnowledgeBases,
  login,
  reindexDocument,
  register,
  streamAgentEvents,
  tokenStore,
  uploadDocument,
  type AnswerQuestionPayload,
  type AgentEvent,
  type AgentTask,
  type AnswerResponse,
  type DocumentProgress,
  type DocumentItem,
  type KnowledgeBase
} from "@/services/api";

type Mode = {
  key: string;
  label: string;
  prompt: string;
  helper: string;
};

const modes: Mode[] = [
  {
    key: "deep-dive",
    label: "知识深挖",
    helper: "把简历或岗位要求拆成高频追问",
    prompt: "请基于我的知识库，围绕这个面试主题生成 5 个高质量追问，并给出每题的答题结构："
  },
  {
    key: "mock",
    label: "模拟面试",
    helper: "先问后评，训练表达闭环",
    prompt: "请扮演资深面试官，基于知识库对我进行模拟面试。先提出一个问题，然后说明优秀回答应覆盖哪些点："
  },
  {
    key: "gap",
    label: "能力补差",
    helper: "定位薄弱点并生成复习路线",
    prompt: "请根据我的目标岗位和知识库，找出我目前最需要补齐的知识点，并按优先级给出学习建议："
  },
  {
    key: "resume",
    label: "项目复盘",
    helper: "把项目经历转成可讲清的面试故事",
    prompt: "请基于我的项目资料，帮我整理一段适合面试表达的 STAR 项目复盘，并列出可能被追问的问题："
  }
];

const auth = reactive({
  username: tokenStore.username(),
  password: "",
  mode: "login" as "login" | "register",
  loading: false,
  error: "",
  notice: ""
});

const state = reactive({
  token: tokenStore.access(),
  kbs: [] as KnowledgeBase[],
  docs: [] as DocumentItem[],
  docProgresses: {} as Record<number, DocumentProgress>,
  tasks: [] as AgentTask[],
  events: [] as AgentEvent[],
  selectedKbId: 0,
  selectedDocId: 0,
  newKbName: "面试提优资料库",
  newKbDescription: "简历、岗位 JD、项目复盘、八股笔记与模拟面试记录。",
  query: modes[0].prompt,
  topK: 6,
  bm25Weight: 0.4,
  useMemory: true,
  rewriteQuery: true,
  answer: null as AnswerResponse | null,
  conversationId: 0,
  selectedMode: modes[0],
  busy: false,
  uploading: false,
  docBusyId: 0,
  taskBusy: false,
  agentStreaming: false,
  error: ""
});

const isAuthed = computed(() => Boolean(state.token));
const selectedKb = computed(() => state.kbs.find((kb) => kb.id === state.selectedKbId));
const selectedDoc = computed(() => state.docs.find((doc) => doc.id === state.selectedDocId));
const activeTask = computed(() => state.tasks[0]);
const readyDocs = computed(() => state.docs.filter((doc) => isIndexedStatus(effectiveDocStatus(doc))).length);
const failedDocs = computed(() => state.docs.filter((doc) => effectiveDocStatus(doc) === "failed").length);
const selectedDocStatus = computed(() => selectedDoc.value ? effectiveDocStatus(selectedDoc.value) : "");
const selectedDocCanSearch = computed(() => !selectedDoc.value || isIndexedStatus(selectedDocStatus.value));
const selectedDocProgress = computed(() => selectedDoc.value ? progressFor(selectedDoc.value) : null);
let documentProgressTimer: number | undefined;
let agentStreamAbort: AbortController | null = null;
let activeAgentStreamTaskId = "";

function messageFromError(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}

function isIndexedStatus(status: string) {
  return status === "indexed" || status === "completed";
}

function isActiveIndexStatus(status: string) {
  return ["uploaded", "queued", "waiting_lock", "retrying", "indexing", "parsing", "embedding"].includes(status);
}

function effectiveDocStatus(doc: DocumentItem) {
  return state.docProgresses[doc.id]?.status || doc.status;
}

function progressFor(doc: DocumentItem) {
  const status = effectiveDocStatus(doc);
  return state.docProgresses[doc.id] ?? {
    document_id: doc.id,
    status,
    document_status: doc.status,
    percent: isIndexedStatus(status) ? 100 : status === "failed" ? 100 : 0
  };
}

function progressLabel(progress: DocumentProgress) {
  const labels: Record<string, string> = {
    uploaded: "等待入队",
    queued: "队列中",
    waiting_lock: "等待锁",
    retrying: "重试中",
    indexing: "索引中",
    parsing: "解析文档",
    embedding: "写入向量",
    indexed: "已索引",
    completed: "已完成",
    failed: "失败"
  };
  return labels[progress.status] ?? progress.status;
}

function upsertAgentEvent(event: AgentEvent) {
  const index = state.events.findIndex((item) => item.event_index === event.event_index);
  if (index >= 0) state.events[index] = event;
  else state.events.push(event);
  state.events.sort((a, b) => a.event_index - b.event_index);
}

function stopAgentStream() {
  agentStreamAbort?.abort();
  agentStreamAbort = null;
  activeAgentStreamTaskId = "";
  state.agentStreaming = false;
}

function isActiveAgentStatus(status: string) {
  return ["pending", "running", "waiting_user"].includes(status);
}

function connectAgentStream(taskId: string) {
  if (activeAgentStreamTaskId === taskId && agentStreamAbort) return;
  stopAgentStream();
  const controller = new AbortController();
  agentStreamAbort = controller;
  activeAgentStreamTaskId = taskId;
  state.agentStreaming = true;
  const afterIndex = Math.max(0, ...state.events.map((event) => event.event_index));
  void streamAgentEvents(taskId, afterIndex, {
    onEvent(event) {
      upsertAgentEvent(event);
    }
  }, controller.signal)
    .catch((error) => {
      if (error instanceof DOMException && error.name === "AbortError") return;
      state.error = messageFromError(error, "Agent 事件流连接失败。");
    })
    .finally(async () => {
      if (activeAgentStreamTaskId === taskId) {
        agentStreamAbort = null;
        activeAgentStreamTaskId = "";
        state.agentStreaming = false;
        await refreshTasks();
      }
    });
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function emptyAnswer(payload: AnswerQuestionPayload): AnswerResponse {
  return {
    query: payload.query,
    rewritten_query: null,
    answer: "",
    conversation_id: payload.conversation_id ?? 0,
    citations: [],
    retrieved_count: 0,
    degraded: false,
    context_compacted: false,
    timings_ms: {}
  };
}

function createTypewriter(onText: (text: string) => void) {
  let queue = "";
  let running = true;

  const loop = async () => {
    while (running || queue.length > 0) {
      if (!queue) {
        await wait(16);
        continue;
      }
      const size = queue.length > 160 ? 5 : queue.length > 60 ? 3 : 1;
      const chunk = queue.slice(0, size);
      queue = queue.slice(size);
      onText(chunk);
      await wait(size > 1 ? 6 : 14);
    }
  };

  const task = loop();
  return {
    push(text: string) {
      queue += text;
    },
    async stop() {
      running = false;
      await task;
    }
  };
}

async function submitAuth() {
  auth.error = "";
  auth.notice = "";
  if (!auth.username.trim() || !auth.password) {
    auth.error = "请输入用户名和密码。";
    return;
  }
  auth.loading = true;
  try {
    if (auth.mode === "register") {
      await register(auth.username.trim(), auth.password);
      auth.mode = "login";
      auth.password = "";
      auth.notice = "账号已创建，请登录。";
      return;
    }
    const tokens = await login(auth.username.trim(), auth.password);
    tokenStore.set(tokens, auth.username.trim());
    state.token = tokens.access_token;
    await bootstrap();
  } catch (error) {
    auth.error = messageFromError(error, auth.mode === "register" ? "注册失败。" : "登录失败。");
  } finally {
    auth.loading = false;
  }
}

function logout() {
  stopAgentStream();
  if (documentProgressTimer !== undefined) {
    window.clearInterval(documentProgressTimer);
    documentProgressTimer = undefined;
  }
  tokenStore.clear();
  state.token = "";
  state.kbs = [];
  state.docs = [];
  state.docProgresses = {};
  state.tasks = [];
  state.events = [];
  state.answer = null;
}

async function bootstrap() {
  state.error = "";
  await Promise.all([refreshKbs(), refreshTasks()]);
}

async function refreshKbs() {
  state.kbs = await listKnowledgeBases();
  if (!state.selectedKbId && state.kbs.length) state.selectedKbId = state.kbs[0].id;
  await refreshDocs();
}

async function refreshDocs() {
  state.docs = await listDocuments(state.selectedKbId || undefined);
  if (state.selectedDocId && !state.docs.some((doc) => doc.id === state.selectedDocId)) state.selectedDocId = 0;
  await refreshDocumentProgresses();
  updateDocumentProgressPolling();
}

async function refreshDocumentProgresses() {
  const targets = state.docs.filter((doc) => {
    const status = effectiveDocStatus(doc);
    return isActiveIndexStatus(status) || status === "failed" || state.docProgresses[doc.id];
  });
  await Promise.all(targets.map(async (doc) => {
    try {
      state.docProgresses[doc.id] = await getDocumentProgress(doc.id);
    } catch {
      delete state.docProgresses[doc.id];
    }
  }));
}

function updateDocumentProgressPolling() {
  const hasActive = state.docs.some((doc) => isActiveIndexStatus(effectiveDocStatus(doc)));
  if (hasActive && documentProgressTimer === undefined) {
    documentProgressTimer = window.setInterval(async () => {
      await refreshDocumentProgresses();
      if (!state.docs.some((doc) => isActiveIndexStatus(effectiveDocStatus(doc)))) {
        window.clearInterval(documentProgressTimer);
        documentProgressTimer = undefined;
        await refreshDocs();
      }
    }, 1600);
  }
  if (!hasActive && documentProgressTimer !== undefined) {
    window.clearInterval(documentProgressTimer);
    documentProgressTimer = undefined;
  }
}

async function addKb() {
  if (!state.newKbName.trim()) return;
  state.busy = true;
  try {
    const kb = await createKnowledgeBase(state.newKbName.trim(), state.newKbDescription.trim());
    state.kbs.unshift(kb);
    state.selectedKbId = kb.id;
    await refreshDocs();
  } catch (error) {
    state.error = messageFromError(error, "创建知识库失败。");
  } finally {
    state.busy = false;
  }
}

async function onUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  state.uploading = true;
  state.error = "";
  try {
    const uploaded = await uploadDocument(file, state.selectedKbId || undefined);
    await refreshDocs();
    if (uploaded?.id) {
      state.docProgresses[uploaded.id] = await getDocumentProgress(uploaded.id);
      updateDocumentProgressPolling();
    }
  } catch (error) {
    state.error = messageFromError(error, "上传失败。");
  } finally {
    state.uploading = false;
    input.value = "";
  }
}

function selectMode(mode: Mode) {
  state.selectedMode = mode;
  state.query = `${mode.prompt}\n\n`;
}

async function retryDocumentIndex(doc: DocumentItem) {
  state.docBusyId = doc.id;
  state.error = "";
  try {
    await reindexDocument(doc.id);
    state.docProgresses[doc.id] = await getDocumentProgress(doc.id);
    await refreshDocs();
    updateDocumentProgressPolling();
  } catch (error) {
    state.error = messageFromError(error, "重建索引失败。");
  } finally {
    state.docBusyId = 0;
  }
}

async function askRag() {
  if (!state.query.trim()) return;
  if (!selectedDocCanSearch.value) {
    state.error = `文档「${selectedDoc.value?.file_name ?? ""}」当前状态为 ${selectedDocProgress.value ? progressLabel(selectedDocProgress.value) : "不可用"}，完成索引后才能检索。`;
    return;
  }
  state.busy = true;
  state.error = "";
  const payload: AnswerQuestionPayload = {
    query: state.query.trim(),
    kb_id: state.selectedKbId || undefined,
    document_id: state.selectedDocId || undefined,
    conversation_id: state.conversationId || undefined,
    top_k: Number(state.topK),
    bm25_weight: Number(state.bm25Weight),
    use_memory: state.useMemory,
    rewrite_query: state.rewriteQuery
  };
  state.answer = emptyAnswer(payload);
  const typewriter = createTypewriter((text) => {
    if (state.answer) state.answer.answer += text;
  });
  try {
    const result = await answerQuestionStream(payload, {
      onToken(delta) {
        typewriter.push(delta);
      },
      onFinal(finalAnswer) {
        if (state.answer) {
          state.answer.rewritten_query = finalAnswer.rewritten_query;
          state.answer.conversation_id = finalAnswer.conversation_id;
          state.answer.citations = finalAnswer.citations;
          state.answer.retrieved_count = finalAnswer.retrieved_count;
          state.answer.degraded = finalAnswer.degraded;
          state.answer.context_compacted = finalAnswer.context_compacted;
          state.answer.timings_ms = finalAnswer.timings_ms;
        }
      }
    });
    await typewriter.stop();
    state.answer = result;
    state.conversationId = result.conversation_id;
  } catch (error) {
    await typewriter.stop();
    state.error = messageFromError(error, "问答失败。");
  } finally {
    state.busy = false;
  }
}

async function startAgentReview() {
  const input = state.answer
    ? `请基于当前 RAG 问答结果，为我生成一份面试提优反馈：指出回答亮点、薄弱点、追问清单和下一轮训练建议。\n\n问题：${state.answer.query}\n\n回答：${state.answer.answer}`
    : `请基于知识库「${selectedKb.value?.name ?? "当前资料"}」生成一轮面试提优训练计划，包含知识盲区、模拟问题和复习优先级。`;
  state.taskBusy = true;
  state.error = "";
  try {
    const task = await createAgentTask({
      user_input: input,
      kb_id: state.selectedKbId || undefined,
      conversation_id: state.conversationId || undefined,
      max_steps: 8,
      max_tool_calls: 12
    });
    await refreshTasks();
    await refreshEvents(task.task_id);
    connectAgentStream(task.task_id);
  } catch (error) {
    state.error = messageFromError(error, "Agent 任务创建失败。");
  } finally {
    state.taskBusy = false;
  }
}

async function refreshTasks() {
  state.tasks = (await listAgentTasks()).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  if (state.tasks[0]) {
    await refreshEvents(state.tasks[0].task_id);
    if (isActiveAgentStatus(state.tasks[0].status)) connectAgentStream(state.tasks[0].task_id);
  }
}

async function refreshEvents(taskId?: string) {
  if (!taskId) return;
  state.events = await listAgentEvents(taskId);
}

async function cancelActiveTask() {
  if (!activeTask.value) return;
  state.taskBusy = true;
  try {
    await cancelAgentTask(activeTask.value.task_id);
    stopAgentStream();
    await refreshTasks();
  } catch (error) {
    state.error = messageFromError(error, "取消任务失败。");
  } finally {
    state.taskBusy = false;
  }
}

onMounted(() => {
  if (state.token) void bootstrap().catch((error) => {
    state.error = messageFromError(error, "初始化失败。");
  });
});

onUnmounted(() => {
  if (documentProgressTimer !== undefined) window.clearInterval(documentProgressTimer);
  stopAgentStream();
});
</script>

<template>
  <main v-if="!isAuthed" class="auth-page">
    <section class="auth-story">
      <div class="auth-grid"></div>
      <header class="auth-brand"><span class="brand-mark"><i></i><i></i><i></i></span><span>面试提优学习</span></header>
      <div class="story-copy">
        <p class="eyebrow light">RAG · INTERVIEW · AGENT RUNTIME</p>
        <h1>把资料、项目和追问，压成一套可练习的面试能力。</h1>
        <p>围绕岗位 JD、简历项目、知识库材料与历史问答，生成有引用的回答、追问清单和提优建议。</p>
        <div class="story-features">
          <div><Database :size="20" /><span><strong>资料入库</strong><small>简历、项目文档、岗位要求统一索引</small></span></div>
          <div><SearchCheck :size="20" /><span><strong>有据回答</strong><small>每次训练都保留引用和来源</small></span></div>
          <div><Network :size="20" /><span><strong>Agent 复盘</strong><small>将问答结果转成薄弱点和训练计划</small></span></div>
          <div><Brain :size="20" /><span><strong>持续记忆</strong><small>结合会话上下文迭代表达质量</small></span></div>
        </div>
      </div>
      <footer>FASTAPI · QDRANT · CELERY · VUE</footer>
    </section>

    <section class="auth-panel">
      <form class="auth-form" @submit.prevent="submitAuth">
        <p class="eyebrow">SECURE WORKSPACE</p>
        <h2>{{ auth.mode === "login" ? "欢迎回来" : "创建训练账号" }}</h2>
        <p class="form-intro">登录后继续你的知识库、面试问答和 Agent 提优记录。</p>
        <label><span>用户名</span><input v-model="auth.username" autocomplete="username" placeholder="请输入用户名" /></label>
        <label><span>密码</span><input v-model="auth.password" type="password" autocomplete="current-password" placeholder="请输入密码" /></label>
        <p v-if="auth.error" class="form-error">{{ auth.error }}</p>
        <p v-if="auth.notice" class="form-success">{{ auth.notice }}</p>
        <button class="primary-button auth-submit" :disabled="auth.loading">
          <span>{{ auth.loading ? "处理中" : auth.mode === "login" ? "进入工作台" : "创建账号" }}</span>
          <ArrowRight v-if="!auth.loading" :size="18" />
          <LoaderCircle v-else :size="18" class="spin" />
        </button>
        <p class="auth-switch">
          {{ auth.mode === "login" ? "还没有账号？" : "已经有账号？" }}
          <button type="button" @click="auth.mode = auth.mode === 'login' ? 'register' : 'login'">
            {{ auth.mode === "login" ? "立即注册" : "返回登录" }}
          </button>
        </p>
        <div class="security-hint"><ShieldCheck :size="15" />资料和训练记录按账号隔离</div>
      </form>
    </section>
  </main>

  <div v-else class="app-frame">
    <aside class="sidebar">
      <div class="sidebar-head">
        <div class="brand"><span class="brand-mark"><i></i><i></i><i></i></span><span>面试提优学习</span></div>
      </div>
      <p class="eyebrow sidebar-eyebrow">TRAINING FLOW</p>
      <nav class="primary-nav">
        <a href="#workspace"><PanelLeft :size="19" /><span><strong>训练工作台</strong><small>RAG 问答与追问</small></span></a>
        <a href="#sources"><Layers3 :size="19" /><span><strong>资料管理</strong><small>知识库与文档</small></span></a>
        <a href="#agent"><Network :size="19" /><span><strong>Agent 提优</strong><small>事件流与复盘</small></span></a>
      </nav>
      <div class="sidebar-note">
        <span class="status-dot" :class="{ idle: !activeTask || activeTask.status !== 'running' }"></span>
        <div>
          <strong>{{ activeTask ? activeTask.status : "Ready" }}</strong>
          <small>{{ selectedKb?.name || "等待选择知识库" }}</small>
        </div>
      </div>
      <div class="account-card">
        <div class="avatar">{{ auth.username.slice(0, 1).toUpperCase() || "U" }}</div>
        <div class="account-copy"><strong>{{ auth.username }}</strong><small>面试训练空间</small></div>
        <button class="icon-button" title="退出" @click="logout"><LogOut :size="17" /></button>
      </div>
    </aside>

    <main class="app-main">
      <div class="coach-layout">
        <aside id="sources" class="control-rail">
          <div class="control-scroll">
            <header class="rail-intro">
              <p class="eyebrow">SOURCE CONTROL</p>
              <h1>资料与目标</h1>
              <p>先把简历、JD、项目复盘和笔记放进知识库，再围绕它们进行面试训练。</p>
            </header>

            <section class="source-card">
              <div class="rail-heading"><h2>知识库</h2><span>{{ state.kbs.length }}</span></div>
              <select v-model.number="state.selectedKbId" class="select-field" @change="refreshDocs">
                <option :value="0">个人文档</option>
                <option v-for="kb in state.kbs" :key="kb.id" :value="kb.id">{{ kb.name }}</option>
              </select>
              <input v-model="state.newKbName" class="text-field" placeholder="新知识库名称" />
              <textarea v-model="state.newKbDescription" class="small-textarea" rows="3" placeholder="知识库说明"></textarea>
              <button class="secondary-button full" :disabled="state.busy" @click="addKb"><Plus :size="15" />创建资料库</button>
            </section>

            <section class="source-card">
              <div class="rail-heading"><h2>文档</h2><span>{{ state.docs.length }}</span></div>
              <label class="upload-box">
                <UploadCloud :size="20" />
                <span>{{ state.uploading ? "上传中" : "上传简历 / JD / 笔记" }}</span>
                <input type="file" @change="onUpload" />
              </label>
              <select v-model.number="state.selectedDocId" class="select-field">
                <option :value="0">全部文档</option>
                <option
                  v-for="doc in state.docs"
                  :key="doc.id"
                  :value="doc.id"
                  :disabled="!isIndexedStatus(effectiveDocStatus(doc))"
                >{{ doc.file_name }} · {{ progressLabel(progressFor(doc)) }}</option>
              </select>
              <div class="doc-list">
                <article v-for="doc in state.docs.slice(0, 6)" :key="doc.id">
                  <FileUp :size="15" />
                  <span>
                    <strong>{{ doc.file_name }}</strong>
                    <small>{{ progressLabel(progressFor(doc)) }}</small>
                    <b class="doc-progress" :class="{ failed: effectiveDocStatus(doc) === 'failed' }">
                      <i :style="{ width: `${progressFor(doc).percent}%` }"></i>
                    </b>
                    <em v-if="progressFor(doc).total_chunks">{{ progressFor(doc).total_chunks }} chunks</em>
                    <button
                      v-if="effectiveDocStatus(doc) === 'failed'"
                      type="button"
                      class="mini-action"
                      :disabled="state.docBusyId === doc.id || !doc.source_retained"
                      @click="retryDocumentIndex(doc)"
                    >重建索引</button>
                  </span>
                </article>
              </div>
            </section>

            <section class="source-card">
              <div class="rail-heading"><h2>训练模式</h2><span>{{ modes.length }}</span></div>
              <button
                v-for="mode in modes"
                :key="mode.key"
                class="mode-card"
                :class="{ active: state.selectedMode.key === mode.key }"
                @click="selectMode(mode)"
              >
                <strong>{{ mode.label }}</strong>
                <small>{{ mode.helper }}</small>
              </button>
            </section>
          </div>
        </aside>

        <section id="workspace" class="learning-workspace">
          <header class="workspace-header">
            <div class="workspace-title">
              <p class="eyebrow">INTERVIEW IMPROVEMENT RAG</p>
              <h2>{{ state.selectedMode.label }} · {{ selectedKb?.name || "个人资料" }}</h2>
            </div>
            <div class="workspace-actions">
              <button class="secondary-button task-toggle" :disabled="state.busy" @click="bootstrap"><RefreshCw :size="15" />刷新</button>
              <button class="primary-button task-toggle" :disabled="state.taskBusy || state.busy" @click="startAgentReview"><Sparkles :size="15" />Agent 复盘</button>
            </div>
          </header>

          <div class="status-ribbon">
            <span><small>知识库</small><strong>{{ selectedKb?.name || "个人文档" }}</strong></span>
            <span><small>可用文档</small><strong>{{ readyDocs }}/{{ state.docs.length }}</strong></span>
            <span><small>引用数量</small><strong>{{ state.answer?.citations.length ?? 0 }}</strong></span>
            <span><small>任务状态</small><strong>{{ activeTask?.status || "idle" }}</strong></span>
          </div>
          <p v-if="state.error" class="workspace-error inline-error">{{ state.error }}</p>

          <div class="workspace-scroll">
            <section class="query-panel">
              <div class="panel-heading">
                <div><p class="eyebrow">PROMPT</p><h3>面试训练输入</h3></div>
                <span class="status-badge active">{{ state.selectedMode.label }}</span>
              </div>
              <textarea v-model="state.query" rows="7" placeholder="输入你的面试主题、岗位要求、项目描述或薄弱点。"></textarea>
              <div class="control-grid">
                <label><span>Top K</span><input v-model.number="state.topK" type="number" min="1" max="20" /></label>
                <label><span>BM25 权重</span><input v-model.number="state.bm25Weight" type="number" min="0" max="1" step="0.1" /></label>
                <label class="toggle"><input v-model="state.useMemory" type="checkbox" />启用记忆</label>
                <label class="toggle"><input v-model="state.rewriteQuery" type="checkbox" />改写问题</label>
              </div>
              <div class="panel-actions">
                <button class="primary-button" :disabled="state.busy || !selectedDocCanSearch" @click="askRag">
                  <LoaderCircle v-if="state.busy" :size="16" class="spin" /><MessageSquareText v-else :size="16" />
                  {{ state.busy ? "生成中" : "生成面试回答" }}
                </button>
              </div>
              <p v-if="!selectedDocCanSearch && selectedDocProgress" class="query-hint">当前文档状态为 {{ progressLabel(selectedDocProgress) }}，索引完成后才能按该文档检索。</p>
            </section>

            <section v-if="!state.answer" class="workspace-empty">
              <div class="empty-constellation"><Target :size="28" /><span></span><i></i><i></i></div>
              <p class="eyebrow">START A DRILL</p>
              <h2>选择训练模式，围绕你的资料开始追问。</h2>
              <p>建议先上传岗位 JD、简历项目描述和知识笔记。回答会保留引用，Agent 可以进一步生成提优复盘。</p>
            </section>

            <section v-else class="answer-layout">
              <article class="answer-card">
                <div class="panel-heading">
                  <div><p class="eyebrow">ANSWER</p><h3>参考回答</h3></div>
                  <span class="status-badge completed">{{ state.answer.retrieved_count }} chunks</span>
                </div>
                <p v-if="state.answer.rewritten_query" class="rewritten">改写问题：{{ state.answer.rewritten_query }}</p>
                <pre :class="{ typing: state.busy }">{{ state.answer.answer }}</pre>
              </article>

              <aside class="citation-panel">
                <div class="panel-heading">
                  <div><p class="eyebrow">EVIDENCE</p><h3>引用来源</h3></div>
                </div>
                <article v-for="citation in state.answer.citations" :key="`${citation.source_id}-${citation.chunk_id}`" class="citation-card">
                  <strong>{{ citation.filename }}</strong>
                  <small>{{ citation.location || `chunk ${citation.chunk_id}` }} · score {{ citation.score.toFixed(3) }}</small>
                  <p>{{ citation.quote }}</p>
                </article>
              </aside>
            </section>
          </div>
        </section>

        <aside id="agent" class="activity-panel">
          <div class="activity-head">
            <div><p class="eyebrow">AGENT RUNTIME</p><h2>提优事件</h2></div>
            <span class="live-pill" :class="{ live: state.agentStreaming || activeTask?.status === 'running' }"><i></i>{{ state.agentStreaming ? "streaming" : activeTask?.status || "idle" }}</span>
          </div>
          <div class="runtime-summary">
            <div><Gauge :size="15" /><span><small>Tasks</small><strong>{{ state.tasks.length }}</strong></span></div>
            <div><Bot :size="15" /><span><small>Events</small><strong>{{ state.events.length }}</strong></span></div>
          </div>
          <div class="activity-actions">
            <button class="secondary-button" @click="refreshTasks"><RefreshCw :size="14" />刷新事件</button>
            <button v-if="activeTask && ['pending', 'running'].includes(activeTask.status)" class="ghost-button" @click="cancelActiveTask"><XCircle :size="14" />取消</button>
          </div>
          <div v-if="!state.events.length" class="activity-empty">
            <div><Network :size="22" /></div>
            <h3>等待 Agent 复盘</h3>
            <p>点击中间顶部的 Agent 复盘，会把当前训练结果送入 Agent Runtime。</p>
          </div>
          <div v-else class="activity-list">
            <article v-for="event in state.events" :key="event.event_index" class="activity-card">
              <span class="activity-line"><i></i></span>
              <div class="activity-card-body">
                <div class="activity-meta">
                  <span class="activity-kind">{{ event.event_type }}</span>
                  <time>#{{ event.event_index }}</time>
                </div>
                <h3>{{ event.agent_name || event.tool_name || "runtime" }}</h3>
                <p>{{ event.message }}</p>
              </div>
            </article>
          </div>
          <article v-if="activeTask?.final_answer" class="final-answer">
            <CheckCircle2 :size="16" />
            <pre>{{ activeTask.final_answer }}</pre>
          </article>
        </aside>
      </div>
    </main>
  </div>
</template>
