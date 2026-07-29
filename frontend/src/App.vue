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
  getLearningDashboard,
  getLearningProfile,
  listAgentEvents,
  listAgentTasks,
  listDocuments,
  listKnowledgeBases,
  listLearningPractices,
  listLearningReviews,
  listLearningWeaknesses,
  login,
  reindexDocument,
  register,
  streamAgentEvents,
  AUTH_INVALID_EVENT,
  tokenStore,
  updateLearningProfile,
  uploadDocument,
  type AnswerQuestionPayload,
  type AgentEvent,
  type AgentTask,
  type AnswerResponse,
  type DocumentProgress,
  type DocumentItem,
  type KnowledgeBase,
  type LearningDashboard,
  type LearningPractice,
  type LearningProfile,
  type LearningReviewItem,
  type LearningWeakness
} from "@/services/api";

type Mode = {
  key: string;
  label: string;
  prompt: string;
  helper: string;
  taskType: string;
  agentGoal: string;
  ragFirst: boolean;
  outputs: string[];
};

type LearningRoute = "question_answering" | "learning_system_design" | "clarify";
type NavPanel = "workspace" | "sources" | "agent";

type RouteDecision = {
  route: LearningRoute;
  label: string;
  reason: string;
  confidence: number;
  ragQuery?: string;
  planningGoal?: string;
};

type ProfileOption = {
  key: string;
  label: string;
  helper: string;
};

const modes: Mode[] = [
  {
    key: "jd_interview_prep",
    label: "JD 面试准备",
    helper: "拆 JD、匹配简历证据、生成追问优先级",
    taskType: "scenario_jd_interview_prep",
    agentGoal: "围绕目标 JD 建立面试准备闭环，输出岗位要求、简历证据、风险追问和训练优先级。",
    ragFirst: true,
    outputs: ["JD 拆解", "证据匹配", "追问清单"],
    prompt: "请针对下面这份 JD 做面试准备：先拆岗位能力要求，再结合我的简历/项目材料找证据和缺口，最后给出高频追问与回答框架。\n\nJD/目标岗位："
  },
  {
    key: "project_deep_dive",
    label: "项目深挖追问",
    helper: "围绕项目事实、取舍、边界做追问树",
    taskType: "scenario_project_deep_dive",
    agentGoal: "针对一个项目生成面试官深挖路径，覆盖背景、方案、难点、取舍、指标和风险边界。",
    ragFirst: true,
    outputs: ["事实线", "追问树", "回答稿"],
    prompt: "请针对下面这个项目做深挖追问：提取项目事实线，生成从基础到高压的技术追问树，并给出 3 分钟项目回答稿。\n\n项目描述/项目名："
  },
  {
    key: "knowledge_ladder_5x",
    label: "5 轮知识追问",
    helper: "从概念到项目迁移，做递进式追问",
    taskType: "scenario_knowledge_ladder_5x",
    agentGoal: "围绕单个知识点设计 5 轮递进追问，逐轮提升深度并连接项目证据。",
    ragFirst: true,
    outputs: ["5 轮问题", "考察点", "误区"],
    prompt: "请围绕下面这个知识点做 5 轮递进追问：每轮说明考察点、优秀回答骨架、常见误区，并在最后连接到项目场景。\n\n知识点："
  },
  {
    key: "resume_star_rewrite",
    label: "项目 STAR 改造",
    helper: "把简历项目改成可讲、可信、可量化表达",
    taskType: "scenario_resume_star_rewrite",
    agentGoal: "将简历项目改造成 STAR 表达，补齐情境、任务、行动、结果和量化证据。",
    ragFirst: true,
    outputs: ["STAR 拆解", "简历 bullet", "口述稿"],
    prompt: "请把下面的简历项目改造成 STAR 表达：指出证据缺口，改写成简历 bullet，并给出面试口述稿。\n\n原项目描述："
  },
  {
    key: "technical_mock_30m",
    label: "30 分钟技术面",
    helper: "生成完整面试议程、问题、评分和复盘",
    taskType: "scenario_technical_mock_30m",
    agentGoal: "模拟一场 30 分钟技术面，按开场、项目、知识、系统设计、收尾组织问题与评分。",
    ragFirst: true,
    outputs: ["面试议程", "分阶段问题", "Rubric"],
    prompt: "请模拟一场 30 分钟技术面：按时间段生成问题、追问策略、评分 Rubric 和复盘建议。\n\n目标岗位/希望考察方向："
  },
  {
    key: "weekly_training_plan",
    label: "下周训练计划",
    helper: "结合历史回答和薄弱点安排 7 天训练",
    taskType: "scenario_weekly_training_plan",
    agentGoal: "根据历史回答、weakness 和当前目标生成下一周训练计划，包含每日练习与验收指标。",
    ragFirst: false,
    outputs: ["7 天日历", "练习题", "复盘指标"],
    prompt: "请根据我的历史回答、已有 weakness 和当前目标，生成下一周训练计划。请说明每日训练主题、练习题、复盘指标和复习间隔。\n\n目标/约束："
  }
];

const interviewCapabilityTitles = ["知识理解", "项目表达", "追问应对", "系统设计", "沟通表达", "岗位匹配"];
const interviewScopeSignals = ["简历", "resume", "cv", "jd", "岗位", "职位", "面试", "项目经历", "项目资料"];
const interviewIntentSignals = ["提升", "提优", "诊断", "薄弱", "weakness", "不足", "复盘", "总结", "追问", "模拟", "准备", "匹配", "能力", "评价", "评估", "练习"];

const candidateStageOptions: ProfileOption[] = [
  { key: "fresh_graduate", label: "应届生", helper: "校招、实习转正、秋招/春招准备" },
  { key: "experienced", label: "社招", helper: "跳槽、晋升、专项岗位面试" },
  { key: "career_switcher", label: "转岗/转行", helper: "补齐基础并重塑项目表达" }
];

const targetRoleOptions: ProfileOption[] = [
  { key: "backend", label: "后端工程师", helper: "Java / Go / Python / 架构设计" },
  { key: "frontend", label: "前端工程师", helper: "框架、工程化、性能与体验" },
  { key: "algorithm", label: "算法工程师", helper: "数据结构、算法题、复杂度分析" },
  { key: "ai_engineer", label: "AI 工程师", helper: "RAG、Agent、模型应用与评估" },
  { key: "product_manager", label: "产品经理", helper: "需求分析、方案表达、业务判断" }
];

const prepGoalOptions: ProfileOption[] = [
  { key: "knowledge_drill", label: "刷八股", helper: "高频知识点、原理追问、体系化复习" },
  { key: "project_story", label: "项目表达", helper: "STAR 复盘、技术亮点、难点取舍" },
  { key: "jd_mock", label: "针对 JD 模拟面试", helper: "岗位匹配、简历追问、面试官视角演练" }
];

const timelineOptions: ProfileOption[] = [
  { key: "short_sprint", label: "短期冲刺", helper: "1-4 周集中准备，优先面试通过率" },
  { key: "long_term_growth", label: "长期学习提升", helper: "持续补短板，沉淀能力地图和练习闭环" }
];

const auth = reactive({
  username: tokenStore.username(),
  password: "",
  mode: "login" as "login" | "register",
  loading: false,
  error: "",
  notice: ""
});

const profileForm = reactive({
  candidateStage: "",
  targetRoles: [] as string[],
  prepGoals: [] as string[],
  timeline: "",
  targetJd: "",
  loading: false,
  error: ""
});

const state = reactive({
  token: tokenStore.access(),
  kbs: [] as KnowledgeBase[],
  docs: [] as DocumentItem[],
  docProgresses: {} as Record<number, DocumentProgress>,
  tasks: [] as AgentTask[],
  events: [] as AgentEvent[],
  learningProfile: null as LearningProfile | null,
  learningDashboard: null as LearningDashboard | null,
  weaknesses: [] as LearningWeakness[],
  practices: [] as LearningPractice[],
  reviews: [] as LearningReviewItem[],
  selectedKbId: 0,
  selectedDocId: 0,
  activePanel: "workspace" as NavPanel,
  newKbName: "面试提优资料库",
  newKbDescription: "简历、岗位 JD、项目复盘、八股笔记与模拟面试记录。",
  newKbVisibility: "private" as "private" | "shared",
  query: modes[0].prompt,
  topK: 6,
  bm25Weight: 0.4,
  rewriteQuery: true,
  answer: null as AnswerResponse | null,
  routeDecision: null as RouteDecision | null,
  routeBusy: false,
  conversationId: 0,
  selectedMode: modes[0],
  busy: false,
  uploading: false,
  docBusyId: 0,
  taskBusy: false,
  agentStreaming: false,
  learningBusy: false,
  bootstrapping: Boolean(tokenStore.access()),
  error: ""
});

const isAuthed = computed(() => Boolean(state.token));
const hasCompletedProfile = computed(() => Boolean(state.learningProfile?.preferences?.profile_completed));
const needsOnboarding = computed(() => isAuthed.value && !state.bootstrapping && !hasCompletedProfile.value);
const selectedKb = computed(() => state.kbs.find((kb) => kb.id === state.selectedKbId));
const selectedKbCanUpload = computed(() => !selectedKb.value || selectedKb.value.can_upload);
const selectedKbScopeLabel = computed(() => {
  if (!selectedKb.value) return "私人文档";
  return selectedKb.value.visibility === "shared" ? "共享知识库" : "私人知识库";
});
const selectedDoc = computed(() => state.docs.find((doc) => doc.id === state.selectedDocId));
const activeTask = computed(() => state.tasks[0]);
const readyDocs = computed(() => state.docs.filter((doc) => isIndexedStatus(effectiveDocStatus(doc))).length);
const failedDocs = computed(() => state.docs.filter((doc) => effectiveDocStatus(doc) === "failed").length);
const selectedDocStatus = computed(() => selectedDoc.value ? effectiveDocStatus(selectedDoc.value) : "");
const selectedDocCanSearch = computed(() => !selectedDoc.value || isIndexedStatus(selectedDocStatus.value));
const selectedDocProgress = computed(() => selectedDoc.value ? progressFor(selectedDoc.value) : null);
const routeLabel = computed(() => state.routeDecision?.label ?? "等待主 Agent 判断");
const routeTone = computed(() => {
  if (!state.routeDecision) return "idle";
  if (state.routeDecision.route === "question_answering") return "qa";
  if (state.routeDecision.route === "learning_system_design") return "agent";
  return "clarify";
});
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

function optionLabel(options: ProfileOption[], key: string) {
  return options.find((item) => item.key === key)?.label ?? key;
}

function selectPanel(panel: NavPanel) {
  state.activePanel = panel;
}

function toggleProfileValue(list: string[], key: string) {
  const index = list.indexOf(key);
  if (index >= 0) list.splice(index, 1);
  else list.push(key);
}

function buildProfileSummary() {
  const stage = optionLabel(candidateStageOptions, profileForm.candidateStage);
  const roles = profileForm.targetRoles.map((key) => optionLabel(targetRoleOptions, key)).join("、");
  const goals = profileForm.prepGoals.map((key) => optionLabel(prepGoalOptions, key)).join("、");
  const timeline = optionLabel(timelineOptions, profileForm.timeline);
  return `${stage}｜目标：${roles}｜当前重点：${goals}｜节奏：${timeline}`;
}

async function submitUserProfile() {
  profileForm.error = "";
  if (!profileForm.candidateStage || !profileForm.targetRoles.length || !profileForm.prepGoals.length || !profileForm.timeline) {
    profileForm.error = "请至少选择身份阶段、目标方向、准备重点和学习节奏。";
    return;
  }
  profileForm.loading = true;
  try {
    const saved = await updateLearningProfile({
      target_role: profileForm.targetRoles.map((key) => optionLabel(targetRoleOptions, key)).join("、"),
      goal: buildProfileSummary(),
      current_level: profileForm.candidateStage,
      weekly_minutes: profileForm.timeline === "short_sprint" ? 600 : 300,
      preferences: {
        ...(state.learningProfile?.preferences ?? {}),
        profile_completed: true,
        candidate_stage: profileForm.candidateStage,
        target_roles: [...profileForm.targetRoles],
        preparation_goals: [...profileForm.prepGoals],
        learning_timeline: profileForm.timeline,
        target_jd: profileForm.targetJd.trim(),
        onboarding_version: 1
      }
    });
    state.learningProfile = saved;
    await bootstrap();
  } catch (error) {
    profileForm.error = messageFromError(error, "画像保存失败。");
  } finally {
    profileForm.loading = false;
  }
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
    warnings: [],
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
  state.learningProfile = null;
  state.learningDashboard = null;
  state.weaknesses = [];
  state.practices = [];
  state.reviews = [];
  state.answer = null;
  state.bootstrapping = false;
}

function handleAuthInvalid(event: Event) {
  const message = event instanceof CustomEvent && typeof event.detail === "string" ? event.detail : "Invalid token";
  logout();
  auth.error = message === "AUTH_TOKEN_EXPIRED" ? "登录已过期，请重新登录。" : "登录状态无效，请重新登录。";
}

async function bootstrap() {
  state.error = "";
  state.bootstrapping = true;
  try {
    await Promise.all([refreshKbs(), refreshTasks(), refreshLearning()]);
  } finally {
    state.bootstrapping = false;
  }
}

async function refreshLearning() {
  if (!state.token) return;
  state.learningBusy = true;
  try {
    const [profile, dashboard, weaknesses, practices, reviews] = await Promise.all([
      getLearningProfile(),
      getLearningDashboard(),
      listLearningWeaknesses(),
      listLearningPractices(),
      listLearningReviews()
    ]);
    state.learningProfile = profile;
    state.learningDashboard = dashboard;
    state.weaknesses = weaknesses;
    state.practices = practices;
    state.reviews = reviews;
  } catch (error) {
    state.error = messageFromError(error, "学习数据刷新失败。");
  } finally {
    state.learningBusy = false;
  }
}

async function refreshKbs() {
  state.kbs = await listKnowledgeBases();
  if (state.selectedKbId && !state.kbs.some((kb) => kb.id === state.selectedKbId)) state.selectedKbId = 0;
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
    const kb = await createKnowledgeBase(
      state.newKbName.trim(),
      state.newKbDescription.trim(),
      state.newKbVisibility
    );
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
  if (!selectedKbCanUpload.value) {
    state.error = "当前账号没有向共享知识库上传的权限，请切回个人文档或使用管理员账号。";
    input.value = "";
    return;
  }
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

function isInterviewCapabilityInput(input: string) {
  const normalized = input.trim().toLowerCase();
  if (!normalized) return false;
  const hasCapabilityTitle = interviewCapabilityTitles.some((item) => normalized.includes(item.toLowerCase()));
  const hasScope = interviewScopeSignals.some((item) => normalized.includes(item.toLowerCase()));
  const hasIntent = interviewIntentSignals.some((item) => normalized.includes(item.toLowerCase()));
  return hasCapabilityTitle || (hasScope && hasIntent);
}

function routeLearningRequest(input: string, scenario: Mode = state.selectedMode): RouteDecision {
  const text = input.trim();
  const normalized = text.toLowerCase();
  if (scenario.taskType.startsWith("scenario_")) {
    return {
      route: "learning_system_design",
      label: scenario.label,
      reason: `一级业务场景「${scenario.label}」会直接进入 Agent 场景流程：${scenario.agentGoal}`,
      confidence: 0.94,
      planningGoal: text
    };
  }
  const planningSignals = [
    "提升", "规划", "计划", "路线", "学习路径", "诊断", "薄弱", "练习", "训练", "模拟",
    "复盘", "批改", "评价我的", "帮我准备", "怎么学", "安排", "制定", "简历", "岗位匹配",
    "improve", "plan", "roadmap", "practice", "diagnose", "mock", "review", "coach"
  ];
  const questionSignals = [
    "是什么", "为什么", "区别", "原理", "解释", "怎么理解", "对比", "举例",
    "what", "why", "explain", "difference", "compare", "how does"
  ];
  const hasPlanningIntent = planningSignals.some((item) => normalized.includes(item.toLowerCase()));
  const hasQuestionIntent = questionSignals.some((item) => normalized.includes(item.toLowerCase())) || /[?？]$/.test(text);
  if (!text) {
    return {
      route: "clarify",
      label: "需要补充输入",
      reason: "请输入一个具体问题，或描述你想提升的学习目标。",
      confidence: 1
    };
  }
  if (hasPlanningIntent) {
    if (isInterviewCapabilityInput(text)) {
      return {
        route: "learning_system_design",
        label: "RAG 首答 + Agent 提优",
        reason: "请求涉及简历/JD面试诊断，会先用 RAG 按能力模型总结并标记 weakness，再进入 Agent 后续流程。",
        confidence: 0.88,
        planningGoal: text
      };
    }
    return {
      route: "learning_system_design",
      label: "学习提升任务",
      reason: "请求涉及计划、诊断、练习或长期提升，需要主 Agent 分发子 Agent。",
      confidence: 0.82,
      planningGoal: text
    };
  }
  if (hasQuestionIntent || text.length < 80) {
    return {
      route: "question_answering",
      label: "知识问答",
      reason: "请求更像一个具体知识问题，优先由主 Agent 调用 RAG 检索回答。",
      confidence: hasQuestionIntent ? 0.86 : 0.68,
      ragQuery: text
    };
  }
  return {
    route: "learning_system_design",
    label: "学习提升任务",
    reason: "输入较开放，按学习系统设计任务处理，交由子 Agent 协作。",
    confidence: 0.64,
    planningGoal: text
  };
}

async function submitLearningRequest() {
  if (!state.query.trim()) return;
  state.routeBusy = true;
  state.error = "";
  state.answer = null;
  state.routeDecision = routeLearningRequest(state.query, state.selectedMode);
  try {
    if (state.routeDecision.route === "clarify") {
      state.error = state.routeDecision.reason;
      return;
    }
    if (state.routeDecision.route === "question_answering") {
      await askRag();
      return;
    }
    if (state.selectedMode.ragFirst || isInterviewCapabilityInput(state.query)) {
      await askRag();
      if (state.error || !state.answer) return;
      await refreshLearning();
    }
    await startLearningAgentTask();
  } finally {
    state.routeBusy = false;
  }
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
    await refreshLearning();
  } catch (error) {
    await typewriter.stop();
    state.error = messageFromError(error, "问答失败。");
  } finally {
    state.busy = false;
  }
}

async function startAgentReview() {
  const input = state.answer
    ? `业务场景：${state.selectedMode.label}\n场景目标：${state.selectedMode.agentGoal}\n\n请基于当前 RAG 问答结果，按知识理解、项目表达、追问应对、系统设计、沟通表达、岗位匹配生成面试提优反馈：指出回答亮点、weakness、追问清单和下一轮训练建议。\n\n问题：${state.answer.query}\n\n回答：${state.answer.answer}`
    : `业务场景：${state.selectedMode.label}\n场景目标：${state.selectedMode.agentGoal}\n\n请基于知识库「${selectedKb.value?.name ?? "当前资料"}」生成场景化面试提优方案，包含 weakness、模拟问题和复习优先级。`;
  state.taskBusy = true;
  state.error = "";
  try {
    const task = await createAgentTask({
      user_input: input,
      task_type: state.selectedMode.taskType,
      scenario_key: state.selectedMode.key,
      scenario_inputs: {
        selected_outputs: state.selectedMode.outputs,
        raw_query: state.query.trim()
      },
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

async function startLearningAgentTask() {
  const input = [
    `一级业务场景：${state.selectedMode.label}`,
    `场景目标：${state.selectedMode.agentGoal}`,
    `预期交付：${state.selectedMode.outputs.join("、")}`,
    "",
    "主 Agent 已判断这是场景化学习提升/系统规划设计任务。",
    "如果上方已有 RAG 首答，请把它作为已完成的第一次回答与 weakness 证据继续推进；否则先检索知识库。",
    "请先读取或遵循该业务场景蓝图，再围绕知识理解、项目表达、追问应对、系统设计、沟通表达、岗位匹配制定后续完整 Agent 流程。",
    "",
    `用户目标：${state.query.trim()}`,
    state.answer
      ? `\n首轮 RAG 问题：${state.answer.query}\n首轮 RAG 回答：${state.answer.answer.slice(0, 1800)}`
      : ""
  ].join("\n");
  state.taskBusy = true;
  state.error = "";
  try {
    const task = await createAgentTask({
      user_input: input,
      task_type: state.selectedMode.taskType,
      scenario_key: state.selectedMode.key,
      scenario_inputs: {
        selected_outputs: state.selectedMode.outputs,
        rag_first: state.selectedMode.ragFirst,
        raw_query: state.query.trim()
      },
      kb_id: state.selectedKbId || undefined,
      document_id: state.selectedDocId || undefined,
      conversation_id: state.conversationId || undefined,
      max_steps: 10,
      max_tool_calls: 16
    });
    await refreshTasks();
    await refreshEvents(task.task_id);
    await refreshLearning();
    connectAgentStream(task.task_id);
  } catch (error) {
    state.error = messageFromError(error, "Agent task failed to start.");
  } finally {
    state.taskBusy = false;
  }
}

async function refreshTasks() {
  state.tasks = (await listAgentTasks()).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  if (state.tasks[0]) {
    await refreshEvents(state.tasks[0].task_id);
    if (isActiveAgentStatus(state.tasks[0].status)) connectAgentStream(state.tasks[0].task_id);
    if (state.tasks[0].status === "completed") void refreshLearning();
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
  window.addEventListener(AUTH_INVALID_EVENT, handleAuthInvalid);
  if (state.token) void bootstrap().catch((error) => {
    state.error = messageFromError(error, "初始化失败。");
  });
});

onUnmounted(() => {
  window.removeEventListener(AUTH_INVALID_EVENT, handleAuthInvalid);
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

  <main v-else-if="needsOnboarding" class="profile-page">
    <section class="profile-shell">
      <header class="profile-hero">
        <div>
          <p class="eyebrow">FIRST RUN PROFILE</p>
          <h1>先把你的面试准备画像定下来。</h1>
          <p>系统会根据身份阶段、目标方向、准备重点和学习节奏，调整默认训练任务、Agent 规划和后续学习指标。</p>
        </div>
        <button class="ghost-button" @click="logout"><LogOut :size="15" />退出</button>
      </header>

      <form class="profile-form" @submit.prevent="submitUserProfile">
        <section class="profile-section">
          <div class="profile-section-title">
            <BookOpenCheck :size="20" />
            <div><h2>你现在处在哪个阶段？</h2><p>用于判断训练深度、项目追问密度和基础知识占比。</p></div>
          </div>
          <div class="profile-options single">
            <button
              v-for="item in candidateStageOptions"
              :key="item.key"
              type="button"
              class="profile-option"
              :class="{ active: profileForm.candidateStage === item.key }"
              @click="profileForm.candidateStage = item.key"
            >
              <strong>{{ item.label }}</strong>
              <small>{{ item.helper }}</small>
            </button>
          </div>
        </section>

        <section class="profile-section">
          <div class="profile-section-title">
            <Target :size="20" />
            <div><h2>目标岗位方向</h2><p>可以多选，适合跨方向准备或 JD 尚未完全确定的用户。</p></div>
          </div>
          <div class="profile-options">
            <button
              v-for="item in targetRoleOptions"
              :key="item.key"
              type="button"
              class="profile-option"
              :class="{ active: profileForm.targetRoles.includes(item.key) }"
              @click="toggleProfileValue(profileForm.targetRoles, item.key)"
            >
              <strong>{{ item.label }}</strong>
              <small>{{ item.helper }}</small>
            </button>
          </div>
        </section>

        <section class="profile-section">
          <div class="profile-section-title">
            <SearchCheck :size="20" />
            <div><h2>当前准备重点</h2><p>这些会成为后续 RAG 提问模板、模拟面试和复盘任务的优先级。</p></div>
          </div>
          <div class="profile-options">
            <button
              v-for="item in prepGoalOptions"
              :key="item.key"
              type="button"
              class="profile-option"
              :class="{ active: profileForm.prepGoals.includes(item.key) }"
              @click="toggleProfileValue(profileForm.prepGoals, item.key)"
            >
              <strong>{{ item.label }}</strong>
              <small>{{ item.helper }}</small>
            </button>
          </div>
        </section>

        <section class="profile-section">
          <div class="profile-section-title">
            <Gauge :size="20" />
            <div><h2>学习节奏</h2><p>短期冲刺会提高默认训练强度，长期提升会更强调复习闭环。</p></div>
          </div>
          <div class="profile-options single">
            <button
              v-for="item in timelineOptions"
              :key="item.key"
              type="button"
              class="profile-option"
              :class="{ active: profileForm.timeline === item.key }"
              @click="profileForm.timeline = item.key"
            >
              <strong>{{ item.label }}</strong>
              <small>{{ item.helper }}</small>
            </button>
          </div>
        </section>

        <section class="profile-section">
          <div class="profile-section-title">
            <FileUp :size="20" />
            <div><h2>目标 JD 或补充说明</h2><p>可选。后续做 JD 模拟面试时，Agent 会优先参考这里的描述。</p></div>
          </div>
          <textarea v-model="profileForm.targetJd" rows="4" placeholder="例如：3 年后端，Java/Spring Cloud，要求 Redis、MySQL 调优、分布式事务，有 RAG 项目加分。"></textarea>
        </section>

        <p v-if="profileForm.error" class="form-error">{{ profileForm.error }}</p>
        <div class="profile-actions">
          <span>这些内容会保存到你的学习画像中，之后可以继续更新。</span>
          <button class="primary-button" :disabled="profileForm.loading">
            <span>{{ profileForm.loading ? "保存中" : "保存画像并进入工作台" }}</span>
            <LoaderCircle v-if="profileForm.loading" :size="17" class="spin" />
            <ArrowRight v-else :size="17" />
          </button>
        </div>
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
        <a href="#workspace" :class="{ active: state.activePanel === 'workspace' }" @click.prevent="selectPanel('workspace')"><PanelLeft :size="19" /><span><strong>训练工作台</strong><small>RAG 问答与追问</small></span></a>
        <a href="#sources" :class="{ active: state.activePanel === 'sources' }" @click.prevent="selectPanel('sources')"><Layers3 :size="19" /><span><strong>资料管理</strong><small>知识库与文档</small></span></a>
        <a href="#agent" :class="{ active: state.activePanel === 'agent' }" @click.prevent="selectPanel('agent')"><Network :size="19" /><span><strong>Agent 提优</strong><small>事件流与复盘</small></span></a>
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
      <div class="coach-layout" :class="`active-${state.activePanel}`">
        <aside id="sources" class="control-rail" :class="{ active: state.activePanel === 'sources' }">
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
                <option v-for="kb in state.kbs" :key="kb.id" :value="kb.id">
                  {{ kb.visibility === "shared" ? "[共享]" : "[私人]" }} {{ kb.name }}
                </option>
              </select>
              <input v-model="state.newKbName" class="text-field" placeholder="新知识库名称" />
              <textarea v-model="state.newKbDescription" class="small-textarea" rows="3" placeholder="知识库说明"></textarea>
              <select v-model="state.newKbVisibility" class="select-field">
                <option value="private">私人知识库</option>
                <option value="shared">共享知识库（管理员）</option>
              </select>
              <button class="secondary-button full" :disabled="state.busy" @click="addKb"><Plus :size="15" />创建资料库</button>
            </section>

            <section class="source-card">
              <div class="rail-heading"><h2>文档</h2><span>{{ state.docs.length }}</span></div>
              <label class="upload-box" :class="{ disabled: !selectedKbCanUpload }">
                <UploadCloud :size="20" />
                <span>
                  {{ state.uploading ? "上传中" : selectedKbCanUpload ? `上传到${selectedKbScopeLabel}` : "仅管理员可上传共享库" }}
                </span>
                <input type="file" :disabled="!selectedKbCanUpload" @change="onUpload" />
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
              <div class="rail-heading"><h2>业务场景</h2><span>{{ modes.length }}</span></div>
              <button
                v-for="mode in modes"
                :key="mode.key"
                class="mode-card"
                :class="{ active: state.selectedMode.key === mode.key }"
                @click="selectMode(mode)"
              >
                <strong>{{ mode.label }}</strong>
                <small>{{ mode.helper }}</small>
                <b>{{ mode.outputs.join(" · ") }}</b>
              </button>
            </section>
          </div>
        </aside>

        <section id="workspace" class="learning-workspace" :class="{ active: state.activePanel === 'workspace' }">
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
          <section class="route-strip" :class="routeTone">
            <div>
              <p class="eyebrow">MAIN AGENT ROUTE</p>
              <h3>{{ routeLabel }}</h3>
              <p>{{ state.routeDecision?.reason || "选择业务场景后会进入对应 Agent 流程；具体知识问题仍可使用仅 RAG 问答。" }}</p>
            </div>
            <span v-if="state.routeDecision">{{ Math.round(state.routeDecision.confidence * 100) }}%</span>
          </section>
          <p v-if="state.error" class="workspace-error inline-error">{{ state.error }}</p>

          <div class="workspace-scroll">
            <section class="query-panel">
              <div class="panel-heading">
                <div><p class="eyebrow">PROMPT</p><h3>面试训练输入</h3></div>
                <span class="status-badge active">一级场景</span>
              </div>
              <textarea v-model="state.query" rows="7" placeholder="输入 JD、项目描述、知识点、历史回答或训练约束。"></textarea>
              <div class="control-grid">
                <label><span>Top K</span><input v-model.number="state.topK" type="number" min="1" max="20" /></label>
                <label><span>BM25 权重</span><input v-model.number="state.bm25Weight" type="number" min="0" max="1" step="0.1" /></label>
                <!-- RAG memory toggle removed; memory is now managed exclusively by the Agent runtime. -->
                <label class="toggle"><input v-model="state.rewriteQuery" type="checkbox" />改写问题</label>
              </div>
              <div class="panel-actions">
                <button class="primary-button" :disabled="state.routeBusy || state.busy || state.taskBusy || !selectedDocCanSearch" @click="submitLearningRequest">
                  <LoaderCircle v-if="state.routeBusy || state.busy || state.taskBusy" :size="16" class="spin" /><Sparkles v-else :size="16" />
                  {{ state.routeBusy || state.busy || state.taskBusy ? "执行中" : "开始学习任务" }}
                </button>
                <button class="secondary-button" :disabled="state.busy || !selectedDocCanSearch" @click="askRag">
                  <MessageSquareText :size="16" />
                  仅 RAG 问答
                </button>
              </div>
              <p v-if="!selectedDocCanSearch && selectedDocProgress" class="query-hint">当前文档状态为 {{ progressLabel(selectedDocProgress) }}，索引完成后才能按该文档检索。</p>
            </section>

            <section v-if="!state.answer" class="workspace-empty">
              <div class="empty-constellation"><Target :size="28" /><span></span><i></i><i></i></div>
              <p class="eyebrow">MAIN AGENT READY</p>
              <h2>选择一个业务场景，然后补充材料或目标。</h2>
              <p>场景会进入对应 Agent 流程；也可以用仅 RAG 问答快速查证某个知识点。</p>
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

        <aside id="agent" class="activity-panel" :class="{ active: state.activePanel === 'agent' }">
          <div class="activity-head">
            <div><p class="eyebrow">AGENT RUNTIME</p><h2>提优事件</h2></div>
            <span class="live-pill" :class="{ live: state.agentStreaming || activeTask?.status === 'running' }"><i></i>{{ state.agentStreaming ? "streaming" : activeTask?.status || "idle" }}</span>
          </div>
          <div class="runtime-summary">
            <div><Gauge :size="15" /><span><small>Tasks</small><strong>{{ state.tasks.length }}</strong></span></div>
            <div><Bot :size="15" /><span><small>Events</small><strong>{{ state.events.length }}</strong></span></div>
          </div>
          <section class="business-dashboard">
            <div class="panel-heading compact">
              <div><p class="eyebrow">BUSINESS VIEW</p><h3>学习闭环指标</h3></div>
              <button class="icon-button" title="刷新学习指标" :disabled="state.learningBusy" @click="refreshLearning"><RefreshCw :size="14" /></button>
            </div>
            <div class="metric-grid">
              <span><small>14日活跃</small><strong>{{ state.learningDashboard?.active_days_14d ?? 0 }}</strong></span>
              <span><small>完成任务</small><strong>{{ state.learningDashboard?.tasks_completed_14d ?? 0 }}</strong></span>
              <span><small>练习正确率</small><strong>{{ Math.round((state.learningDashboard?.practice_accuracy ?? 0) * 100) }}%</strong></span>
              <span><small>节省人工</small><strong>{{ state.learningDashboard?.agent_saved_minutes ?? 0 }}m</strong></span>
            </div>
            <div class="profile-strip">
              <strong>{{ state.learningProfile?.target_role || "学习目标待诊断" }}</strong>
              <small>准备度 {{ Math.round((state.learningProfile?.readiness_score ?? 0) * 100) }}% · 待复习 {{ state.learningDashboard?.due_reviews ?? 0 }}</small>
            </div>
            <div class="weakness-stack">
              <article v-for="item in state.weaknesses.slice(0, 3)" :key="item.id">
                <span>{{ item.topic }}</span>
                <b><i :style="{ width: `${Math.round(item.severity * 100)}%` }"></i></b>
              </article>
            </div>
          </section>
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
          <section class="practice-queue">
            <div class="panel-heading compact">
              <div><p class="eyebrow">PRACTICE</p><h3>下一轮练习</h3></div>
            </div>
            <article v-for="item in state.practices.slice(0, 3)" :key="item.id">
              <strong>{{ item.topic }}</strong>
              <p>{{ item.question }}</p>
            </article>
            <article v-if="!state.practices.length" class="empty-mini">
              <strong>暂无练习</strong>
              <p>运行一次学习提升任务后会自动生成练习和复习项。</p>
            </article>
          </section>
        </aside>
      </div>
    </main>
  </div>
</template>
