export type TokenResponse = {
  access_token: string;
  refresh_token: string;
  token_type: string;
};

export type KnowledgeBase = {
  id: number;
  name: string;
  description?: string | null;
  role?: string | null;
  member_count: number;
  created_at: string;
  updated_at: string;
};

export type DocumentItem = {
  id: number;
  file_name: string;
  content_type?: string;
  file_size?: number;
  status: string;
  source_retained?: boolean;
  created_at: string;
  kb_id?: number | null;
};

export type DocumentProgress = {
  document_id: number;
  status: string;
  document_status?: string;
  percent: number;
  task_id?: string | null;
  attempt?: number | null;
  total_chunks?: number | null;
  total_embeddings?: number | null;
  error_message?: string | null;
  updated_at?: string | null;
};

export type Citation = {
  source_id: number;
  chunk_id: number;
  document_id?: number | null;
  kb_id?: number | null;
  filename: string;
  location?: string | null;
  score: number;
  quote: string;
};

export type AnswerResponse = {
  query: string;
  rewritten_query?: string | null;
  answer: string;
  conversation_id: number;
  citations: Citation[];
  retrieved_count: number;
  degraded: boolean;
  context_compacted: boolean;
  timings_ms: Record<string, number>;
};

export type AnswerQuestionPayload = {
  query: string;
  kb_id?: number;
  document_id?: number;
  conversation_id?: number;
  top_k: number;
  bm25_weight: number;
  use_memory: boolean;
  rewrite_query: boolean;
};

export type AgentTask = {
  session_id: string;
  task_id: string;
  run_id: string;
  status: "pending" | "running" | "waiting_user" | "completed" | "failed" | "cancelled";
  user_input: string;
  task_type: string;
  kb_id?: number | null;
  document_id?: number | null;
  conversation_id?: number | null;
  final_answer?: string | null;
  created_at: string;
  updated_at: string;
};

export type AgentEvent = {
  session_id: string;
  task_id: string;
  run_id?: string | null;
  event_type: string;
  event_index: number;
  agent_name?: string | null;
  skill_name?: string | null;
  tool_name?: string | null;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
};

const API_BASE = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "";
const ACCESS_KEY = "interview_improvement_rag_access";
const REFRESH_KEY = "interview_improvement_rag_refresh";
const USER_KEY = "interview_improvement_rag_user";

export const tokenStore = {
  access: () => localStorage.getItem(ACCESS_KEY) ?? "",
  refresh: () => localStorage.getItem(REFRESH_KEY) ?? "",
  username: () => localStorage.getItem(USER_KEY) ?? "",
  set(tokens: TokenResponse, username?: string) {
    localStorage.setItem(ACCESS_KEY, tokens.access_token);
    localStorage.setItem(REFRESH_KEY, tokens.refresh_token);
    if (username) localStorage.setItem(USER_KEY, username);
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
  }
};

function authHeaders() {
  return { Authorization: `Bearer ${tokenStore.access()}` };
}

async function parseError(response: Response) {
  const raw = await response.text().catch(() => "");
  try {
    const data = raw ? JSON.parse(raw) : {};
    if (typeof data.detail === "string") return data.detail;
    if (data.detail) return JSON.stringify(data.detail);
  } catch {
    if (raw) return raw;
  }
  return `请求失败 (HTTP ${response.status})`;
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...authHeaders(),
      ...(init.headers ?? {})
    }
  });
  if (!response.ok) throw new Error(await parseError(response));
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function login(username: string, password: string) {
  const response = await fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<TokenResponse>;
}

export async function register(username: string, password: string) {
  const response = await fetch(`${API_BASE}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export function listKnowledgeBases() {
  return requestJson<KnowledgeBase[]>("/kb");
}

export function createKnowledgeBase(name: string, description: string) {
  return requestJson<KnowledgeBase>("/kb", {
    method: "POST",
    body: JSON.stringify({ name, description })
  });
}

export function listDocuments(kbId?: number) {
  const suffix = kbId ? `?kb_id=${kbId}` : "";
  return requestJson<DocumentItem[]>(`/document/list${suffix}`);
}

export async function uploadDocument(file: File, kbId?: number) {
  const form = new FormData();
  form.append("file", file);
  if (kbId) form.append("kb_id", String(kbId));
  const response = await fetch(`${API_BASE}/document/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: form
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export function answerQuestion(payload: AnswerQuestionPayload) {
  return requestJson<AnswerResponse>("/embedding/rag/answer", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

type StreamHandlers = {
  onToken?: (delta: string) => void;
  onFinal?: (response: AnswerResponse) => void;
};

function parseSseBlock(block: string) {
  let event = "message";
  const data: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator >= 0 ? line.slice(0, separator) : line;
    let value = separator >= 0 ? line.slice(separator + 1) : "";
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  return { event, data: data.join("\n") };
}

export async function answerQuestionStream(
  payload: AnswerQuestionPayload,
  handlers: StreamHandlers = {}
) {
  const response = await fetch(`${API_BASE}/embedding/rag/answer/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders()
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(await parseError(response));
  if (!response.body) return answerQuestion(payload);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: AnswerResponse | null = null;

  const consume = (block: string) => {
    const packet = parseSseBlock(block);
    if (!packet.data) return;
    const data = JSON.parse(packet.data);
    if (packet.event === "token") {
      handlers.onToken?.(String(data.delta ?? ""));
      return;
    }
    if (packet.event === "final") {
      finalResponse = data as AnswerResponse;
      handlers.onFinal?.(finalResponse);
      return;
    }
    if (packet.event === "error") {
      throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let separator = buffer.indexOf("\n\n");
    while (separator >= 0) {
      consume(buffer.slice(0, separator));
      buffer = buffer.slice(separator + 2);
      separator = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode().replace(/\r\n/g, "\n");
  if (buffer.trim()) consume(buffer);
  if (!finalResponse) throw new Error("Stream ended before the final answer was received.");
  return finalResponse;
}

export function createAgentTask(payload: {
  user_input: string;
  task_type?: string;
  kb_id?: number;
  conversation_id?: number;
  max_steps?: number;
  max_tool_calls?: number;
}) {
  return requestJson<{ session_id: string; task_id: string; run_id: string; status: string }>("/agent/tasks", {
    method: "POST",
    body: JSON.stringify({
      task_type: "interview_improvement",
      ...payload,
      idempotency_key: `interview-${Date.now()}`
    })
  });
}

export function listAgentTasks() {
  return requestJson<AgentTask[]>("/agent/tasks");
}

export function listAgentEvents(taskId: string, afterIndex = 0) {
  return requestJson<AgentEvent[]>(`/agent/tasks/${taskId}/events?after_index=${afterIndex}`);
}

export function getDocumentProgress(documentId: number) {
  return requestJson<DocumentProgress>(`/document/${documentId}/progress`);
}

export function reindexDocument(documentId: number) {
  return requestJson<{ id: number; file_name: string; status: string; task_id: string; kb_id?: number | null }>(
    `/document/${documentId}/reindex`,
    { method: "POST" }
  );
}

type AgentStreamHandlers = {
  onEvent?: (event: AgentEvent) => void;
  onToken?: (delta: string) => void;
};

export async function streamAgentEvents(
  taskId: string,
  afterIndex = 0,
  handlers: AgentStreamHandlers = {},
  signal?: AbortSignal
) {
  const response = await fetch(`${API_BASE}/agent/tasks/${taskId}/stream?after_index=${afterIndex}`, {
    headers: authHeaders(),
    signal
  });
  if (!response.ok) throw new Error(await parseError(response));
  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const consume = (block: string) => {
    const packet = parseSseBlock(block);
    if (!packet.data) return;
    const data = JSON.parse(packet.data);
    if (packet.event === "llm.token") {
      handlers.onToken?.(String(data.payload?.text ?? data.text ?? ""));
      return;
    }
    if (typeof data.event_index === "number") handlers.onEvent?.(data as AgentEvent);
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let separator = buffer.indexOf("\n\n");
    while (separator >= 0) {
      consume(buffer.slice(0, separator));
      buffer = buffer.slice(separator + 2);
      separator = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode().replace(/\r\n/g, "\n");
  if (buffer.trim()) consume(buffer);
}

export function cancelAgentTask(taskId: string) {
  return requestJson<AgentTask>(`/agent/tasks/${taskId}/cancel`, { method: "POST" });
}
