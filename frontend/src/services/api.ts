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
  created_at: string;
  kb_id?: number | null;
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

export function answerQuestion(payload: {
  query: string;
  kb_id?: number;
  document_id?: number;
  conversation_id?: number;
  top_k: number;
  bm25_weight: number;
  use_memory: boolean;
  rewrite_query: boolean;
}) {
  return requestJson<AnswerResponse>("/embedding/rag/answer", {
    method: "POST",
    body: JSON.stringify(payload)
  });
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

export function cancelAgentTask(taskId: string) {
  return requestJson<AgentTask>(`/agent/tasks/${taskId}/cancel`, { method: "POST" });
}
