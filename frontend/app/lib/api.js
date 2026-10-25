/**
 * Dynamically resolves the full backend API URL using process.env.NEXT_PUBLIC_API_URL.
 * Prevents double slashes (e.g., if env has a trailing slash) and ensures
 * no trailing slash at the end of the route.
 */
export function getApiBase() {
  let raw = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").trim();

  // If set with markdown link syntax like [https://...](https://...)
  const mdMatch = raw.match(/\[.*?\]\((https?:\/\/[^\s\)]+)\)/i);
  if (mdMatch) {
    raw = mdMatch[1];
  } else {
    // If set with markdown label like [https://ir-sem-5.onrender.com]
    const labelMatch = raw.match(/\[(https?:\/\/[^\]]+)\]/i);
    if (labelMatch) {
      raw = labelMatch[1];
    }
  }

  // Strip wrapping quotes and brackets if any
  raw = raw.replace(/^["'\[]+|["'\]]+$/g, "").trim();

  // Add https:// if protocol is missing (unless localhost)
  if (!/^https?:\/\//i.test(raw)) {
    if (raw.startsWith("localhost") || raw.startsWith("127.0.0.1")) {
      raw = `http://${raw}`;
    } else {
      raw = `https://${raw}`;
    }
  }

  return raw.replace(/\/+$/, "");
}

export function resolveApiUrl(path) {
  const base = getApiBase();
  // Separate query string if present
  const [pathname, search] = path.split("?");
  // Ensure exactly one leading slash before api, and strip any trailing slash at the end of the route
  const cleanPathname = ("/" + pathname.replace(/^\/+/, "")).replace(/\/+$/, "");
  const queryString = search ? `?${search}` : "";

  return `${base}${cleanPathname}${queryString}`;
}

/** Sync the token to both localStorage AND a cookie so middleware can read it. */
function persistToken(token) {
  if (typeof window === "undefined") return;
  if (token) {
    localStorage.setItem("token", token);
    // SameSite=Lax; not HttpOnly so JS can clear it; max-age 7 days
    document.cookie = `auth_token=${encodeURIComponent(token)}; path=/; max-age=${60 * 60 * 24 * 7}; SameSite=Lax`;
  } else {
    localStorage.removeItem("token");
    document.cookie = "auth_token=; path=/; max-age=0; SameSite=Lax";
  }
}

/**
 * fetch wrapper that adds auth + JSON handling.
 */
async function apiFetch(path, options = {}) {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;

  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const method = options.method || "GET";
  const url = resolveApiUrl(path);
  const startTime = performance.now();
  console.log(`[API] → ${method} ${url}`);

  const res = await fetch(url, { ...options, headers });
  const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

  if (res.status === 204) {
    console.log(`[API] ← ${method} ${url} → 204 (${elapsed}s)`);
    return null;
  }

  let data = null;
  let text = "";
  try {
    text = await res.text();
    data = JSON.parse(text);
  } catch {
    // text remains raw string if not JSON
  }

  if (!res.ok) {
    console.error(`[API] <- ${method} ${url} -> ${res.status} (${elapsed}s)`, data || text);
    const detailMsg = data?.detail || text || `Request failed (${res.status})`;
    throw new Error(detailMsg);
  }

  console.log(`[API] <- ${method} ${url} -> ${res.status} (${elapsed}s)`);
  return data;
}

// ── Auth ───────────────────────────────────────────────────────────
export async function register(email, name, password) {
  const data = await apiFetch("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, name, password }),
  });
  persistToken(data.access_token);
  return data;
}

export async function login(email, password) {
  const data = await apiFetch("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  persistToken(data.access_token);
  return data;
}

export async function getMe() {
  return apiFetch("/api/auth/me");
}

export function logout() {
  persistToken(null); // clears both localStorage and cookie
}

// ── PDFs ───────────────────────────────────────────────────────────
export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch("/api/pdfs/upload", { method: "POST", body: formData });
}

export async function listPdfs() {
  return apiFetch("/api/pdfs/");
}

export async function deletePdf(pdfId) {
  return apiFetch(`/api/pdfs/${pdfId}`, { method: "DELETE" });
}

export function getPdfViewUrl(pdfId) {
  const token = typeof window !== "undefined" ? localStorage.getItem("token") : "";
  const base = getApiBase();
  return `${base}/api/pdfs/${pdfId}/view?token=${encodeURIComponent(token || "")}`;
}

// ── Chats ──────────────────────────────────────────────────────────
export async function createChat(pdfId) {
  return apiFetch(`/api/chats/?pdf_id=${pdfId}`, { method: "POST" });
}

export async function listChats() {
  return apiFetch("/api/chats/");
}

export async function getChat(chatId) {
  return apiFetch(`/api/chats/${chatId}`);
}

export async function deleteChat(chatId) {
  return apiFetch(`/api/chats/${chatId}`, { method: "DELETE" });
}

export async function renameChat(chatId, title) {
  return apiFetch(`/api/chats/${chatId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function sendMessage(chatId, content) {
  return apiFetch(`/api/chats/${chatId}/message`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}
