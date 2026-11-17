/**
 * DocuMind Client Application
 * Hybrid RAG Assistant with Real-time Grounding & Document Utilities
 */

// Application State
const state = {
  documents: [],
  totalChunks: 0,
  history: [], // [{role: 'user', content: '...'}, {role: 'model', content: '...'}]
  activeScope: "",
  settings: {
    provider: "gemini",
    apiKey: "",
    model: "gemini-1.5-flash",
    customEndpoint: "",
    chunkSize: 700,
    topK: 5
  },
  activeCitation: null,
  isProcessing: false
};

// DOM Elements
const docListEl = document.getElementById("documents-list");
const docScopeSelect = document.getElementById("doc-scope-select");
const docCountBadge = document.getElementById("doc-count-badge");
const totalChunksLabel = document.getElementById("total-chunks-label");
const activeEngineBadge = document.getElementById("active-engine-badge");
const activeProviderLabel = document.getElementById("active-provider-label");
const messagesContainer = document.getElementById("messages-container");
const welcomeHero = document.getElementById("welcome-hero");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat-btn");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const loadSamplesBtn = document.getElementById("load-samples-btn");
const clearDocsBtn = document.getElementById("clear-docs-btn");

// Modals
const settingsModal = document.getElementById("settings-modal");
const openSettingsBtn = document.getElementById("open-settings-btn");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const cancelSettingsBtn = document.getElementById("cancel-settings-btn");
const settingsForm = document.getElementById("settings-form");

const docPreviewModal = document.getElementById("doc-preview-modal");
const closePreviewBtn = document.getElementById("close-preview-btn");
const previewModalTitle = document.getElementById("preview-modal-title");
const previewModalContent = document.getElementById("preview-modal-content");
const previewStatsBar = document.getElementById("preview-stats-bar");

const citationModal = document.getElementById("citation-modal");
const closeCitationBtn = document.getElementById("close-citation-btn");
const citationFilename = document.getElementById("citation-filename");
const citationPage = document.getElementById("citation-page");
const citationScore = document.getElementById("citation-score");
const citationSnippet = document.getElementById("citation-snippet");

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  loadStoredSettings();
  fetchDocuments();
  setupEventListeners();
  autoResizeTextarea();
});

// Setup Event Listeners
function setupEventListeners() {
  // Dropzone drag-and-drop
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-indigo-500", "bg-indigo-950/30");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-indigo-500", "bg-indigo-950/30");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-indigo-500", "bg-indigo-950/30");
    if (e.dataTransfer.files.length > 0) {
      handleFilesUpload(e.dataTransfer.files);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFilesUpload(e.target.files);
    }
  });

  // Action Buttons
  loadSamplesBtn.addEventListener("click", loadSampleDocuments);
  clearDocsBtn.addEventListener("click", clearAllDocuments);
  clearChatBtn.addEventListener("click", clearChat);

  // Chat Submission
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  // Scope Change
  docScopeSelect.addEventListener("change", (e) => {
    state.activeScope = e.target.value;
  });

  // Quick Action Buttons
  document.querySelectorAll(".quick-action-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      executeQuickAction(action);
    });
  });

  // Suggestion Prompt Chips
  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const text = chip.querySelector("span").textContent.trim();
      userInput.value = text;
      handleSendMessage();
    });
  });

  // Settings Modal Controls
  openSettingsBtn.addEventListener("click", openSettings);
  closeSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));
  cancelSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));
  document.getElementById("setting-provider").addEventListener("change", (e) => {
    const isOllama = e.target.value === "ollama";
    document.getElementById("custom-endpoint-group").classList.toggle("hidden", !isOllama);
    const isOffline = e.target.value === "offline";
    document.getElementById("api-key-group").classList.toggle("opacity-50", isOffline);
  });
  settingsForm.addEventListener("submit", saveSettings);

  // Sliders display values
  document.getElementById("setting-chunk-size").addEventListener("input", (e) => {
    document.getElementById("chunk-size-val").textContent = e.target.value;
  });
  document.getElementById("setting-top-k").addEventListener("input", (e) => {
    document.getElementById("top-k-val").textContent = e.target.value;
  });

  // Modal Closures
  closePreviewBtn.addEventListener("click", () => docPreviewModal.classList.add("hidden"));
  closeCitationBtn.addEventListener("click", () => citationModal.classList.add("hidden"));

  // Click outside modal to close
  window.addEventListener("click", (e) => {
    if (e.target === settingsModal) settingsModal.classList.add("hidden");
    if (e.target === docPreviewModal) docPreviewModal.classList.add("hidden");
    if (e.target === citationModal) citationModal.classList.add("hidden");
  });
}

function autoResizeTextarea() {
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
  });
}

// ==================== DOCUMENT MANAGEMENT ====================

async function fetchDocuments() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    state.documents = data.documents || [];
    state.totalChunks = data.total_chunks || 0;
    renderDocumentsList();
    updateScopeSelect();
  } catch (err) {
    console.error("Failed to fetch documents:", err);
  }
}

function renderDocumentsList() {
  docCountBadge.textContent = `${state.documents.length} docs`;
  totalChunksLabel.textContent = `${state.totalChunks} indexed chunks`;

  if (state.documents.length === 0) {
    docListEl.innerHTML = `
      <div id="empty-docs-placeholder" class="text-center py-8 text-slate-500 text-xs">
        <i data-lucide="files" class="w-8 h-8 mx-auto mb-2 text-slate-600 opacity-60"></i>
        No documents loaded.<br>Upload files or load samples to begin.
      </div>
    `;
    lucide.createIcons();
    return;
  }

  docListEl.innerHTML = "";
  state.documents.forEach(doc => {
    const ext = doc.file_type || doc.filename.split('.').pop();
    const iconName = getFileIcon(ext);
    
    const card = document.createElement("div");
    card.className = "p-2.5 rounded-xl bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 transition flex items-center justify-between group";
    card.innerHTML = `
      <div class="flex items-center gap-2.5 min-w-0 flex-1 cursor-pointer view-doc-btn" data-filename="${escapeHtml(doc.filename)}">
        <div class="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center flex-shrink-0">
          <i data-lucide="${iconName}" class="w-4 h-4"></i>
        </div>
        <div class="min-w-0 flex-1">
          <p class="text-xs font-medium text-slate-200 truncate group-hover:text-indigo-300 transition" title="${escapeHtml(doc.filename)}">
            ${escapeHtml(doc.filename)}
          </p>
          <div class="flex items-center gap-2 text-[10px] text-slate-400 mt-0.5">
            <span>${doc.chunk_count} chunks</span>
            <span>•</span>
            <span>${Math.round(doc.total_chars / 100) / 10}k chars</span>
          </div>
        </div>
      </div>
      <div class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition flex-shrink-0">
        <button class="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-white view-doc-btn" data-filename="${escapeHtml(doc.filename)}" title="Preview Text">
          <i data-lucide="eye" class="w-3.5 h-3.5"></i>
        </button>
        <button class="p-1 rounded hover:bg-rose-950/60 text-slate-400 hover:text-rose-400 delete-doc-btn" data-filename="${escapeHtml(doc.filename)}" title="Remove">
          <i data-lucide="trash" class="w-3.5 h-3.5"></i>
        </button>
      </div>
    `;

    // Bind item click
    card.querySelectorAll(".view-doc-btn").forEach(btn => {
      btn.addEventListener("click", () => openDocPreview(doc.filename));
    });
    card.querySelector(".delete-doc-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteDocument(doc.filename);
    });

    docListEl.appendChild(card);
  });

  lucide.createIcons();
}

function updateScopeSelect() {
  const currentVal = docScopeSelect.value;
  docScopeSelect.innerHTML = `<option value="">All Documents (Full Corpus)</option>`;
  state.documents.forEach(doc => {
    const opt = document.createElement("option");
    opt.value = doc.filename;
    opt.textContent = doc.filename;
    if (doc.filename === currentVal) opt.selected = true;
    docScopeSelect.appendChild(opt);
  });
}

function getFileIcon(ext) {
  ext = (ext || "").toLowerCase();
  if (ext === "pdf") return "file-badge";
  if (ext === "docx" || ext === "doc") return "file-text";
  if (ext === "csv" || ext === "tsv") return "table";
  if (ext === "json") return "file-json";
  if (ext === "md" || ext === "txt") return "file-lines";
  if (["py", "js", "ts", "html"].includes(ext)) return "file-code";
  return "file";
}

// Upload Files
async function handleFilesUpload(files) {
  for (const file of files) {
    const filename = file.name;
    const isBinary = file.name.endsWith(".pdf") || file.name.endsWith(".docx");

    try {
      showToast(`Uploading ${filename}...`, "info");
      
      let payload = { filename: filename };
      if (isBinary) {
        const base64 = await fileToBase64(file);
        payload.content_base64 = base64;
      } else {
        const text = await fileToText(file);
        payload.content_text = text;
      }

      const res = await fetch("/api/documents/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonSafe(payload)
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Indexed ${filename} (${data.chunk_count} chunks)`, "success");
      } else {
        showToast(`Failed ${filename}: ${data.error}`, "error");
      }
    } catch (err) {
      showToast(`Upload error: ${err.message}`, "error");
    }
  }
  await fetchDocuments();
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = error => reject(error);
  });
}

function fileToText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsText(file, "UTF-8");
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
}

// Load Samples
async function loadSampleDocuments() {
  try {
    showToast("Loading 3 enterprise sample documents...", "info");
    const res = await fetch("/api/documents/load-samples", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      showToast(`Loaded ${data.document_count} sample documents!`, "success");
      await fetchDocuments();
    }
  } catch (err) {
    showToast("Failed to load sample documents", "error");
  }
}

// Clear All
async function clearAllDocuments() {
  if (!confirm("Are you sure you want to remove all indexed documents?")) return;
  try {
    await fetch("/api/documents/clear", { method: "POST" });
    await fetchDocuments();
    showToast("All documents cleared.", "info");
  } catch (err) {
    showToast("Failed to clear documents", "error");
  }
}

// Delete Single Document
async function deleteDocument(filename) {
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      showToast(`Removed ${filename}`, "info");
      await fetchDocuments();
    }
  } catch (err) {
    showToast(`Error deleting document: ${err.message}`, "error");
  }
}

// Open Document Preview Modal
async function openDocPreview(filename) {
  try {
    const res = await fetch(`/api/documents/content?filename=${encodeURIComponent(filename)}`);
    const data = await res.json();
    if (data.error) {
      showToast(data.error, "error");
      return;
    }
    previewModalTitle.textContent = data.filename;
    previewStatsBar.innerHTML = `
      <span><strong>Chunks:</strong> ${data.chunks.length}</span>
      <span><strong>Total Characters:</strong> ${data.total_chars.toLocaleString()}</span>
      ${data.truncated ? '<span class="text-amber-400 font-semibold">• Preview truncated to first 15k chars</span>' : ''}
    `;
    previewModalContent.textContent = data.full_text;
    docPreviewModal.classList.remove("hidden");
    lucide.createIcons();
  } catch (err) {
    showToast("Failed to load document content", "error");
  }
}

// ==================== CHAT & RAG ENGINE ====================

async function handleSendMessage() {
  const query = userInput.value.trim();
  if (!query || state.isProcessing) return;

  // Clear input
  userInput.value = "";
  userInput.style.height = "auto";

  // Hide welcome hero
  if (welcomeHero) welcomeHero.classList.add("hidden");

  // Append user message bubble
  appendMessage("user", query);
  state.history.push({ role: "user", content: query });

  // Add assistant typing indicator
  const typingEl = appendTypingIndicator();
  state.isProcessing = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        filter_filename: state.activeScope || null,
        history: state.history,
        top_k: state.settings.topK
      })
    });

    const data = await res.json();
    typingEl.remove();

    if (data.answer) {
      appendMessage("assistant", data.answer, data.citations, data.provider, data.model);
      state.history.push({ role: "model", content: data.answer });
    } else {
      appendMessage("assistant", "⚠️ " + (data.error || "An unexpected error occurred."));
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("assistant", `⚠️ **Network/Server Error**: ${err.message}`);
  } finally {
    state.isProcessing = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// Execute Quick Action (Summary, Actions, FAQ, Risks)
async function executeQuickAction(action) {
  if (state.documents.length === 0) {
    showToast("Please upload documents or load samples first!", "warning");
    return;
  }

  const actionTitles = {
    summary: "📑 Generating Executive Summary...",
    actions: "🎯 Extracting Action Items & Deliverables...",
    faq: "❓ Formulating Auto FAQ...",
    risks: "⚠️ Running Risk & Compliance Audit..."
  };

  if (welcomeHero) welcomeHero.classList.add("hidden");

  appendMessage("user", `[Quick Action: ${action.toUpperCase()}] Requesting document analysis.`);
  const typingEl = appendTypingIndicator();
  state.isProcessing = true;

  try {
    const res = await fetch("/api/tools/quick-action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: action,
        filter_filename: state.activeScope || null
      })
    });

    const data = await res.json();
    typingEl.remove();

    if (data.answer) {
      appendMessage("assistant", data.answer, data.citations, data.provider);
      state.history.push({ role: "model", content: data.answer });
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("assistant", `⚠️ Error executing tool: ${err.message}`);
  } finally {
    state.isProcessing = false;
  }
}

// Append Message Bubble
function appendMessage(role, text, citations = [], provider = "", model = "") {
  const msgWrapper = document.createElement("div");
  msgWrapper.className = `flex gap-3 max-w-4xl mx-auto ${role === "user" ? "justify-end" : "justify-start"}`;

  if (role === "user") {
    msgWrapper.innerHTML = `
      <div class="max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-600 text-white px-4 py-3 text-sm shadow-md leading-relaxed whitespace-pre-wrap">
        ${escapeHtml(text)}
      </div>
    `;
  } else {
    // Assistant Message with Markdown + Citations
    const htmlContent = marked.parse(text);
    const sanitizedHtml = DOMPurify.sanitize(htmlContent);

    let citationsHtml = "";
    if (citations && citations.length > 0) {
      citationsHtml = `
        <div class="mt-3 pt-3 border-t border-slate-800 flex flex-wrap items-center gap-1.5 text-xs">
          <span class="text-slate-400 font-semibold text-[11px] flex items-center gap-1 mr-1">
            <i data-lucide="check-check" class="w-3.5 h-3.5 text-emerald-400"></i> Sources:
          </span>
          ${citations.map((c, idx) => `
            <button class="citation-pill" data-chunk-id="${escapeHtml(c.chunk_id)}" data-filename="${escapeHtml(c.filename)}" data-page="${c.page_num}" data-score="${c.relevance_score}">
              <i data-lucide="file-check-2" class="w-3 h-3"></i>
              <span>${escapeHtml(c.filename)} (P.${c.page_num})</span>
            </button>
          `).join("")}
        </div>
      `;
    }

    const badgeInfo = provider ? `<span class="text-[10px] text-slate-400 font-mono">${escapeHtml(provider)} ${model ? `• ${escapeHtml(model)}` : ''}</span>` : "";

    msgWrapper.innerHTML = `
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white flex-shrink-0 shadow-md shadow-indigo-500/20 mt-1">
        <i data-lucide="bot" class="w-4 h-4"></i>
      </div>
      <div class="max-w-[85%] rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800/80 px-5 py-4 shadow-xl">
        <div class="flex items-center justify-between mb-2 pb-1 border-b border-slate-800/50">
          <span class="text-xs font-bold text-slate-200">DocuMind</span>
          ${badgeInfo}
        </div>
        <div class="markdown-body">
          ${sanitizedHtml}
        </div>
        ${citationsHtml}
        <div class="mt-3 flex items-center justify-end gap-2 text-slate-400 text-xs">
          <button class="copy-msg-btn hover:text-white transition flex items-center gap-1" title="Copy response">
            <i data-lucide="copy" class="w-3.5 h-3.5"></i>
            <span class="text-[11px]">Copy</span>
          </button>
        </div>
      </div>
    `;

    // Bind Citation Pills
    if (citations && citations.length > 0) {
      msgWrapper.querySelectorAll(".citation-pill").forEach((pill, i) => {
        pill.addEventListener("click", () => {
          openCitationModal(citations[i]);
        });
      });
    }

    // Bind Copy Button
    const copyBtn = msgWrapper.querySelector(".copy-msg-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(text);
        copyBtn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i><span class="text-[11px] text-emerald-400">Copied!</span>`;
        lucide.createIcons();
        setTimeout(() => {
          copyBtn.innerHTML = `<i data-lucide="copy" class="w-3.5 h-3.5"></i><span class="text-[11px]">Copy</span>`;
          lucide.createIcons();
        }, 2000);
      });
    }
  }

  messagesContainer.appendChild(msgWrapper);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  lucide.createIcons();
}

function appendTypingIndicator() {
  const typingWrapper = document.createElement("div");
  typingWrapper.className = "flex gap-3 max-w-4xl mx-auto justify-start";
  typingWrapper.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-indigo-400 flex-shrink-0 mt-1">
      <i data-lucide="bot" class="w-4 h-4 animate-spin"></i>
    </div>
    <div class="rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-800 px-4 py-3 shadow-lg flex items-center gap-1.5">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <span class="text-xs text-slate-400 ml-2">Retrieving & synthesizing...</span>
    </div>
  `;
  messagesContainer.appendChild(typingWrapper);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  lucide.createIcons();
  return typingWrapper;
}

function clearChat() {
  state.history = [];
  messagesContainer.innerHTML = "";
  if (welcomeHero) {
    messagesContainer.appendChild(welcomeHero);
    welcomeHero.classList.remove("hidden");
  }
}

// Open Citation Modal
function openCitationModal(citation) {
  citationFilename.textContent = citation.filename;
  citationPage.textContent = `Page ${citation.page_num}`;
  citationScore.textContent = `Score: ${citation.relevance_score}`;
  citationSnippet.textContent = citation.snippet;
  citationModal.classList.remove("hidden");
}

// ==================== SETTINGS & STORAGE ====================

async function loadStoredSettings() {
  const saved = localStorage.getItem("documind_settings");
  if (saved) {
    try {
      state.settings = { ...state.settings, ...JSON.parse(saved) };
    } catch (e) {}
  }

  // Fetch backend settings
  try {
    const res = await fetch("/api/settings");
    const serverSettings = await res.json();
    if (serverSettings.has_api_key) {
      document.getElementById("api-key-status").textContent = `Active (${serverSettings.api_key_masked})`;
      document.getElementById("api-key-status").className = "text-[11px] text-emerald-400 font-semibold";
    }
  } catch (err) {}

  updateProviderUI();
}

function updateProviderUI() {
  const p = state.settings.provider;
  if (p === "gemini") {
    activeProviderLabel.textContent = `Google Gemini (${state.settings.model})`;
  } else if (p === "openai") {
    activeProviderLabel.textContent = `OpenAI (${state.settings.model})`;
  } else if (p === "ollama") {
    activeProviderLabel.textContent = `Local Ollama (${state.settings.model})`;
  } else {
    activeProviderLabel.textContent = `Offline Extractive Engine (Zero API Key)`;
  }
}

function openSettings() {
  document.getElementById("setting-provider").value = state.settings.provider;
  document.getElementById("setting-api-key").value = state.settings.apiKey || "";
  document.getElementById("setting-model").value = state.settings.model;
  document.getElementById("setting-endpoint").value = state.settings.customEndpoint || "";
  document.getElementById("setting-chunk-size").value = state.settings.chunkSize || 700;
  document.getElementById("chunk-size-val").textContent = state.settings.chunkSize || 700;
  document.getElementById("setting-top-k").value = state.settings.topK || 5;
  document.getElementById("top-k-val").textContent = state.settings.topK || 5;

  const isOllama = state.settings.provider === "ollama";
  document.getElementById("custom-endpoint-group").classList.toggle("hidden", !isOllama);

  settingsModal.classList.remove("hidden");
}

async function saveSettings(e) {
  e.preventDefault();
  const provider = document.getElementById("setting-provider").value;
  const apiKey = document.getElementById("setting-api-key").value.trim();
  const model = document.getElementById("setting-model").value.trim();
  const customEndpoint = document.getElementById("setting-endpoint").value.trim();
  const chunkSize = parseInt(document.getElementById("setting-chunk-size").value, 10);
  const topK = parseInt(document.getElementById("setting-top-k").value, 10);

  state.settings = {
    provider,
    apiKey: apiKey || state.settings.apiKey,
    model,
    customEndpoint,
    chunkSize,
    topK
  };

  localStorage.setItem("documind_settings", JSON.stringify(state.settings));

  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider,
        api_key: apiKey,
        model,
        custom_endpoint: customEndpoint,
        chunk_size: chunkSize,
        top_k: topK
      })
    });
    showToast("Settings saved successfully!", "success");
  } catch (err) {
    showToast("Failed to save settings to server", "error");
  }

  updateProviderUI();
  settingsModal.classList.add("hidden");
}

// ==================== TOAST UTILITY ====================
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  const bgColors = {
    info: "bg-slate-800 border-slate-700 text-slate-200",
    success: "bg-emerald-950/90 border-emerald-700 text-emerald-200",
    warning: "bg-amber-950/90 border-amber-700 text-amber-200",
    error: "bg-rose-950/90 border-rose-700 text-rose-200"
  };

  toast.className = `fixed bottom-20 right-6 z-50 px-4 py-2.5 rounded-xl border text-xs shadow-2xl flex items-center gap-2 transform transition-all duration-300 translate-y-4 opacity-0 ${bgColors[type] || bgColors.info}`;
  toast.innerHTML = `
    <span>${escapeHtml(message)}</span>
  `;

  document.body.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove("translate-y-4", "opacity-0");
  });

  setTimeout(() => {
    toast.classList.add("translate-y-4", "opacity-0");
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Helpers
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function jsonSafe(obj) {
  return JSON.stringify(obj);
}
