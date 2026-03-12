"""
Legal AI Agent — FastAPI Backend
Analyzes legal documents: summaries, clause extraction, risk analysis, AI chat.
"""
import json
import logging
import os
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("legal-ai")

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Legal AI Agent", version="1.0.0")


@app.on_event("startup")
async def startup_check():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or not key.startswith("sk-"):
        log.error("=" * 60)
        log.error("ANTHROPIC_API_KEY is missing or invalid!")
        log.error("Add it to your .env file:  ANTHROPIC_API_KEY=sk-ant-...")
        log.error("=" * 60)
    else:
        log.info(f"Anthropic API key loaded (sk-...{key[-6:]})")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory document store: {doc_id: {...}}
# ---------------------------------------------------------------------------
documents: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Anthropic client  (async — required for use inside FastAPI async endpoints)
# ---------------------------------------------------------------------------
client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

# Validate required packages are available at startup (not silently at upload time)
try:
    from pypdf import PdfReader as _PdfReader
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False
    log.warning("pypdf not installed — PDF uploads will fail. Run: pip install pypdf")

try:
    from docx import Document as _DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False
    log.warning("python-docx not installed — DOCX uploads will fail. Run: pip install python-docx")


def _extract_pdf(data: bytes) -> str:
    if not _PYPDF_OK:
        raise HTTPException(
            status_code=500,
            detail="pypdf is not installed on the server. Run: pip install pypdf",
        )
    try:
        reader = _PdfReader(BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open PDF: {exc}")

    pages_text: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
            if t.strip():
                pages_text.append(t)
        except Exception as exc:
            log.warning("PDF page %d extraction skipped: %s", i, exc)

    result = "\n\n".join(pages_text).strip()
    log.info("PDF: extracted %d chars from %d/%d pages", len(result), len(pages_text), len(reader.pages))

    if not result:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this PDF. "
                "It may be a scanned image-only PDF. "
                "Try converting it to a text-based PDF or copy-paste the text into a .txt file."
            ),
        )
    return result


def _extract_docx(data: bytes) -> str:
    if not _DOCX_OK:
        raise HTTPException(
            status_code=500,
            detail="python-docx is not installed on the server. Run: pip install python-docx",
        )
    try:
        doc = _DocxDocument(BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open DOCX: {exc}")

    parts: list[str] = []

    # Top-level paragraphs
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)

    # Tables — iterate every cell in every table
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    t = p.text.strip()
                    if t:
                        parts.append(t)

    # Headers and footers
    for section in doc.sections:
        for hf in (section.header, section.footer):
            if hf:
                for p in hf.paragraphs:
                    t = p.text.strip()
                    if t:
                        parts.append(t)

    result = "\n\n".join(parts).strip()
    log.info("DOCX: extracted %d chars from %d text blocks", len(result), len(parts))

    if not result:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this DOCX. "
                "The document may be empty or contain only images/embedded objects."
            ),
        )
    return result


def extract_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext in (".docx", ".doc"):
        return _extract_docx(data)
    if ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            raise HTTPException(status_code=422, detail="The text file is empty.")
        return text
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type '{ext}'. Supported: PDF, DOCX, DOC, TXT, MD.",
    )


# ---------------------------------------------------------------------------
# Claude analysis helpers
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM = """You are an expert legal analyst AI. Your task is to analyze legal documents
and produce structured JSON output. Always respond with ONLY valid JSON — no markdown fences,
no extra text. Be thorough, precise, and highlight risks clearly."""

# NOTE: Use plain string concatenation for the document text — never .format() or
# f-strings — because legal documents routinely contain { } characters (e.g. ${1,000},
# Section {a}(i), exhibit labels) which would crash Python's str.format().
ANALYSIS_PROMPT_PREFIX = """Analyze the following legal document and return a JSON object with
exactly this structure (no extra keys):

{
  "document_type": "string (e.g. Service Agreement, NDA, Employment Contract, Lease, etc.)",
  "parties": ["list of party names as strings"],
  "effective_date": "string or null",
  "expiry_date": "string or null",
  "governing_law": "string or null",
  "summary": "3-5 sentence executive summary of the document",
  "risk_level": "Low | Medium | High",
  "risk_factors": ["list of specific risk concerns as strings"],
  "key_obligations": [
    {"party": "party name", "obligation": "what they must do"}
  ],
  "clauses": [
    {
      "type": "clause category (e.g. Termination, Payment, Confidentiality, Liability, IP, Non-Compete, Dispute Resolution, Force Majeure, Indemnification, Amendment)",
      "title": "short clause title",
      "content": "verbatim or close-paraphrase of the clause text",
      "risk_level": "Low | Medium | High",
      "risk_note": "brief note on why this clause is risky or what to watch out for (null if low risk)"
    }
  ]
}

Document to analyze:
---
"""

ANALYSIS_PROMPT_SUFFIX = "\n---"


async def analyze_document(text: str, filename: str) -> dict:
    """Call Claude to analyze a legal document. Returns structured dict."""
    truncated = text[:150_000]
    # Build prompt by concatenation — NOT .format() — to avoid KeyError on { } in docs
    prompt = ANALYSIS_PROMPT_PREFIX + truncated + ANALYSIS_PROMPT_SUFFIX

    log.info("Sending %d chars to Claude for analysis…", len(truncated))
    response = await client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    log.info("Claude response received (stop_reason=%s)", response.stop_reason)

    raw = next(
        (b.text for b in response.content if b.type == "text"), "{}"
    )

    # Strip accidental markdown fences if model added them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback minimal structure
        return {
            "document_type": "Unknown",
            "parties": [],
            "effective_date": None,
            "expiry_date": None,
            "governing_law": None,
            "summary": "Analysis could not be parsed. Please review the document manually.",
            "risk_level": "Medium",
            "risk_factors": [],
            "key_obligations": [],
            "clauses": [],
        }


async def stream_chat_response(
    doc_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """Stream a Claude response for document chat as SSE events."""
    doc = documents.get(doc_id)
    if not doc:
        yield f"data: {json.dumps({'error': 'Document not found'})}\n\n"
        return

    analysis = doc.get("analysis", {})
    doc_type = analysis.get("document_type", "Unknown")
    parties  = ", ".join(analysis.get("parties", []))

    # Build system prompt by concatenation — NOT .format() — to avoid KeyError on { } in docs
    system_prompt = (
        "You are an expert AI legal assistant. You are helping a user understand a legal document. "
        "Be clear, precise, and helpful. When referencing specific clauses, cite them. "
        "Flag any risks or obligations the user should be aware of.\n\n"
        f"Document type: {doc_type}\n"
        f"Parties: {parties}\n\n"
        "Full document text:\n---\n"
        + doc["text"][:120_000]
        + "\n---"
    )

    # Build message history (keep last 20 turns)
    history = doc.get("chat_history", [])
    messages = history[-20:] + [{"role": "user", "content": user_message}]

    # Stream via Claude
    async with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    ) as stream:
        full_response = ""
        async for text_chunk in stream.text_stream:
            full_response += text_chunk
            yield f"data: {json.dumps({'delta': text_chunk})}\n\n"

    # Persist conversation history
    doc["chat_history"].append({"role": "user", "content": user_message})
    doc["chat_history"].append({"role": "assistant", "content": full_response})

    yield f"data: {json.dumps({'done': True})}\n\n"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ClearHistoryRequest(BaseModel):
    pass


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a legal document, extract text, and auto-analyze with Claude."""
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")

    # ── Text extraction ──────────────────────────────────────────────────
    try:
        text = extract_text(data, file.filename or "document.txt")
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Text extraction error: %s", traceback.format_exc())
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    log.info("Extracted %d chars from '%s'", len(text), file.filename)

    # ── Claude analysis ──────────────────────────────────────────────────
    try:
        analysis = await analyze_document(text, file.filename or "document")
    except anthropic.AuthenticationError:
        log.error("Anthropic AuthenticationError — check ANTHROPIC_API_KEY in .env")
        raise HTTPException(
            status_code=401,
            detail="Invalid Anthropic API key. Open .env and set ANTHROPIC_API_KEY=sk-ant-...",
        )
    except anthropic.RateLimitError as exc:
        log.error("Anthropic RateLimitError: %s", exc)
        raise HTTPException(status_code=429, detail="Anthropic rate limit reached. Wait a moment and retry.")
    except anthropic.APIConnectionError as exc:
        log.error("Anthropic APIConnectionError: %s", exc)
        raise HTTPException(status_code=503, detail=f"Cannot reach Anthropic API: {exc}")
    except anthropic.BadRequestError as exc:
        log.error("Anthropic BadRequestError: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Anthropic rejected the request: {exc}")
    except anthropic.APIStatusError as exc:
        log.error("Anthropic APIStatusError %s: %s", exc.status_code, exc)
        raise HTTPException(status_code=502, detail=f"Anthropic API error {exc.status_code}: {exc}")
    except Exception as exc:
        log.error("Unexpected analysis error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    doc_id = str(uuid.uuid4())
    documents[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "size": len(data),
        "text": text,
        "analysis": analysis,
        "chat_history": [],
    }

    log.info("Document %s stored successfully", doc_id)
    return {
        "id": doc_id,
        "filename": file.filename,
        "size": len(data),
        "analysis": analysis,
    }


@app.get("/api/documents")
async def list_documents():
    """Return metadata for all uploaded documents (no full text)."""
    return [
        {
            "id": d["id"],
            "filename": d["filename"],
            "size": d["size"],
            "document_type": d["analysis"].get("document_type", "Unknown"),
            "risk_level": d["analysis"].get("risk_level", "Unknown"),
            "parties": d["analysis"].get("parties", []),
        }
        for d in documents.values()
    ]


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Return full analysis for a document."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "size": doc["size"],
        "analysis": doc["analysis"],
        "chat_history": doc["chat_history"],
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from memory."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    del documents[doc_id]
    return {"status": "deleted"}


@app.post("/api/documents/{doc_id}/chat")
async def chat_with_document(doc_id: str, body: ChatRequest):
    """Stream an AI response about the document as Server-Sent Events."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    return StreamingResponse(
        stream_chat_response(doc_id, body.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/documents/{doc_id}/chat")
async def clear_chat_history(doc_id: str):
    """Clear the conversation history for a document."""
    doc = documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc["chat_history"] = []
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.get("/api/test")
async def test_claude():
    """Quick smoke-test: sends a tiny message to Claude and returns the result."""
    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Reply with just the word: OK"}],
        )
        reply = next((b.text for b in response.content if b.type == "text"), "")
        return {"status": "ok", "model": MODEL, "reply": reply}
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid ANTHROPIC_API_KEY")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Claude test failed: {exc}")


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
