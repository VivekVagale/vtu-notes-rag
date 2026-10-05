# Integration guide

For the team adding this to a website. Everything here was measured or run on a
real instance; where something is untested it says so.

---

## 1. What you are receiving

A **backend service**. It answers questions from a library of VTU notes PDFs and
cites the file and page behind every claim. It also accepts PDF contributions
from visitors, keeps them private until a named owner approves them, and only
then lets them answer anyone's questions.

**It is not** a front end. There is no UI to drop in — a Streamlit app ships in
the repo for local use by the maintainer, not for your site. You build the
interface; you call the API below.

**It is not serverless-compatible.** It needs one always-on process with a
persistent disk. See §6.

### What runs where

```
student's browser                 your server                      outside
─────────────────                 ───────────────                  ───────
uploads PDF        ──HTTP──►      extract text (PyMuPDF)
                                  embed chunks (ONNX, local, free)
                                  store in Chroma
asks a question    ──HTTP──►      embed question (local, free)
                                  search + rerank → top pages
                                  write the answer ──────────────►  LLM provider
                   ◄──JSON───     answer + [file.pdf, p.14]
```

Your front end only ever speaks HTTP to this service. Nothing ships to the
browser — no model, no API key, no vector work. Embedding and search run
server-side at no per-use cost, and **uploading a PDF never calls an LLM at
all**, so contributions cost you CPU and disk but no API spend.

The only outbound dependency is writing the final prose. Even that is optional:
with `LLM_PROVIDER=extractive` the service still returns the correct pages and
citations, stitched from the notes' own sentences rather than written — see §7
for why that mode excludes contributed material. And `ollama` means a model
running on *your server*, not on the visitor's machine.

---

## 2. Decisions you need to make before deploying

| Decision | Why it matters | Where |
|---|---|---|
| **LLM provider** | `anthropic` (API key, costs money, best answers), `ollama` (self-hosted, free), or `extractive` (no model at all — see the warning in §7) | `LLM_PROVIDER` |
| **Who moderates** | Exactly one account approves contributed PDFs. Nobody is owner until you set this, and every moderation route refuses | `OWNER_EMAIL` |
| **Auth** | A local HMAC token driver ships for development. For production you will almost certainly replace it with your own identity provider — see §8 | `src/auth.py` |
| **Your site's origin** | The browser cannot call the API without it | `CORS_ORIGINS` |
| **Whether to accept uploads at all** | The read-only half (ask questions about the maintainer's own notes) works with none of the upload machinery enabled | — |

---

## 3. Run it

**Locally, to try it in five minutes:**

```bash
pip install -r requirements-service.txt
python scripts/make_sample_pdfs.py     # or drop real PDFs in data/pdfs/<Subject>/
python -m src.cli ingest
uvicorn src.api:app --port 8000
```

Then `http://localhost:8000/docs` is a live, clickable API console.

**As a container:** a `Dockerfile` is included. It was *not* built — Docker is
not installed on the authoring machine — but the dependency set it installs was
verified in a clean virtual environment (ingest + serve + cited answer, no
torch). Treat it as a starting point.

### Environment

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `ollama` \| `extractive` |
| `ANTHROPIC_API_KEY` | — | server-side only, never send it to a browser |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.1:8b` | |
| `EMBED_BACKEND` | `sentence-transformers` | **set `onnx` in production** — identical vectors, no torch |
| `PDF_DIR` / `INDEX_DIR` | `data/pdfs` / `data/index` | |
| `REGISTRY_PATH` / `UPLOAD_DIR` | `data/registry.db` / `data/uploads` | must sit outside `INDEX_DIR` and `PDF_DIR` |
| `OWNER_EMAIL` | unset | unset ⇒ nobody can moderate |
| `ALLOW_DEV_LOGIN` | `true` | **set `false` anywhere reachable from outside** |
| `AUTH_SECRET` | per-process random | unset ⇒ tokens die on restart |
| `CORS_ORIGINS` | `http://localhost:3000` | comma-separated |
| `EXTRACTIVE_COMMUNITY` | `false` | see §7 |
| `UPLOAD_MAX_MB` / `UPLOAD_MAX_PAGES` | `25` / `400` | |
| `QUOTA_DOCS_TOTAL` / `QUOTA_BYTES_TOTAL` | `30` / `300MB` | per account |
| `TOP_K` / `RERANKER` | `5` / `lexical` | `cross-encoder` is better and downloads ~270MB |

Full list with comments: `.env.example`.

---

## 4. Measured resource profile

On a Windows laptop, `EMBED_BACKEND=onnx`, `LLM_PROVIDER=extractive`:

| | |
|---|---|
| Cold start to first healthy `/health` | **3.1 s** (includes loading the embedding model) |
| Resident memory, steady state | **312 MB** |
| Warm `/ask`, no LLM call | **10–34 ms** |
| Index on disk | **615 KB** for 14 chunks (~44 KB/chunk) |
| Service install size | 94 packages; the dev set is 120 and adds ~1 GB of torch |

Add your LLM provider's latency on top of `/ask` — that will dominate. The
numbers above are retrieval only.

A 512 MB instance is enough for the service. Embedding a large upload is the
memory spike; the 400-page cap bounds it.

---

## 5. The API

Machine-readable contract: **`openapi.json`** in the repo root, generated from
the running app. Live console at `/docs`. Twelve paths:

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | status, chunk count, queue depth, free disk |
| GET | `/subjects` | none | subject list for a dropdown |
| POST | `/ask` | optional | the main call |
| POST | `/auth/dev-login` | none | local development only; disable in production |
| GET | `/me` | user | quota usage |
| POST | `/uploads` | user | multipart PDF; returns `202` + `Location` |
| GET | `/jobs/{job_id}` | owner of the doc | poll indexing progress |
| GET | `/documents` | user | the caller's own uploads |
| PUT/DELETE | `/documents/{id}/offer` | owner of the doc | offer to the library / withdraw |
| GET | `/moderation/queue` | site owner | what is awaiting review |
| POST | `/moderation/{id}/approve` | site owner | publish it |
| POST | `/moderation/{id}/reject` | site owner | decline it; uploader keeps their copy |

### Asking a question

```http
POST /ask
{ "question": "Explain the difference between 3NF and BCNF",
  "subject": "DBMS",            // optional
  "scopes": ["curated","community"],  // optional; omit for the public tiers
  "exam_mode": true, "marks": 10,     // VTU-style structured answer
  "k": 5 }
```

Response carries `answer`, `not_found`, `scopes` (the tiers actually searched),
`warnings`, and `sources[]` — each with `file`, `page`, `subject`, `visibility`,
`citation`, `text`, `text_truncated`, `rerank_score`.

**Render the citation as a link.** The answer contains inline markers like
`[DBMS_Module2.pdf, p.14]`, and `sources[]` tells you which file and page each
one is. Making those click through to that page of the PDF (`pdf.js`, or
`<iframe src="...#page=14">`) is the single thing that makes this feel finished
rather than like another chatbot.

**`not_found: true`** means the notes do not answer the question. Show the
message as-is; do not fall back to a general-purpose answer, because the whole
promise is that every sentence traces to a page.

### Accepting a contribution

```
POST /uploads (multipart: file, subject)
  → 202 { doc_id, job_id }  + Location: /jobs/{job_id}
  → 200 { duplicate: true } if that account already uploaded the same file
  → 409 already_in_library | 413 too_large | 415 not_a_pdf | 429 quota_* | 503 intake_paused

GET /jobs/{job_id}  every ~1s
  → { state: queued|running|done|failed, stage, progress, message, error_code }

PUT /documents/{doc_id}/offer  { "attested": true }   ← must be an explicit user action
  → the document enters the moderation queue
```

Error bodies on the newer routes are `{"detail": {"code", "message", ...}}`.
Branch on `code`, not on the message text. `/health`, `/subjects` and `/ask`
keep FastAPI's plain `{"detail": "..."}` shape.

Someone else's document returns **404, not 403** — deliberately, so the API
never confirms that an id exists.

---

## 6. Hard constraints

- **One always-on process.** The ingest worker runs in-process and the vector
  index wants a single writer. Do not run multiple replicas against the same
  disk without first moving the index to a shared store.
- **Persistent disk, mounted at `data/`.** It holds the index, the SQLite
  registry and uploaded PDFs. An ephemeral filesystem loses the library on every
  deploy.
- **Rate limiting is in-memory and per-IP.** It does not work across instances.
  Put a real limiter at your edge if you expose this publicly.
- **`/uploads` is slow by design** — it returns `202` and the work happens in the
  background. Never block a request on indexing.
- **CORS must expose `Location`** (already configured) or the browser cannot read
  the job URL off the `202`.

---

## 7. The one thing that will surprise you

`EXTRACTIVE_COMMUNITY` defaults to **`false`**, and with `LLM_PROVIDER=extractive`
that means: an approved contributed document is indexed, public and
retrievable — **and never quoted in an answer.**

The extractive provider is a no-LLM fallback that copies source sentences
verbatim. Letting it quote contributed notes would republish a stranger's words
under your site's own citation format. So those passages are dropped and the
answer carries a warning saying how many.

**Running a public library of contributed notes means running a real LLM
provider.** Set `LLM_PROVIDER=anthropic` or `ollama`. Only set
`EXTRACTIVE_COMMUNITY=true` if you have decided verbatim republication is
acceptable.

---

## 8. Security notes for whoever hosts it

- The API key and the index live only in this service. A browser must never
  call an LLM provider directly.
- `ALLOW_DEV_LOGIN=false` in production. The dev login mints a token for any
  email address — including the owner's.
- `OWNER_EMAIL` decides who can approve contributions. It is compared against a
  verified sign-in, never a header or query parameter. Leave it unset and
  moderation is closed to everyone, which is the safe default.
- Uploaded PDFs and the registry (which holds contributor emails) are
  gitignored. Keep them out of backups that get shared.
- Community passages are served as text plus citation. **Do not add an endpoint
  that serves the raw contributed PDF** — that is the line between quoting
  someone's notes and redistributing their file.
- Text from uploads is sanitized and scanned for prompt injection at ingest;
  flagged pages are never embedded. Passages reach the model inside nonce-fenced
  tags so a document cannot forge a boundary and start issuing instructions.

---

## 9. Known gaps

Honest list of what is not built. None of it blocks a read-only deployment;
all of it matters if you accept contributions at scale.

- No takedown or abuse-report endpoints. Removing a document today is a manual
  call to the registry plus `delete_by_source`.
- No per-day quotas (only per-account totals) and no audit log of moderation
  decisions beyond `reviewed_by` on the row.
- No reconciliation command for the case where the SQLite registry and the
  vector index disagree — the most likely week-two operational surprise.
- The SQLite registry and local blob store are single-instance. Interfaces are
  in place for a Postgres/pgvector and object-storage driver, but those drivers
  are not written.
- Near-duplicate detection is not implemented; exact SHA-256 duplicates are.
- The injection scanner logs, it does not block (`INJECTION_ENFORCE=false`).
  Hard-flagged pages are already excluded from embedding regardless.

---

## 10. Verifying it yourself

```bash
pip install -r requirements.txt   # the full dev set
python -m pytest                   # 163 tests, fully offline
python -m src.evaluate             # retrieval hit rate against known pages
```

The suite needs no network, no API key and no model download — it uses a
deterministic hash embedder. It covers tier isolation, the upload and approval
flow end to end over HTTP, injection handling, and the citation contract.

`python -m src.evaluate` scores retrieval against `eval/questions.json`
(question → the page that should answer it). On the bundled sample corpus it
reports 13/13 at k=5. That number says the pipeline is wired correctly, not that
it is good — the corpus is small and the questions were written against it.
Replace that file with your own once real notes are indexed.
