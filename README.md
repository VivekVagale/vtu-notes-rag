# VTU Notes RAG

**By [Vivek Vagale](https://github.com/VivekVagale)**

> Integrating this into a site? Start with **[INTEGRATION.md](INTEGRATION.md)** —
> what to deploy, the API contract, measured resource usage, and the known gaps.

Ask questions about your own VTU syllabus and notes PDFs and get answers that
are **grounded in those PDFs only**, with a `[file.pdf, p.14]` citation after
every claim. If the notes do not contain the answer, the bot says
`Not found in your notes` instead of inventing one.

- Local embeddings (`BAAI/bge-small-en-v1.5`) - indexing costs nothing, and
  `EMBED_BACKEND=onnx` runs them through onnxruntime with no torch at all
- ChromaDB persisted to disk, hash check so re-ingest skips unchanged files
- Page-level citations, subject filter, VTU exam-mode answers (5/10 marks)
- Swappable LLM: Anthropic API, local Ollama, or a no-LLM extractive mode
- CLI + Streamlit UI + a retrieval eval script

## What runs where

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

A reader needs nothing but a browser - no model, no install, no API key.
Embedding and search run on the server for free, and uploading a PDF never
touches an LLM at all. The only thing that reaches outside is writing the final
prose, and even that is optional: with `LLM_PROVIDER=extractive` the right pages
and citations still come back, just stitched from the notes' own sentences
instead of written. "Local Ollama" means local to *the server*, not to the
reader.

---

## 1. Setup

Python 3.11 or newer.

```bash
cd vtu-rag
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy .env.example .env          # cp .env.example .env on macOS/Linux
```

The first ingest downloads the embedding model (~130 MB) once, then works offline.

## 2. Put your notes in

One folder per subject - the folder name becomes the subject tag used by the
filter and the dropdown:

```
data/pdfs/
  DBMS/
    DBMS_Module2.pdf
    DBMS_Module3.pdf
  OS/
    OS_Module3.pdf
```

No PDFs yet? Generate the sample VTU notes used by the demo and the eval set:

```bash
python scripts/make_sample_pdfs.py
```

Scanned PDFs with no text layer are reported and skipped - run OCR on them first.

## 3. Index

```bash
python -m src.cli ingest
```

Every file is hashed (SHA-256). Unchanged files are skipped, changed files have
their old chunks deleted and re-indexed, deleted files are dropped from the
index. `--force` re-indexes everything, `--subject DBMS` limits the run.

## 4. Ask

```bash
python -m src.cli ask "What are the four necessary conditions for deadlock?"
python -m src.cli ask "Difference between 3NF and BCNF" --subject DBMS
python -m src.cli ask "Explain the Banker algorithm" --exam --marks 10
python -m src.cli ask "What is thrashing?" --json
```

Other commands: `python -m src.cli subjects`, `python -m src.cli status`,
`python -m src.cli eval`.

## 5. Streamlit UI

```bash
streamlit run src/app.py
```

Subject dropdown, k slider, reranker choice, exam-mode toggle with 5/10 marks,
a re-index button, the answer, and one expander per source showing the file,
the page and the full chunk text.

## 6. Library tiers

Every chunk carries a tier, so the same index can hold your own notes and other
people's without the two mixing:

| Tier | What it is | Who can retrieve it |
|---|---|---|
| `curated` | your own vetted notes, ingested from `data/pdfs/` | everyone |
| `private` | an uploader's own file | only that uploader |
| `community` | a contributed file the owner has approved | everyone, text + citation only |

`DEFAULT_VISIBILITY` (default `curated`) is the tier stamped at ingest. Retrieval
takes a `scopes` filter; omitting it searches the public tiers and never private
material, and the `private` tier is reachable only through a positive owner
predicate - never a negative one, because a `$ne` filter matches rows that lack
the key at all. Contributed chunks rank at 0.94x so they never outrank vetted
notes on a tie, and an unrecognised tier scores 0.0: a filter mistake degrades to
"never wins" rather than "serves anyway".

### How a contributed file travels

```
upload ──► private (only the uploader can retrieve it)
             │  uploader offers it, with a rights attestation
             ▼
          pending ──► you approve ──► community (everyone, text + citation only)
             │                         └─ withdraw or reject sends it back to private
             └─ a page tripping a hard injection rule is never embedded at all,
                and its document lands in a tier no scope allowlist contains
```

Uploads are capped (25MB, 400 pages, per-account quotas), deduplicated by
SHA-256, and extracted in a separate killable process. One answer may lean on at
most `COMMUNITY_CHUNKS_PER_ANSWER` contributed chunks, and only
`COMMUNITY_SNIPPET_CHARS` of each ever reaches a browser.

**One consequence worth knowing before you run a public library.**
`EXTRACTIVE_COMMUNITY` defaults to `false`. The `extractive` provider copies
source sentences verbatim, so with contributed notes it would republish a
stranger's words under your own citation format. Those passages are dropped from
the answer instead, with a warning. Serving a community library usefully means
running a real LLM provider.

## 7. HTTP service (what a website calls)

```bash
uvicorn src.api:app --port 8000
```

It is the only process that touches the index, the uploads or an LLM key - a
browser never does. Per-IP rate limiting, CORS from `CORS_ORIGINS`, request
validation and intake valves on queue depth and free disk are on by default.

| | |
|---|---|
| `GET /health` `GET /subjects` `POST /ask` | ask questions, anonymous or signed in |
| `POST /auth/dev-login` `GET /me` | local sign-in (disable with `ALLOW_DEV_LOGIN=false`) |
| `POST /uploads` `GET /jobs/{id}` `GET /documents` | contribute a PDF and watch it index |
| `PUT`/`DELETE /documents/{id}/offer` | offer it to the library, or take the offer back |
| `GET /moderation/queue` `POST /moderation/{id}/approve` `.../reject` | yours alone, via `OWNER_EMAIL` |

Try the whole flow locally:

```bash
OWNER_EMAIL=you@example.com uvicorn src.api:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is Belady anomaly?\"}"
```

## 8. Choosing the LLM (.env)

| `LLM_PROVIDER` | needs | notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | set `ANTHROPIC_MODEL`, default `claude-sonnet-5` |
| `ollama` | `ollama serve` running | set `OLLAMA_MODEL`, default `llama3.1:8b` |
| `extractive` | nothing | no LLM: the answer is stitched from your own sentences, each with its citation |

`extractive` is the default in `.env.example` so the project runs with zero
setup. It is a fallback, not a writer - switch to `anthropic` or `ollama` for
answers in real prose. Override per run with
`python -m src.cli --provider ollama ask "..."`.

## 9. Evaluation

`eval/questions.json` holds question / expected-page pairs. The script reports
how often the expected page comes back in the top-k, plus MRR:

```bash
python -m src.evaluate --k 5
```

On the generated sample corpus (14 pages, 2 subjects) the current settings score
**13/13 = 100% hit rate @5, MRR 1.000**. That number says the pipeline is wired
correctly - it is not a benchmark, because the corpus is tiny and the questions
were written against it. Replace the file with your own questions and pages once
your real notes are indexed.

## 10. Tests

```bash
python -m pytest
```

163 tests, fully offline: they use a deterministic hash embedder
(`EMBED_BACKEND=hash`) and the extractive provider, so nothing is downloaded and no
API is called. Beyond chunking, extraction, ingest and retrieval they pin the
things that would hurt if they broke:

- one uploader's private chunks are unreachable by anyone else, and an empty
  scope returns nothing rather than everything
- somebody else's document answers 404, never 403
- approval that publishes zero chunks is a failure, not a success
- a page carrying an injection payload is never embedded
- a contributed document cannot forge a passage boundary in the prompt
- contact details in contributed notes never reach an answer
- the injection scanner raises zero hard flags across the real curated corpus

## 11. How it works

```
data/pdfs/<Subject>/<file>.pdf
   |
   |  pdf_loader.py   PyMuPDF text per page, hyphenation and whitespace cleanup
   |  chunking.py     ~500 tokens, ~80 overlap, using the bge tokenizer itself
   |  embeddings.py   bge-small-en-v1.5, normalised vectors (CPU)
   v
Chroma collection (data/index/) - every chunk carries
   file, source, subject, page, chunk_index, file_hash, chars
   |
   |  retrieve.py     fetch_k candidates, subject filter
   |  rerank.py       lexical blend (default) or cross-encoder, max 2 chunks/page
   |  prompts.py      grounding rules + exam format
   |  llm.py          anthropic | ollama | extractive
   v
Answer + inline [file.pdf, p.N] citations + the source chunks
```

`answer.py` re-checks every citation the model emits against the passages that
were actually retrieved and reports a warning for anything invented.

### Layout

```
src/        config, pdf_loader, chunking, embeddings, store, manifest,
            ingest, retrieve, rerank, prompts, llm, extractive, answer,
            cli, app (Streamlit), evaluate
scripts/    make_sample_pdfs.py
eval/       questions.json
tests/      33 offline tests
data/pdfs/  your notes, one folder per subject
data/index/ Chroma database + manifest.json (gitignored)
```

### Tuning

Everything is an env var (see `.env.example`): `CHUNK_TOKENS`, `CHUNK_OVERLAP`,
`TOP_K`, `FETCH_K`, `RERANKER` (`lexical` / `cross-encoder` / `none`),
`EMBED_MODEL`, `EMBED_BACKEND`, `DEFAULT_VISIBILITY`, `LLM_MAX_TOKENS`,
`LLM_TEMPERATURE`.

`EMBED_BACKEND` picks how the embedding model runs. `sentence-transformers` is the
default and needs torch. `onnx` produces byte-identical vectors through onnxruntime
with no torch installed and a much faster cold start, which is what a small server
should use. `hash` is an offline stub for tests.

Slide-style notes with a few words per page do better with
`CHUNK_TOKENS=250 CHUNK_OVERLAP=40`. Dense textbook scans do better with the
cross-encoder reranker (`RERANKER=cross-encoder`, downloads ~270 MB once).

## Notes and limits

- Runs entirely on your own machine. There is no hosted instance: clone it,
  install it, and it serves on your own `localhost:8501` against your own PDFs.
- Only text-layer PDFs are read. Scanned images need OCR first.
- Diagrams, tables and equations come out as plain text; the citation still
  points at the right page, so keep the PDF open next to the answer.
- The index lives in `data/index/`. Delete that folder to rebuild from scratch.
- Your PDFs and the index are gitignored - nobody else's notes ever leave their
  laptop, and yours never leave yours.

---

## Author

**Vivek Vagale** - [@VivekVagale](https://github.com/VivekVagale)

Designed, written and tested by me. Issues and pull requests are welcome.

## Credits

Built on these open-source projects:

| Project | Used for |
|---|---|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | per-page PDF text extraction |
| [sentence-transformers](https://www.sbert.net/) | local embedding pipeline |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | the embedding model, and its tokenizer for chunking |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | optional cross-encoder reranking |
| [ChromaDB](https://www.trychroma.com/) | persistent vector store |
| [Streamlit](https://streamlit.io/) | the web UI |
| [Anthropic API](https://docs.anthropic.com/) / [Ollama](https://ollama.com/) | swappable answer generation |
| [pytest](https://pytest.org/) | the test suite |

Sample notes content in `scripts/make_sample_pdfs.py` was written for this repo
as demo material and follows the general VTU syllabus outline for DBMS and
Operating Systems. It is not copied from any textbook or university document.

## License

MIT - see [LICENSE](LICENSE).
