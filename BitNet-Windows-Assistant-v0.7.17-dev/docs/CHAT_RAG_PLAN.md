# Chat + RAG plan — style C

## Implemented through v0.4.2-dev

- Style-C Tkinter chat window with ChatGPT AI-editor attribution.
- Local llama.cpp/BitNet CLI configuration and inference adapter, including prompt-only memory context. An opt-in startup inference check is available in `Moteur…`; it is unchecked by default and persists in the local engine configuration. The current completion executable remains one-shot and does not stay loaded.
- Transcript selection with explicit high-contrast selection foreground/background, read-only protection, clipboard actions, and Enter-to-send / Shift+Enter-newline behavior.
- SQLite persistent conversations and message history, a recent-conversation list, create-new without clearing older chats, confirmed current-chat clear, per-conversation permanent delete, and TXT/JSON export.
- User-controlled persistent memories stored in SQLite: create manually or from a selected transcript passage, edit, individually enable/disable, delete, and globally disable prompt use. No automatic memory creation.
- Memory selection uses category/lexical relevance and a small context budget. Preference/rule memories may be considered across topics; other memories require word overlap. Document-grounded RAG prompts only receive preference/rule memories, separated from document evidence.
- Explicit document selection followed by an **Indexer la sélection** action.
- Local text extraction from PDF pages and supported text formats: TXT, Markdown, RST, CSV, HTML, HTM, LOG and JSON.
- Heading/section-aware chunks with provenance: document name and PDF page or text line range; chunks do not cross short standalone headings.
- Lightweight local lexical BM25-style retrieval. The number of source passages is bounded by the configured context size.
- SQLite-persisted RAG index containing extracted chunks and source metadata. PDF passage IDs remain unique across every page, and repeated persistence of a multi-page PDF is covered by regression tests. Existing chunks are reused only when source files still exist and their byte size and modification timestamp are unchanged. Changed files are flagged for reindexing; missing files are skipped.
- RAG passages added to the local model prompt with stronger directions not to mix named stories/sections or add unsupported details; matching source cards appear in the source panel. No matching passage means no model call in RAG mode.
- Bundled pypdf 5.9.0 with third-party license notice.
- Fast extractive summary from original sentences and optional local BitNet map/reduce summary with a preflight confirmation, progress status and source references.
- Conservative guard that rejects empty or obviously truncated model summaries (for example a result ending with the dangling phrase “avec du”).

## Not implemented yet

- OCR for scanned PDFs or images; selected images are skipped with a warning.
- Local image understanding/vision model.
- Internet/Web search; the checkbox remains blocked and does not trigger any network request.
- Optional local embeddings/vector search/HNSW. The current lexical retriever remains the default.
- Streaming/cancellation of active generation and a long-lived model process; the CLI reloads the model for each request.
- A final installer or single-file EXE distribution.

## Next milestones

1. Validate v0.4.2-dev on the target Windows PC: confirm automatic engine checking stays disabled by default, then enable/save it and confirm a local control inference runs after reopening. Also confirm the compact project-editor footer is visible, re-index the same multi-page PDF twice, restart, and verify conversation persistence, memory controls, index restoration and stale-source detection.
2. Test memory context with the actual local BitNet model, both enabled and disabled, and refine only from observed behavior.
3. Add index removal/rebuild controls per individual document, while preserving other documents' valid chunks.
4. Add opt-in Web search with visible URLs, request limits, and no automatic upload of private documents.
5. Add OCR for scanned PDFs/images only after evaluating a tested local package and its resource requirements; then evaluate separate local vision-capable models.
6. Benchmark optional local embeddings/HNSW against the lexical retriever on real document/question sets before deciding whether to enable it.
7. Add streaming/cancellation and consider a persistent model process if the local CLI/runtime supports it reliably.
8. Prepare release packaging only after the candidate passes automated and Windows acceptance tests. Never overwrite prior stable/reference artifacts.

## Testing notes

- `tests/fixtures/rag-sample.pdf` verifies pypdf extraction; `tests/fixtures/rag-stories-sample.pdf` guards against combining adjacent story sections.
- UI source cards cite actual chunks retrieved for that query, not prefilled sample content.
- `docs/ui-style-c-concept.png` is illustrative, not live telemetry.
- SQLite writes run only to the user-local app-data database (or an isolated test database when `BITNET_ASSISTANT_DATA_DIR` is set); no network access is required for storage.
