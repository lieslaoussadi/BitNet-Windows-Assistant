# Architecture notes — v0.4.1 development

## Current runtime

```text
UI entry points
 ├── gui.py       — original diagnostic interface
 └── chat_gui.py  — local chat, persistent conversations, memories, and RAG panel
      ├── engine.py  — local llama.cpp/BitNet CLI adapter and per-user model configuration
      ├── rag.py     — local PDF/text extraction, heading-aware chunks and lexical retrieval
      ├── summarize.py — extractive summaries and optional local BitNet map/reduce summaries
      └── storage.py — SQLite-backed conversations, approved memories, local settings and RAG chunks

Per-user app-data directory
 ├── engine.json             — saved local executable/model settings
 ├── assistant_data.sqlite3  — messages, conversations, memories, settings and RAG chunks
 └── SQLite WAL/SHM files    — transient SQLite journal files when enabled
```

## Persistence semantics

- Conversations and visible messages are saved automatically. Conversation creation, clear/delete, and TXT/JSON export are explicit UI actions. The most recently updated conversation is opened on launch.
- Memories are not created automatically. Users add one manually or from a selected transcript passage, choose a category, and may edit, individually disable, or delete it. A global checkbox prevents all saved memories from being sent into prompts without deleting them.
- Memory prompt selection is lightweight and local. Preferences/rules may apply across queries; other memories are chosen on lexical overlap. Only preferences/rules are eligible as auxiliary context for document-grounded RAG answers so unrelated personal facts do not masquerade as source evidence.
- The RAG index stores extracted chunks and file metadata, not a second copy of the source document. Before chunks are reused after restart, the original source must still exist with the same size and modification timestamp. Changed files appear as needing reindexing. Because validation uses metadata, it is not a cryptographic freshness guarantee.
- Conversation, memory, settings, and index data never require an external service. Web is not implemented and the Web checkbox remains blocked.

## Safety and user-control invariants

- No cloud AI API is called.
- No memory is saved unless the user explicitly adds it.
- A memory can be edited, deactivated without deletion, or deleted individually.
- Deleting/clearing a conversation is explicit and confirmed; starting a new one does not erase prior conversations.
- Original documents are read only after the user asks to index them, and source files are never modified. Persistent extracted chunks remain local in SQLite.
- UI persistence failures are surfaced as status messages rather than hidden as successful saves.
- Preserve the v0.1 reference and all prior candidate archives; every new candidate is a separate archive.

## Planned components

1. Web search opt-in with visible source URLs, bounded fetching, and no automatic upload of local documents.
2. OCR for scanned PDFs and supported image formats, then separately evaluated local vision models for image understanding.
3. Optional local embedding model plus vector/HNSW retrieval, benchmarked against the current lexical scorer before becoming available.
4. Streaming/cancellable generation and possibly a persistent model process, contingent on compatibility with the tested local runtime.
5. Packaging and validation on the target Windows machine before a public release.
