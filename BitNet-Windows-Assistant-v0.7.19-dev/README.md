# BitNet Windows Assistant — v0.5.1-dev

Windows-first, open-source prototype for chatting with a locally configured Microsoft BitNet/llama.cpp CLI. This development snapshot adds targeted document reindexing and safe index deletion on top of persistent local conversations, user-controlled persistent memories, a persistent lexical RAG index and high-contrast free text selection. It is not a final installer or public release.

## What works in this milestone

- Read-only environment diagnostics (`doctor`, `doctor --json`) and explicit temporary C++ smoke test (`doctor --build-check --json`).
- Explicit local library initialization after the user chooses and confirms a directory.
- Style-C Tkinter chat UI with the project credit **“Éditeur IA du projet : ChatGPT”**.
- Local chat through a user-selected `llama-completion.exe` (recommended for one-shot prompts) or compatible `llama-cli.exe`, plus a local GGUF model. No cloud AI endpoint is called.
- Per-user engine settings saved only after an explicit save/test action.
- Output cleanup for visible termination markers such as `[end of text]`.
- **Local RAG v1:** after the user clicks **Indexer la sélection**, the app reads selected PDF/text files locally, splits content at short headings/section labels and uses a lightweight lexical BM25-style scorer to find passages related to a question. The source panel shows the full retrieved passage and its filename/page or line locator. Generated RAG answers receive a deterministic footer listing the retrieved source IDs and locations even if the model omits citations.
- **Two RAG answer modes:** `BitNet (génératif)` sends retrieved evidence to the local model for a written response. `Extrait fidèle (sans reformulation)` selects one verbatim sentence with lexical overlap and shows its exact source reference; this mode does not invoke the model and is designed for simple fact lookups, not summaries or reasoning. If no sentence matches the query terms, the app reports that rather than fabricating a response. Generative RAG requests omit prior chat turns to avoid contamination from earlier mistakes. Model prompt adherence still requires testing on the target PC.
- **Two document-summary modes:** `Résumé rapide (sans modèle)` ranks and reproduces original sentences from one selected, indexed document; no model is called, so the wording is source-faithful but may read as a collection of excerpts rather than a fluent abstract. `Résumer avec BitNet` produces a local model-written summary, using one generation for a small document or bounded per-section summaries plus final synthesis for larger documents. Both modes show source references in the side panel and answer footer.
- **Clipboard usability:** drag-select any passage in the conversation, double-click a word using standard text selection, copy the last answer or the full transcript, and use a mouse context menu plus Ctrl+C/Ctrl+V/Ctrl+X in the message composer. The transcript remains read-only while supporting selection. `Enter` sends; `Shift+Enter` adds a line break. Pasting a supported absolute Windows file path displays a hint to attach and index the file instead of treating the path as file contents.
- **Optional engine startup check:** the setting in `Moteur…` is unchecked by default. When saved as enabled, the app runs a local control inference after opening and reports success or failure. With the current one-shot `llama-completion` connector, this does not keep the model loaded between messages.
- **Persistent conversations:** every visible message is automatically saved locally. The latest conversation is restored when the app starts; recent conversations can be selected from the sidebar. `Nouvelle conversation` starts another conversation without deleting the old one. `Effacer` clears the current conversation after confirmation, and a right-click on a recent conversation allows permanent deletion. TXT/JSON export remains available.
- **Controllable persistent memories:** the **Souvenirs** panel lets the user explicitly add memories, edit them, enable/disable them or delete them. A selected passage can be saved via right-click → **Créer un souvenir depuis la sélection…**. Nothing is remembered automatically. A separate checkbox controls whether enabled memories are supplied to the local model. Approved preferences/rules can be included across topics; other memories are selected by lightweight lexical relevance, with a small context limit. For source-grounded RAG responses, only preference/rule memories may be added, so personal factual memories are not presented as document evidence.
- **Persistent local RAG index:** indexed passage text and source metadata are stored in SQLite in the user's local application-data folder. On restart, the app checks each source file's size and modification timestamp before reusing its chunks. Missing or changed files are not silently used as current evidence; changed files remain selectable for reindexing. Original documents remain at their original locations and are never modified.
- **Summary truncation guard:** local model summaries that end with a clearly dangling French connector (for example “avec du”) are rejected with an actionable message instead of silently being presented as complete. This conservative check does not detect every possible semantic or grammatical error.
- Formats indexed: PDF with extractable text; `.txt`, `.md`, `.rst`, `.csv`, `.html`, `.htm`, `.log` and `.json`.
- pypdf 5.9.0 is bundled in `src/pypdf` for PDF text extraction, with its BSD-style license included in `third_party/PYPDF-LICENSE.txt` and noted in `THIRD_PARTY_NOTICES.md`. No separate pip installation is required for this PDF feature.

## Still not implemented
- Optional vector/semantic retrieval using local embeddings and HNSW; the existing lexical BM25-style retrieval remains the default and is not replaced.

- OCR for scanned PDFs or images; image files can be selected but are skipped by indexing with a clear warning.
- Semantic/vector embeddings. Retrieval is lexical and lightweight, not a semantic search engine. Matching quality must be evaluated against real documents.
- Streaming/cancellation of active generation and a long-lived model process. The CLI currently starts again for each request and reloads the model.
- Internet/Web search. Its checkbox is intentionally blocked while unsupported; no Web request is run.
- A final installer, a single-file EXE distribution, or validation on other Windows computers.

## Requirements

- Python 3.10 or newer with Tkinter.
- A compatible local llama.cpp/BitNet CLI executable and an existing `.gguf` model for chat.
- The RAG runtime uses only Python standard-library code plus the bundled pypdf 5.9.0 package.
- The existing CMake, Clang, Git and Visual Studio detection features do not guarantee that the full BitNet source will build.

## Run from PowerShell

From the extracted project folder:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m bitnet_windows doctor
python -m bitnet_windows doctor --json
python -m bitnet_windows doctor --build-check --json
python -m bitnet_windows.chat_gui
```

Or double-click `Lancer-Chat-BitNet-Assistant.bat`.

## Configure the local model

1. Open **Moteur…**.
2. Select the local `llama-completion.exe` (recommended) or compatible `llama-cli.exe` and the model `.gguf`.
3. Keep the `llama-bpe` override enabled only for the BitNet tokenizer/model combination for which it was tested.
4. Click **Enregistrer et tester**. This runs a short local inference probe.
5. Send a normal chat message after the probe succeeds.

The adapter uses the tested `User:` / `Assistant:` / `<|eot_id|>` prompt convention and the `-no-cnv` mode for the completion executable. It deliberately avoids `--no-show-timings`, which the tested `llama-completion` rejected. Other CLI builds and other models may require different options or chat templates.

## Use local RAG

1. Click **＋ Joindre PDF / image / texte** and select PDF or text documents.
2. Open the **Documents** tab on the right and click **Indexer la sélection**. File contents are not read before this explicit action.
3. Review the indexing report, including any warnings. A scanned PDF may have no extractable text; image OCR is not included in this milestone.
4. Enable **Utiliser le RAG** and ask a question about the indexed content.
5. The app uses heading-aware chunks and lexical matching. The source panel shows the full retrieved passage. Choose `BitNet (génératif)` to let the model write an answer from the retrieved evidence. Choose `Extrait fidèle (sans reformulation)` for a direct, verbatim sentence from the document; the app adds its exact source locator without calling the model. This second mode is deliberately extractive and is not a summary or semantic reasoner. If no sentence matches the question terms, the app reports that rather than inventing one. In generated RAG mode, prior chat turns are omitted to reduce contamination from earlier mistakes; source IDs, filenames and page/line locations are always appended by the application.

Source text is treated as untrusted evidence rather than instructions. All extraction, chunking and retrieval run locally. The app does not upload or modify selected source files. The index's extracted passage text and file metadata are saved in the local SQLite database so they can be reused on later launches. File sizes are limited to 50 MiB per item and the index has bounded document/chunk counts.

## Conversations and memories

- Recent conversations are listed in the left sidebar and are restored from the local database on launch. The app opens the most recently updated conversation by default.
- **Nouvelle conversation** preserves existing conversations and creates a new one. **Effacer** removes all messages from the currently open conversation after a confirmation. To remove one old conversation and its messages, right-click its row in the sidebar and choose **Supprimer définitivement cette conversation**.
- Open the **Souvenirs** tab in the right panel to add a memory manually, or select a passage in the chat and use right-click → **Créer un souvenir depuis la sélection…**. Memories are never created automatically.
- Each memory can be edited, individually enabled/disabled or deleted. The checkbox **Utiliser les souvenirs actifs dans le chat** disables prompt use without deleting stored memories. Mark preferences and rules with the corresponding category if they should be considered across topics; other memories are injected only when lexical terms match the question.
- The current version does not automatically suggest new memories for approval. Only explicitly added memories enter the memory database.

## Where local data is saved

The app stores its SQLite database at `%LOCALAPPDATA%\BitNetWindowsAssistant\assistant_data.sqlite3` on Windows. It contains conversation messages, approved memories, local settings and extracted RAG chunks/metadata. This is separate from the project folder and the original documents. Keep a backup of this database if its contents matter to you. The Settings window can create a consistent SQLite backup to a user-selected file and verify the active database with SQLite integrity checks. The backup includes conversations, memories, settings, and RAG index metadata/chunks; original source documents are not copied. A one-click restore workflow is intentionally not offered yet; validate a backup by opening it as a separate database before any future restoration.

The index validity check uses file existence, modification timestamp and byte size. If a file changes and retains exactly the same size and timestamp, that change may not be detected by this first implementation; reindex after editing important source files.

## Résumer un document

1. Add the PDF with selectable text or a supported text file and click **Indexer la sélection**.
2. Open the **Documents** tab, select one indexed file in the list, then choose **Résumé rapide (sans modèle)** or **Résumer avec BitNet…**.
3. The quick mode returns up to eight original sentences chosen by a lightweight local scoring heuristic. It is fast and makes no factual paraphrases, but the result can feel like selected excerpts rather than a polished summary.
4. The BitNet mode asks the configured local model to summarize the indexed content. It handles longer documents in sections, then combines the partial summaries. Before starting, the app shows the estimated number of launches and asks for confirmation.

**Resource and quality limits:** the current CLI starts and reloads the model for each generation, so a multi-section summary can take several minutes on modest hardware. The first implementation allows at most eight section groups plus a final synthesis (up to nine generations); if a document exceeds that bound, split it or use the quick mode. The generated summary may omit or misstate details, so review the page/line references in **Sources utilisées**. Neither summary mode sends files to a remote service or modifies the source files.

## Privacy and safety notes

- Local chat sends the prompt to the selected local executable, not to a cloud AI service. The adapter uses a temporary UTF-8 prompt file and attempts to remove it after generation.
- Conversations, memories and extracted RAG chunks are stored locally in the SQLite database. Memory use can be disabled without deleting memories. No memory is added automatically.
- Web is not connected in v0.4.0-dev. Selecting that checkbox blocks sending rather than silently skipping the search.
- Images and scanned-PDF pages are not OCR'd in this version.
- Use a deliberate file selection and keep backups. This prototype is not a security boundary.

## Response labels

Generated answers are labelled `BitNet 1.58`. Verbatim extractive answers are labelled `Extrait fidèle — RAG local`, because no model generation is used. If lexical retrieval finds no suitable passage, the result is labelled `RAG local — aucun passage trouvé`. The dropdown is explicitly named `Mode pour la prochaine réponse RAG`; switching it affects the next submitted message and does not rename earlier messages. Document-summary actions use separate explicit buttons in the Documents tab.

## Test suite

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

The 113-test suite includes the local engine adapter and opt-in automatic startup check (default off), settings persistence, output cleanup, document extraction/retrieval, persistent conversations and memories, SQLite RAG-index restoration, multi-page PDF passage numbering, repeated index saves and stale-source checks, memory-aware prompts, focused evidence for direct action questions, exact-sentence extraction mode, RAG-history isolation, deterministic source-reference footers, clipboard text normalization, extractive and BitNet summary workflows, incomplete-summary guard checks, bounded context/map-reduce tests, path validation and the original library/build-check tests. Automated prompt tests do not guarantee that every local model will obey the answer instruction.

## Upstream references

- Official BitNet repository: https://github.com/microsoft/BitNet
- Official BitNet README: https://github.com/microsoft/BitNet/blob/main/README.md
- llama.cpp CLI options at the tested revision: https://github.com/ggml-org/llama.cpp/blob/390c307752ab78fd8189f359d6954c9ba1be74af/tools/cli/README.md
- pypdf documentation: https://pypdf.readthedocs.io/

## Signature et attribution IA

**Éditeur IA du projet : ChatGPT**

ChatGPT a contribué à la conception technique, à la rédaction et au développement assisté du prototype. La direction du projet, les choix finaux, les tests sur machine, la publication et la maintenance restent sous la responsabilité du mainteneur humain. Cette attribution ne signifie pas qu'OpenAI détient ou publie légalement le projet.

## Licensing

Original project files are licensed under the included MIT license. The bundled pypdf library is distributed under its own BSD-style license; see `third_party/PYPDF-LICENSE.txt` and `THIRD_PARTY_NOTICES.md`. These notices do not change the licenses applying to BitNet, model weights or other upstream components.


## Gestion documentaire ciblée (v0.4.5-dev)

- La bibliothèque vérifie la taille et l’horodatage des fichiers indexés avant une recherche RAG. Un document modifié est marqué comme devant être réindexé et ses anciens passages ne sont pas utilisés comme preuve actuelle.
- L’indexation met à jour uniquement les documents traités avec succès et conserve les index des autres fichiers.
- « Supprimer l’index » supprime les passages locaux du document sélectionné après confirmation. Le fichier source, les conversations et les autres index restent intacts.
- Les sources restent sur leur emplacement d’origine ; aucune API distante n’est utilisée.


## v0.4.5-dev — document freshness feedback

- While the **Documents** tab is visible, the app checks source file metadata every two seconds and marks changed/missing indexed files for reindexing.
- Switching back to **Documents** immediately refreshes file freshness status. Stale passages are excluded from active RAG retrieval until the source is reindexed.
- Refreshing the document list preserves the user's current selection.
- This metadata check compares file size and modification timestamp; it is not a content hash and cannot detect every possible same-size/same-timestamp change.
- Real Windows acceptance is still required for visual timing and file-editor behavior.


## v0.4.7-dev — navigation cleanup

A source document that disappears from disk remains listed after restart with the status “source introuvable — index conservé”. Its saved SQLite record is preserved, while its passages are excluded from retrieval. Restore the unchanged file to its original path to reuse the index, or explicitly delete the document index from the UI. This is a development candidate and still requires real Windows acceptance.


## v0.4.8-dev — Web search and English interface

- Web search is optional: enable **Recherche Internet / Web search** before sending a question. It fetches public search-result HTML and sends only result titles, URLs and snippets to the configured local model. No remote AI API is used. An Internet connection is required, and search providers may block automated requests.
- Results include source URLs in the response; search snippets are not equivalent to reading every linked page. Verify important claims by opening the sources.
- Settings now includes a French/English selector for core navigation and composer controls. This is an initial localization pass; some detailed dialogs and status messages may remain French.
- OCR for scanned PDFs and images is still not implemented.


## v0.5.1-dev — Web result quality

Search results and source references are deduplicated by normalized URL, including common tracking parameters. When duplicate URLs provide different snippets, the more informative snippet is retained. The local model is instructed to avoid inventing facts and to acknowledge missing weather details. The application still sends search-result snippets rather than downloading full page contents; live search availability depends on the network and public providers.


### v0.5.5-dev notes

Manual backups default to the project's `Sauvegardes` directory when it is writable; the save dialog still allows another destination. On Windows, the app requests per-monitor DPI awareness before creating the Tkinter window to reduce blur caused by OS bitmap scaling. This must be confirmed on the target Windows display settings. Keeping backups beside the app is convenient but is not a substitute for a copy on another drive.
## Interface nouvelle (v0.7.18)
- case Plan à côté de Recherche Internet dans les contrôles du chat
- mode plan : agent de planification local, OFF par défaut
- base documentaire par défaut (15 passages) incluse
- lancement sans pop CMD cachée
- rappel textuel redondant retiré des paramètres
