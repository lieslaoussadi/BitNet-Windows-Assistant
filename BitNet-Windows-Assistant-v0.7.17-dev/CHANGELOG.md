## v0.6.8-dev
- Added a dedicated Diagnostics result window with a direct return to conversation; quick diagnostic output no longer pollutes the chat transcript.
- Unified button styling around a single teal accent color.
- Kept local prompt, RAG, conversation storage, and clock behavior unchanged.

## 0.6.7-dev
- Unifié le fond de tous les messages en thème clair et sombre.
- Bouton d’envoi compact avec une flèche seule.

## v0.6.7-dev — Conversation uniforme
- Harmonisation des fonds de tous les corps de message dans les thèmes clair et sombre.
- Bouton d’envoi compact avec une flèche seule et indication au survol.

# Changelog

## 0.6.6-dev
- Unifie le fond des messages de conversation par thème pour éviter les blocs contrastés.
- Thème clair : fond de messages uniforme blanc cassé.
- Thème sombre : fond de messages uniforme bleu nuit/ardoise.
- Remplace le libellé du bouton Envoyer par une flèche compacte.
- Préserve le prompt local, l’horloge d’interface, les fonctions RAG et les préférences de thème.

# Changelog

## 0.6.5-dev
- Unified conversation transcript and message-card colors across light and dark themes.
- Dark mode now uses coordinated slate/blue message backgrounds instead of bright white blocks.
- Light mode retains readable, subtle message backgrounds.


## v0.6.4-dev
- General settings dialog is resizable and scrollable, preventing controls from being clipped on smaller displays and high-DPI layouts.
- Improved spacing and prompt editor readability.
- Preserves the existing local prompt, light/dark themes, clock isolation, and RAG behavior.

# Changelog

## v0.6.3-dev
- Added a local light/dark appearance selector in Settings; preference is stored locally.
- Added best-effort recoloring for chat, document and dialog widgets with readable dark surfaces and text.
- Kept the UI clock separate from the model prompt and preserved RAG reference behavior.
- Based on the previously tested v0.6.2-dev package; this is a development build requiring Windows acceptance testing.

# Changelog

## 0.5.9-dev
- Replaced the persistent “Chat + RAG” top-bar heading with an on-demand Help button.
- Shortened the welcome transcript to remove lengthy setup instructions from the conversation area.
- Moved the local document pipeline into the Docs tab so it is visible alongside document-indexing controls.
- Shortened the sidebar label to “RAG”, reduced the brand mark/text size, and slightly reduced chat/composer font sizes.
- Simplified the source-panel guidance; no local data or existing feature settings are deleted.
- 135 automated tests passed; Windows visual acceptance remains pending.

## 0.5.8-dev
- Removed the non-functional Conversations navigation button; saved conversation switching remains available through Sessions.
- Removed the duplicate Souvenirs navigation button; memory management remains available in the right-hand Mém. tab.
- Kept all underlying conversations and memories intact; only duplicate navigation entry points were removed.
- Automated tests and GUI import checked in the development environment; visual acceptance remains on the user's Windows PC.

## 0.5.7-dev
- Expanded general settings window so the recent-chat visibility and saved-session controls are visible without clipping.
- Shortened document panel tabs to Sources / Docs / Mém. to prevent truncation in French.
- Kept the existing DPI-awareness fix and local data behavior.

v0.5.6-dev
- Kept the DPI-awareness fix from v0.5.5-dev.
- Added a persistent setting to show/hide recent chats in the sidebar.
- Added a saved-sessions picker so conversations remain accessible when recent chats are hidden.
- Shortened navigation labels and preserved French/English UI switching.
- No conversation, memory, document index, or source files are deleted by hiding the recent list.


- The manual SQLite backup dialog now opens in the project's `Sauvegardes` folder by default, creating it when writable; users can still choose another location.
- Added a Windows per-monitor DPI-awareness request before creating Tkinter widgets, with a legacy fallback for older Windows versions, to reduce OS bitmap-scaling blur.
- Kept the existing SQLite online-backup and integrity-check behavior unchanged.
- Automated tests cover the default backup-folder location. Real 1920×1080/DPI behavior still requires Windows acceptance testing.

## [0.5.4-dev] - 2026-10-18

- Added a user-selected, consistent SQLite backup using SQLite's online backup API (includes conversations, messages, memories, settings, and RAG index data).
- The backup is written to a temporary file, checked with `PRAGMA integrity_check`, and atomically moved into place only after verification.
- Added an integrity-check action to Settings and a clear backup confirmation/error message in French and English.
- Added offline tests for integrity checks, backup data preservation, and refusal to overwrite the active database.
- Restore is not yet offered in the UI; recovery from a backup should be tested in a separate database before any future restore workflow is added.
- Windows acceptance remains pending.

## [0.5.3-dev] - 2026-10-18

- Removed the “Interface locale • prototype de travail” subtitle from the main header.
- Improved primary button contrast with a deep teal treatment and clearer focus outline.
- Modernized the recent-conversations list styling for better readability and selection visibility.
- Made the local document pipeline information vertically scrollable and removed the obsolete Web-disconnected note.
- Kept the document-library action visible below the scrollable explanation.
- Validation: 131 automated tests passed; Python compilation passed. Windows acceptance remains pending.

## v0.5.2-dev — horloge visible et interface RAG réorganisée

- Affiche la date et l'heure locales Windows au démarrage et les actualise dans la barre supérieure.
- Tente en arrière-plan de vérifier l'accès réseau via un en-tête HTTP Date ; l'absence d'Internet ne bloque pas l'application et l'horloge système n'est jamais modifiée.
- Ajoute le contexte date/heure local aux requêtes du modèle pour interpréter les dates relatives.
- Rend les boutons plus visibles avec une bordure et un état de survol.
- Déplace le choix du mode de réponse RAG dans le panneau « Sources et documents », au-dessus des sources et documents.
- Retire le long texte explicatif sous le compositeur et raccourcit le message d'accueil système.
- Les fonctions de résumé rapide sans modèle et de résumé avec BitNet restent dans l'onglet Documents.

## v0.5.1-dev — date-aware search and stricter weather grounding

- Relative weather questions containing “demain/tomorrow” are expanded with the exact local date and desired forecast fields.
- Web context now explicitly distinguishes current conditions from dated forecasts and prohibits unsupported rain/temperature claims.
- Added offline tests for query rewriting and weather-specific grounding instructions.

# Changelog

## v0.5.0-dev — cleaner Web evidence and duplicate suppression
- Deduplicates search results by normalized URL and removes common tracking parameters.
- Keeps the most useful snippet when duplicate URLs return different snippets.
- Deduplicates displayed references so each source appears only once.
- Tightens the local model's Web-grounding instructions; weather questions receive explicit guidance not to invent temperatures or hourly forecasts.
- Search still uses public HTML pages from DuckDuckGo with Bing fallback; no search API key or remote AI generation is used. Page-body extraction is not included: the model receives search-result snippets, and this limitation is stated explicitly.
- Added offline regression tests for URL normalization, duplicate results and reference lists.


## v0.4.9-dev — Web fallback and Settings localization
- Added a second public HTML search provider as fallback when DuckDuckGo times out, returns a non-200 response (including HTTP 202), or provides no parseable results.
- Improved diagnostics so errors identify each provider and the failure type.
- Translated the general Settings dialog controls and explanatory text when English is selected.
- Added offline tests for the fallback parser and provider failover. Live Internet and Windows acceptance still require user testing.


## v0.4.8-dev — optional Web search and language switch
- Added opt-in public Web search using HTML search results (no search or AI API key).
- Search results are passed as context to the configured local model; URLs are included in the answer.
- Added a French/English language selector in Settings for core navigation and composer controls. Some detailed dialogs/messages remain French in this development build.
- Preserved v0.4.7 navigation cleanup, local SQLite persistence, conversations, memories and lexical RAG.
- OCR/vision remains deferred. Web requests require Internet and are never launched unless the checkbox is enabled.

## v0.4.7-dev — navigation cleanup
- Simplified navigation and added general settings with local database location.
- Removed duplicate top-bar diagnostics/about links.

## 0.6.1-dev — reconstruction de contrôle

- Reconstruit à partir de la base v0.5.9-dev, sans réutiliser l’archive v0.6.0-dev signalée par Microsoft Defender.
- Déplacement du bouton Aide de la barre supérieure vers le bas de la barre latérale.
- Suppression des caches Python précompilés (`__pycache__`, `.pyc`, `.pyo`) du paquet distribué.
- Version de test : cette reconstruction ne bénéficie d’aucune signature Microsoft ni d’une garantie d’absence de détection antivirus.


## 0.6.2-dev
- Ajout d’un prompt système personnalisable, enregistré localement dans les paramètres du moteur.
- L’horloge reste visible dans l’interface mais la date et l’heure ne sont plus injectées dans les messages envoyés au modèle.
- Le prompt encourage des réponses naturelles, concises et honnêtes.
