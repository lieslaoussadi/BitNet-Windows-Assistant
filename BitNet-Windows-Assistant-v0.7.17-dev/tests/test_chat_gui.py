import unittest
from pathlib import Path
from types import SimpleNamespace

from bitnet_windows.chat_gui import (
    ChatAssistantWindow,
    attachment_kind,
    looks_like_supported_file_path,
    normalize_message,
    selected_mode_summary,
    format_diagnostic_report,
    append_retrieved_source_references,
    append_summary_source_references,
    message_display_name,
    normalize_clipboard_text,
)
from bitnet_windows.rag import DocumentChunk, RetrievalHit
from bitnet_windows.engine import EngineConfig


class ChatGuiHelperTests(unittest.TestCase):

    def test_sources_mousewheel_scrolls_only_the_sources_canvas(self):
        calls = []
        fake = SimpleNamespace(sources_canvas=SimpleNamespace(
            yview_scroll=lambda units, what: calls.append((units, what))
        ))
        self.assertEqual(ChatAssistantWindow._scroll_sources_mousewheel(fake, SimpleNamespace(delta=120, num=None)), "break")
        self.assertEqual(ChatAssistantWindow._scroll_sources_mousewheel(fake, SimpleNamespace(delta=-120, num=None)), "break")
        self.assertEqual(ChatAssistantWindow._scroll_sources_mousewheel(fake, SimpleNamespace(delta=0, num=4)), "break")
        self.assertEqual(ChatAssistantWindow._scroll_sources_mousewheel(fake, SimpleNamespace(delta=0, num=5)), "break")
        self.assertEqual(calls, [(-1, "units"), (1, "units"), (-1, "units"), (1, "units")])

    def test_document_poll_marks_modified_source_and_preserves_selection(self):
        import tempfile
        from bitnet_windows.rag import DocumentChunk, RAGIndex

        class FakeListbox:
            def __init__(self):
                self.items = []
                self.selected = (0,)
                self.reselected = []
            def curselection(self):
                return self.selected
            def delete(self, _start, _end):
                self.items.clear()
                self.reselected.clear()
            def insert(self, _index, value):
                self.items.append(value)
            def selection_set(self, index):
                self.reselected.append(index)

        class FakeRoot:
            def __init__(self):
                self.callbacks = []
            def after(self, delay, callback):
                self.callbacks.append((delay, callback))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test_bitnet.txt"
            source.write_text("Le mot secret est VIOLET.", encoding="utf-8")
            chunk = DocumentChunk(str(source.resolve()), source.name, "Lignes 1-1", "Le mot secret est BLEU.", 0)
            fake_list = FakeListbox()
            messages = []
            fake = SimpleNamespace(
                current_tab="documents", busy=False, root=FakeRoot(),
                store=SimpleNamespace(list_stale_rag_files=lambda: [str(source.resolve())]),
                rag_index=RAGIndex(chunks=[chunk], indexed_files=[str(source.resolve())]),
                selected_files=[source], documents_list=fake_list,
                documents_heading=SimpleNamespace(configure=lambda **kwargs: None),
                status_var=SimpleNamespace(set=lambda value: messages.append(value)),
                _last_document_status_snapshot=frozenset(),
            )
            fake._current_stale_document_keys = lambda: frozenset({str(source.resolve()).casefold()})
            fake._refresh_documents_list = lambda: ChatAssistantWindow._refresh_documents_list(fake)
            fake._poll_document_changes = lambda: ChatAssistantWindow._poll_document_changes(fake)
            ChatAssistantWindow._poll_document_changes(fake)

            self.assertEqual(fake_list.reselected, [0])
            self.assertIn("modifié — à réindexer", fake_list.items[0])
            self.assertEqual(fake.rag_index.chunks, [])
            self.assertEqual(fake.rag_index.indexed_files, [])
            self.assertTrue(any("Modification détectée" in message for message in messages))
            self.assertEqual(fake.root.callbacks[0][0], 2000)

    def test_document_poll_does_not_refresh_hidden_documents_tab(self):
        calls = []
        class FakeRoot:
            def after(self, delay, callback):
                calls.append((delay, callback))
        fake = SimpleNamespace(
            current_tab="sources", busy=False, root=FakeRoot(),
            _current_stale_document_keys=lambda: self.fail("should not scan while tab hidden"),
            _refresh_documents_list=lambda: self.fail("should not refresh while tab hidden"),
        )
        fake._poll_document_changes = lambda: ChatAssistantWindow._poll_document_changes(fake)
        ChatAssistantWindow._poll_document_changes(fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 2000)

    def test_auto_start_does_nothing_when_option_is_disabled(self):
        called = []
        fake = SimpleNamespace(
            engine_config=EngineConfig(executable="", model_path="", auto_start=False),
            _run_task=lambda *args, **kwargs: called.append(args),
        )
        ChatAssistantWindow._start_engine_automatically(fake)
        self.assertEqual(called, [])

    def test_auto_start_runs_local_inference_when_option_is_enabled(self):
        called = []
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / ("llama-completion.exe" if __import__("os").name == "nt" else "llama-completion")
            model = Path(directory) / "model.gguf"
            exe.write_bytes(b"test executable")
            model.write_bytes(b"test model")
            config = EngineConfig(executable=str(exe), model_path=str(model), auto_start=True)
            fake = SimpleNamespace(
                engine_config=config,
                _run_task=lambda *args, **kwargs: called.append(args),
            )
            ChatAssistantWindow._start_engine_automatically(fake)
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][0], "Démarrage automatique du moteur BitNet")

    def test_auto_start_reports_invalid_configuration_without_starting_task(self):
        calls = []
        messages = []
        fake = SimpleNamespace(
            engine_config=EngineConfig(executable="", model_path="", auto_start=True),
            engine_test_passed=False,
            _update_engine_status=lambda: calls.append("status"),
            _append_message=lambda *args, **kwargs: messages.append(args),
            status_var=SimpleNamespace(set=lambda value: calls.append(value)),
            _run_task=lambda *args, **kwargs: calls.append("run"),
        )
        ChatAssistantWindow._start_engine_automatically(fake)
        self.assertNotIn("run", calls)
        self.assertTrue(messages)
        self.assertIn("Démarrage automatique ignoré", messages[0][1])

    def test_extractive_answer_is_not_mislabeled_as_bitnet(self):
        self.assertEqual(
            message_display_name("assistant", "Extrait fidèle — RAG local"),
            "Extrait fidèle — RAG local",
        )

    def test_default_assistant_label_is_bitnet_for_generated_answers(self):
        self.assertEqual(message_display_name("assistant"), "BitNet 1.58")

    def test_default_user_label_is_user(self):
        self.assertEqual(message_display_name("user"), "Vous")

    def test_detects_pdf_attachment(self):
        self.assertEqual(attachment_kind(Path("guide.PDF")), "PDF")

    def test_detects_image_attachment_case_insensitive(self):
        self.assertEqual(attachment_kind("Capture.PNG"), "Image")

    def test_detects_markdown_document(self):
        self.assertEqual(attachment_kind("notes.md"), "Texte")

    def test_rejects_unrecognized_attachment_type(self):
        self.assertEqual(attachment_kind("archive.exe"), "Non pris en charge")

    def test_normalizes_user_message_without_destroying_internal_newlines(self):
        self.assertEqual(normalize_message("  Bonjour\nmon ami  "), "Bonjour\nmon ami")

    def test_clipboard_text_normalizes_windows_line_endings(self):
        self.assertEqual(normalize_clipboard_text("alpha\r\nbeta\rgamma\n"), "alpha\nbeta\ngamma\n")

    def test_clipboard_text_preserves_leading_and_trailing_blank_lines(self):
        self.assertEqual(normalize_clipboard_text("\r\nalpha\r\n\r\n"), "\nalpha\n\n")

    def test_clipboard_text_preserves_french_accents_and_code(self):
        self.assertEqual(normalize_clipboard_text("Été — test\nprint(\"OK\")"), "Été — test\nprint(\"OK\")")

    def test_detects_supported_windows_file_path_pasted_as_text(self):
        self.assertTrue(looks_like_supported_file_path(
            "E:/Deepseek Hernes/DOSSIER/05_petites_histoires.txt"
        ))

    def test_detects_quoted_windows_file_path(self):
        self.assertTrue(looks_like_supported_file_path('"C:\\docs\\notes.md"'))

    def test_does_not_treat_regular_text_or_multiple_lines_as_file_path(self):
        self.assertFalse(looks_like_supported_file_path("Voici mon fichier.txt"))
        self.assertFalse(looks_like_supported_file_path("C:\\docs\\a.txt\nC:\\docs\\b.txt"))

    def test_enter_sends_and_shift_enter_allows_newline(self):
        sent = []
        fake_window = SimpleNamespace(send_message=lambda: sent.append(True))
        normal_event = SimpleNamespace(state=0)
        shift_event = SimpleNamespace(state=0x0001)
        self.assertEqual(ChatAssistantWindow._composer_return_shortcut(fake_window, normal_event), "break")
        self.assertEqual(sent, [True])
        self.assertIsNone(ChatAssistantWindow._composer_return_shortcut(fake_window, shift_event))
        self.assertEqual(sent, [True])

    def test_chat_transcript_blocks_typing_but_allows_navigation(self):
        fake_window = object()
        self.assertEqual(
            ChatAssistantWindow._protect_chat_transcript_edit(fake_window, SimpleNamespace(keysym="a", char="a")),
            "break",
        )
        self.assertIsNone(
            ChatAssistantWindow._protect_chat_transcript_edit(fake_window, SimpleNamespace(keysym="Left", char="")),
        )
        self.assertIsNone(
            ChatAssistantWindow._protect_chat_transcript_edit(fake_window, SimpleNamespace(keysym="F4", char="")),
        )
        self.assertEqual(
            ChatAssistantWindow._protect_chat_transcript_edit(fake_window, SimpleNamespace(keysym="Delete", char="")),
            "break",
        )

    def test_blank_message_stays_blank(self):
        self.assertEqual(normalize_message("   \n "), "")

    def test_mode_notice_is_honest_when_rag_and_web_are_selected(self):
        summary = selected_mode_summary(True, True, rag_ready=False)
        self.assertIn("RAG local demandé mais index non prêt", summary)
        self.assertIn("recherche Web activée", summary)

    def test_mode_notice_can_report_ready_local_rag(self):
        summary = selected_mode_summary(True, False, rag_ready=True)
        self.assertIn("RAG local actif", summary)

    def test_mode_notice_defaults_to_local_chat(self):
        summary = selected_mode_summary(False, False)
        self.assertIn("chat local demandé", summary)
        self.assertIn("ne démarre que si sa case est cochée", summary)

    def test_rag_answer_gets_deterministic_source_reference_footer(self):
        chunk = DocumentChunk(
            source_path="C:/docs/05_petites_histoires.pdf",
            filename="05_petites_histoires.pdf",
            locator="Page 1",
            text="Son inventeur prépare des exemples plus variés et mesure les erreurs.",
            chunk_index=0,
        )
        hit = RetrievalHit(source_id="S1", chunk=chunk, score=1.0)
        answer = append_retrieved_source_references(
            "L'inventeur prépare des exemples plus variés et mesure les erreurs.", [hit]
        )
        self.assertIn("L'inventeur prépare des exemples plus variés", answer)
        self.assertIn("Références documentaires retrouvées par le RAG : [S1]", answer)
        self.assertIn("05_petites_histoires.pdf — Page 1", answer)

    def test_rag_answer_references_all_retrieved_sources(self):
        hits = [
            RetrievalHit("S1", DocumentChunk("a", "a.txt", "Lignes 1-2", "alpha", 0), 1.0),
            RetrievalHit("S2", DocumentChunk("b", "b.pdf", "Page 3", "beta", 1), 0.8),
        ]
        answer = append_retrieved_source_references("Réponse.", hits)
        self.assertIn("[S1] a.txt — Lignes 1-2", answer)
        self.assertIn("[S2] b.pdf — Page 3", answer)

    def test_summary_footer_says_sources_were_consulted_not_retrieved_by_rag(self):
        hit = RetrievalHit("S1", DocumentChunk("a", "guide.pdf", "Page 2", "Le texte source.", 0), 1.0)
        answer = append_summary_source_references("Résumé du document.", [hit])
        self.assertIn("Sources consultées pour ce résumé : [S1] guide.pdf — Page 2", answer)
        self.assertNotIn("retrouvées par le RAG", answer)

    def test_summary_footer_leaves_no_source_summary_unchanged(self):
        self.assertEqual(append_summary_source_references("  Résumé.  ", []), "Résumé.")

    def test_non_rag_answer_is_not_modified(self):
        self.assertEqual(append_retrieved_source_references("  Bonjour.  ", []), "Bonjour.")

    def test_empty_answer_still_shows_rag_references(self):
        hit = RetrievalHit("S1", DocumentChunk("a", "a.txt", "Lignes 1-2", "alpha", 0), 1.0)
        self.assertEqual(
            append_retrieved_source_references("  ", [hit]),
            "Références documentaires retrouvées par le RAG : [S1] a.txt — Lignes 1-2",
        )

    def test_format_diagnostic_report_includes_check_and_notice(self):
        report = {
            "app": "BitNet Windows Assistant",
            "mode": "read-only",
            "checks": [{"name": "Python", "status": "found", "detail": "3.13", "path": "C:/Python"}],
            "notice": "Aucun changement système.",
        }
        formatted = format_diagnostic_report(report)
        self.assertIn("[FOUND] Python: 3.13", formatted)
        self.assertIn("C:/Python", formatted)
        self.assertIn("Aucun changement système.", formatted)


if __name__ == "__main__":
    unittest.main()
