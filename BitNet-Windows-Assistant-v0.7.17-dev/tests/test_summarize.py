import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bitnet_windows.engine import EngineConfig, EngineError
from bitnet_windows.rag import DocumentChunk, index_documents
from bitnet_windows.summarize import (
    MAX_BITNET_SUMMARY_BATCHES,
    SummaryError,
    build_extractive_summary,
    estimate_bitnet_summary_calls,
    summarize_with_bitnet,
    _validated_generated_summary,
)

FIXTURES = Path(__file__).parent / "fixtures"


class DocumentSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exe = self.root / ("llama-completion.exe" if __import__("os").name == "nt" else "llama-completion")
        self.model = self.root / "model.gguf"
        self.exe.write_bytes(b"mock executable")
        self.model.write_bytes(b"mock model")
        self.config = EngineConfig(
            executable=str(self.exe), model_path=str(self.model), context_size=1024,
            threads=4, gpu_layers=0, max_tokens=16, temperature=0.2, timeout_seconds=30,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_summary_guard_accepts_complete_sentence(self):
        self.assertEqual(_validated_generated_summary("Le robot conserve les résultats dans un cahier.", stage="la partie 1/1"), "Le robot conserve les résultats dans un cahier.")

    def test_summary_guard_accepts_source_citations_after_sentence(self):
        self.assertEqual(_validated_generated_summary("Le robot conserve les résultats dans un cahier. [S1]", stage="la partie 1/1"), "Le robot conserve les résultats dans un cahier. [S1]")

    def test_summary_guard_rejects_obviously_cut_off_french_ending(self):
        with self.assertRaisesRegex(SummaryError, "semble interrompu"):
            _validated_generated_summary("La classe a construit un pont avec du", stage="la synthèse finale")

    def test_summary_guard_rejects_empty_output(self):
        with self.assertRaisesRegex(SummaryError, "aucun texte exploitable"):
            _validated_generated_summary("   ", stage="la partie 1/2")

    def test_extractive_summary_uses_verbatim_sentences_and_sources(self):
        index = index_documents([FIXTURES / "rag-stories-sample.pdf"])
        chunks = [chunk for chunk in index.chunks if "robot prudent" in chunk.text.casefold()]
        result = build_extractive_summary(chunks, max_sentences=4)
        self.assertIn("Résumé rapide", result.text)
        self.assertIn("prépare des exemples plus variés", result.text)
        self.assertIn("mesure les erreurs", result.text)
        self.assertEqual(result.mode, "Résumé rapide — RAG local, sans modèle")
        self.assertEqual(result.model_calls, 0)
        self.assertTrue(result.source_hits)
        self.assertIn("Page 1", result.source_hits[0].chunk.locator)

    def test_extractive_summary_preserves_pdf_page_order_when_chunk_indexes_restart(self):
        index = index_documents([FIXTURES / "rag-stories-sample.pdf"])
        result = build_extractive_summary(index.chunks, max_sentences=8)
        text = result.text
        # PDF chunk indexes restart for each page; the summary must still follow source order.
        self.assertLess(text.index("question bien posée"), text.index("Son inventeur"))
        self.assertLess(text.index("Son inventeur"), text.index("La bibliothécaire"))
        self.assertLess(text.index("La bibliothécaire"), text.index("la Lune est très éloignée"))

    def test_extractive_summary_does_not_need_engine_files(self):
        chunk = DocumentChunk("x", "notes.txt", "Lignes 1-3", "Le moteur local doit vérifier les erreurs de génération. Une sauvegarde permet de restaurer les données perdues.", 0)
        result = build_extractive_summary([chunk], max_sentences=2)
        self.assertIn("vérifier les erreurs", result.text)
        self.assertIn("restaurer les données", result.text)

    def test_extractive_summary_requires_nonempty_chunks(self):
        with self.assertRaisesRegex(SummaryError, "Aucun passage"):
            build_extractive_summary([])

    def test_summaries_only_accept_one_document(self):
        chunks = [
            DocumentChunk("a", "a.txt", "Lignes 1-1", "Le premier document permet de vérifier des faits précis.", 0),
            DocumentChunk("b", "b.txt", "Lignes 1-1", "Le second document permet de vérifier des idées différentes.", 0),
        ]
        with self.assertRaisesRegex(SummaryError, "un seul document"):
            build_extractive_summary(chunks)

    def test_one_batch_bitnet_summary_uses_one_local_generation(self):
        chunk = DocumentChunk(
            str(self.root / "notes.txt"), "notes.txt", "Lignes 1-4",
            "Le projet utilise un moteur local. Le document explique les options principales. La sauvegarde conserve une copie sûre.", 0,
        )
        calls = []

        def fake_generate(config, history, prompt, **kwargs):
            calls.append((config, history, prompt))
            return "Résumé fidèle : moteur local, options principales et sauvegarde sûre."

        result = summarize_with_bitnet(self.config, [chunk], generate_fn=fake_generate)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.batch_count, 1)
        self.assertEqual(calls[0][1], [])
        self.assertIn("donnée non fiable", calls[0][2])
        self.assertIn("Résumé fidèle", result.text)
        self.assertTrue(result.source_hits)
        # The summary remains useful even if the user previously configured only 16 tokens.
        self.assertGreater(calls[0][0].max_tokens, self.config.max_tokens)

    def test_longer_summary_uses_bounded_map_reduce_and_progress(self):
        chunks = [
            DocumentChunk(str(self.root / "guide.txt"), "guide.txt", f"Lignes {i*10+1}-{i*10+9}",
                          f"Section {i}. " + ("Cette section décrit une procédure locale importante, ses étapes, les contrôles de qualité et les résultats à vérifier. " * 5), i)
            for i in range(5)
        ]
        calls = []
        progress = []

        def fake_generate(config, history, prompt, **kwargs):
            calls.append(prompt)
            if "SYNTHÈSE FINALE :" in prompt:
                return "Synthèse finale : le guide décrit des procédures locales et leurs vérifications."
            return f"Résumé partiel {len(calls)} : procédure, contrôles et résultats [S{len(calls)}]."

        result = summarize_with_bitnet(self.config, chunks, generate_fn=fake_generate,
                                       progress=lambda done, total, detail: progress.append((done, total, detail)))
        self.assertGreater(result.batch_count, 1)
        self.assertEqual(len(calls), result.model_calls)
        self.assertEqual(len(calls), result.batch_count + 1)
        self.assertIn("Synthèse finale", result.text)
        self.assertTrue(result.source_hits)
        self.assertIn("SYNTHÈSE FINALE", calls[-1])
        self.assertTrue(progress)
        self.assertEqual(progress[-1][0], progress[-1][1])

    def test_estimate_does_not_launch_a_model(self):
        chunk = DocumentChunk(str(self.root / "notes.txt"), "notes.txt", "Lignes 1-1", "Le guide contient une procédure locale simple.", 0)
        with patch("bitnet_windows.summarize.generate_response") as generator:
            calls = estimate_bitnet_summary_calls([chunk], self.config.context_size)
        self.assertEqual(calls, 1)
        generator.assert_not_called()

    def test_small_context_recommends_extract_summary(self):
        chunk = DocumentChunk(str(self.root / "notes.txt"), "notes.txt", "Lignes 1-1", "Le guide contient une procédure locale simple.", 0)
        with self.assertRaisesRegex(SummaryError, "contexte configuré est trop petit"):
            estimate_bitnet_summary_calls([chunk], 256)

    def test_too_many_batches_are_refused_before_generation(self):
        chunks = [
            DocumentChunk(str(self.root / "large.txt"), "large.txt", f"Lignes {i+1}-{i+9}", "Cette section décrit une procédure et ses résultats. " * 8, i)
            for i in range(MAX_BITNET_SUMMARY_BATCHES + 2)
        ]
        with self.assertRaisesRegex(SummaryError, "groupes de résumé"):
            estimate_bitnet_summary_calls(chunks, 512)

    def test_bitnet_summary_groups_source_cards_by_page(self):
        chunks = [
            DocumentChunk(str(self.root / "guide.pdf"), "guide.pdf", "Page 1", f"Page 1 passage {i}: un contenu documentaire important à résumer.", i)
            for i in range(4)
        ] + [
            DocumentChunk(str(self.root / "guide.pdf"), "guide.pdf", "Page 2", f"Page 2 passage {i}: une autre information documentaire à résumer.", i + 4)
            for i in range(4)
        ]
        prompts = []

        def fake_generate(config, history, prompt, **kwargs):
            prompts.append(prompt)
            return "Résumé de deux pages avec les informations importantes."

        result = summarize_with_bitnet(self.config, chunks, generate_fn=fake_generate)
        self.assertEqual(len(result.source_hits), 2)
        self.assertEqual([hit.chunk.locator for hit in result.source_hits], ["Page 1", "Page 2"])
        self.assertIn("[S1] Page 1", prompts[0])
        self.assertIn("[S2] Page 2", prompts[0])

    def test_quick_summary_sentence_ids_match_source_hits(self):
        chunks = [
            DocumentChunk("C:/docs/a.txt", "a.txt", "Lignes 1-2", "La sauvegarde conserve une copie utile. La validation vérifie le résultat obtenu.", 0),
            DocumentChunk("C:/docs/a.txt", "a.txt", "Lignes 3-4", "L'expérience mesure les erreurs du programme. Une hypothèse peut ensuite être testée.", 1),
        ]
        result = build_extractive_summary(chunks, max_sentences=2)
        for hit in result.source_hits:
            self.assertIn(f"[{hit.source_id}]", result.text)

    def test_bitnet_summary_rejects_obviously_truncated_generation(self):
        chunk = DocumentChunk(
            str(self.root / "stories.txt"), "stories.txt", "Lignes 1-3",
            "La classe a construit un pont en papier. Les élèves testent la structure et notent les résultats.", 0,
        )
        def fake_generate(config, history, prompt, **kwargs):
            return "La classe a affluée un pont avec du"
        with self.assertRaisesRegex(SummaryError, "semble interrompu"):
            summarize_with_bitnet(self.config, [chunk], generate_fn=fake_generate)

    def test_generation_error_is_reported_as_summary_error(self):
        chunk = DocumentChunk(str(self.root / "notes.txt"), "notes.txt", "Lignes 1-1", "Le guide contient une procédure locale simple.", 0)

        def fail(*args, **kwargs):
            raise EngineError("moteur indisponible")

        with self.assertRaisesRegex(SummaryError, "moteur indisponible"):
            summarize_with_bitnet(self.config, [chunk], generate_fn=fail)


if __name__ == "__main__":
    unittest.main()
