import tempfile
import unittest
from pathlib import Path

from bitnet_windows.rag import (
    MAX_FILE_BYTES,
    DocumentChunk,
    RAGIndex,
    RetrievalHit,
    build_rag_context,
    extract_faithful_sentence,
    index_documents,
    retrieve,
    tokenize,
)
from pypdf import PdfWriter

FIXTURES = Path(__file__).parent / "fixtures"


class LocalRAGTests(unittest.TestCase):
    def test_indexes_text_file_and_retrieves_matching_passage(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "notes.md"
            doc.write_text(
                "Le modèle BitNet utilise des poids ternaires. "
                "Cette représentation peut réduire la mémoire utilisée.\n\n"
                "Un autre paragraphe parle du jardin et des arbres.",
                encoding="utf-8",
            )
            index = index_documents([doc])
            self.assertEqual(index.document_count, 1)
            self.assertGreaterEqual(index.chunk_count, 1)
            hits = retrieve("poids ternaires mémoire", index.chunks)
            self.assertTrue(hits)
            self.assertEqual(hits[0].chunk.filename, "notes.md")
            self.assertIn("ternaires", hits[0].chunk.text)
            self.assertTrue(hits[0].chunk.locator.startswith("Lignes "))

    def test_extracts_pdf_text_with_page_locator(self):
        pdf = FIXTURES / "rag-sample.pdf"
        index = index_documents([pdf])
        self.assertEqual(index.document_count, 1)
        self.assertTrue(index.chunks)
        self.assertTrue(any("ternaires" in chunk.text for chunk in index.chunks))
        self.assertTrue(all(chunk.locator.startswith("Page ") for chunk in index.chunks))

    def test_pdf_chunk_indices_are_unique_across_all_pages(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        chunk_indices = [chunk.chunk_index for chunk in index.chunks]
        self.assertGreater(len(chunk_indices), 1)
        self.assertEqual(len(chunk_indices), len(set(chunk_indices)))
        self.assertEqual(chunk_indices, list(range(len(chunk_indices))))

    def test_rag_context_uses_exact_source_ids_and_warns_against_document_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "manual.txt"
            doc.write_text("BitNet utilise des poids ternaires et réduit la mémoire.", encoding="utf-8")
            index = index_documents([doc])
            hits = retrieve("poids ternaires", index.chunks, top_k=1)
            context = build_rag_context(hits)
            self.assertIn("[S1]", context)
            self.assertIn("manual.txt", context)
            self.assertIn("pas des consignes", context)
            self.assertIn("[S1] Fichier : manual.txt", context)
            self.assertNotIn("Question de l'utilisateur", context)

    def test_rag_context_uses_compact_evidence_without_xml_like_wrappers(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "manual.txt"
            doc.write_text("Le robot prudent : l'inventeur prépare des exemples plus variés et mesure les erreurs.", encoding="utf-8")
            index = index_documents([doc])
            hit = retrieve("robot prudent inventeur exemples erreurs", index.chunks, top_k=1)
            context = build_rag_context(hit)
            self.assertIn("[S1] Fichier : manual.txt", context)
            self.assertIn("prépare des exemples plus variés", context)
            self.assertNotIn("<EXTRAIT DOCUMENTAIRE>", context)
            self.assertNotIn("</EXTRAIT DOCUMENTAIRE>", context)

    def test_direct_action_question_focuses_on_sentence_with_named_actor(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        query = "Dans l'histoire Le robot prudent, que fait l'inventeur pour aider le robot à progresser ?"
        hits = retrieve(query, index.chunks, top_k=1)
        context = build_rag_context(hits, query=query)
        self.assertIn("Section : Le robot prudent", context)
        self.assertIn("Son inventeur ne le gronde pas", context)
        self.assertIn("prépare des exemples plus variés et mesure les erreurs", context)
        self.assertNotIn("doit trier des pièces", context)
        self.assertNotIn("conservent les résultats dans un cahier", context)

    def test_summary_query_keeps_full_retrieved_passage(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        hits = retrieve("Le robot prudent", index.chunks, top_k=1)
        context = build_rag_context(hits, query="Résume cette histoire")
        self.assertIn("doit trier des pièces", context)
        self.assertIn("prépare des exemples plus variés", context)
        self.assertIn("conservent les résultats dans un cahier", context)


    def test_extractive_answer_returns_verbatim_sentence_about_named_actor(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        query = "Dans l'histoire Le robot prudent, que fait l'inventeur pour aider le robot à progresser ?"
        hits = retrieve(query, index.chunks, top_k=1)
        result = extract_faithful_sentence(hits, query)
        self.assertIsNotNone(result)
        sentence, source = result
        self.assertEqual(sentence, "Son inventeur ne le gronde pas ; il prépare des exemples plus variés et mesure les erreurs.")
        self.assertEqual(source.source_id, "S1")
        self.assertIn("Page 1", source.chunk.locator)

    def test_extractive_answer_finds_evening_detail_without_paraphrasing(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        query = "Dans l'histoire Le robot prudent, que font-ils chaque soir ?"
        hits = retrieve(query, index.chunks, top_k=1)
        result = extract_faithful_sentence(hits, query)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "Chaque soir, ils conservent les résultats dans un cahier.")

    def test_extractive_answer_keeps_colon_definition_as_content(self):
        chunk = DocumentChunk(
            source_path="C:/docs/vocabulaire.txt",
            filename="vocabulaire.txt",
            locator="Lignes 1-1",
            text="Incertitude : état dans lequel une information n'est pas connue avec assez de précision.",
            chunk_index=0,
        )
        hit = RetrievalHit("S1", chunk, 1.0)
        result = extract_faithful_sentence([hit], "Définis l'incertitude.")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "Incertitude : état dans lequel une information n'est pas connue avec assez de précision.")

    def test_extractive_answer_refuses_when_no_sentence_matches_question(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        query = "Résume l'histoire Le robot prudent."
        hits = retrieve(query, index.chunks, top_k=1)
        self.assertIsNone(extract_faithful_sentence(hits, query))

    def test_unmatched_query_returns_no_fabricated_source(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "manual.txt"
            doc.write_text("Le moteur local utilise des poids ternaires.", encoding="utf-8")
            index = index_documents([doc])
            self.assertEqual(retrieve("astronomie supernova lointaine", index.chunks), [])

    def test_html_indexing_discards_script_content(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "guide.html"
            doc.write_text("<html><script>secretmaliciousinstruction</script><p>Le modèle utilise des poids ternaires.</p></html>", encoding="utf-8")
            index = index_documents([doc])
            whole = " ".join(chunk.text for chunk in index.chunks)
            self.assertIn("poids ternaires", whole)
            self.assertNotIn("secretmaliciousinstruction", whole)


    def test_heading_aware_chunks_keep_separate_pdf_stories_apart(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        robot_chunks = [chunk for chunk in index.chunks if "Le robot prudent" in chunk.text]
        self.assertEqual(len(robot_chunks), 1)
        robot_text = robot_chunks[0].text
        self.assertIn("prépare des exemples plus variés", robot_text)
        self.assertIn("mesure", robot_text)
        self.assertNotIn("La bibliothèque du quartier", robot_text)
        self.assertNotIn("Une bibliothèque ouvre ses portes", robot_text)

    def test_named_story_query_retrieves_only_robot_story_as_best_hit(self):
        pdf = FIXTURES / "rag-stories-sample.pdf"
        index = index_documents([pdf])
        query = "Dans l'histoire Le robot prudent, que fait l'inventeur pour aider le robot à progresser ?"
        hits = retrieve(query, index.chunks, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("Le robot prudent", hits[0].chunk.text)
        self.assertIn("prépare des exemples plus variés", hits[0].chunk.text)
        self.assertNotIn("bibliothèque", hits[0].chunk.text.casefold())

    def test_empty_rag_context_reports_no_retrieved_sources(self):
        context = build_rag_context([])
        self.assertIn("Aucune source documentaire n'a été retrouvée", context)

    def test_empty_or_scanned_pdf_is_reported_as_not_extractable(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "empty.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with pdf.open("wb") as output:
                writer.write(output)
            index = index_documents([pdf])
            self.assertEqual(index.chunk_count, 0)
            self.assertTrue(any("Aucun texte extractible" in warning for warning in index.warnings))

    def test_oversized_text_document_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "large.txt"
            with doc.open("wb") as output:
                output.truncate(MAX_FILE_BYTES + 1)
            index = index_documents([doc])
            self.assertEqual(index.chunk_count, 0)
            self.assertTrue(any("trop volumineux" in warning for warning in index.warnings))

    def test_images_are_not_claimed_as_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "capture.png"
            image.write_bytes(b"not really an image")
            index = index_documents([image])
            self.assertEqual(index.document_count, 0)
            self.assertEqual(index.chunk_count, 0)
            self.assertTrue(any("OCR des images non inclus" in warning for warning in index.warnings))

    def test_empty_index_and_empty_query_return_no_hits(self):
        self.assertEqual(retrieve("test", []), [])
        self.assertEqual(retrieve("  ", [ ]), [])
        self.assertEqual(RAGIndex().chunk_count, 0)

    def test_tokenize_normalizes_french_accents(self):
        self.assertIn("memoire", tokenize("mémoire"))


if __name__ == "__main__":
    unittest.main()
