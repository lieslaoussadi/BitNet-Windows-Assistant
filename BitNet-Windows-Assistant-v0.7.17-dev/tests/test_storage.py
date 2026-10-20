import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bitnet_windows.rag import index_documents
from bitnet_windows.storage import LocalStore, StorageError, default_database_path, default_backup_directory


class BackupDirectoryTests(unittest.TestCase):
    def test_default_backup_directory_is_named_sauvegardes_in_project_root(self):
        directory = default_backup_directory()
        self.assertEqual(directory.name, "Sauvegardes")
        self.assertEqual(directory, Path(__file__).resolve().parents[1] / "Sauvegardes")


class LocalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "assistant_data.sqlite3"
        self.store = LocalStore(self.database)

    def tearDown(self):
        self.temp.cleanup()

    def test_conversations_and_messages_survive_reopening_database(self):
        conversation_id = self.store.create_conversation("Discussion générale")
        self.store.add_message(conversation_id, "system", "Bienvenue")
        self.store.add_message(conversation_id, "user", "Bonjour mon ami", "2026-10-18T12:00:00+01:00")
        self.store.add_message(conversation_id, "assistant", "Salut !", display_name="BitNet 1.58")
        self.store.update_conversation_title(conversation_id, "Bonjour mon ami")

        reopened = LocalStore(self.database)
        conversations = reopened.list_conversations()
        self.assertEqual(conversations[0]["id"], conversation_id)
        self.assertEqual(conversations[0]["title"], "Bonjour mon ami")
        messages = reopened.list_messages(conversation_id)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[1]["timestamp"], "2026-10-18T12:00:00+01:00")
        self.assertEqual(messages[2]["display_name"], "BitNet 1.58")

    def test_clear_and_delete_conversation_are_explicit_and_cascade(self):
        conversation_id = self.store.create_conversation("À supprimer")
        self.store.add_message(conversation_id, "user", "Message privé")
        self.store.clear_conversation(conversation_id)
        self.assertEqual(self.store.list_messages(conversation_id), [])
        self.store.add_message(conversation_id, "user", "À nouveau")
        self.store.delete_conversation(conversation_id)
        self.assertIsNone(self.store.get_conversation(conversation_id))
        self.assertEqual(self.store.list_conversations(), [])

    def test_memories_are_persistent_editable_enableable_and_deletable(self):
        preference_id = self.store.create_memory(
            "Réponds de préférence en français et explique étape par étape.",
            category="Préférence",
        )
        project_id = self.store.create_memory(
            "BitNet Assistant utilise un index SQLite local.", category="Projet"
        )
        self.store.set_memory_enabled(project_id, False)
        reopened = LocalStore(self.database)
        items = {item["id"]: item for item in reopened.list_memories()}
        self.assertEqual(items[preference_id]["content"], "Réponds de préférence en français et explique étape par étape.")
        self.assertEqual(items[project_id]["enabled"], 0)
        context = reopened.enabled_memories_for_query("Comment fonctionne SQLite dans BitNet ?")
        self.assertIn(preference_id, [item["id"] for item in context])
        self.assertNotIn(project_id, [item["id"] for item in context])
        reopened.update_memory(preference_id, "Réponses courtes en français.", category="Règle")
        self.assertEqual(reopened.get_memory(preference_id)["content"], "Réponses courtes en français.")
        reopened.delete_memory(preference_id)
        self.assertIsNone(reopened.get_memory(preference_id))

    def test_non_matching_fact_memories_are_not_injected_into_unrelated_questions(self):
        self.store.create_memory("Le projet utilise une base SQLite.", category="Projet")
        memories = self.store.enabled_memories_for_query("Quelle est la météo sur Mars ?")
        self.assertEqual(memories, [])

    def test_memory_query_limit_zero_returns_no_memories(self):
        self.store.create_memory("Réponds en français.", category="Préférence")
        self.assertEqual(self.store.enabled_memories_for_query("Bonjour", limit=0), [])

    def test_settings_survive_reopening(self):
        self.store.set_setting("memories_enabled", "0")
        self.assertEqual(LocalStore(self.database).get_setting("memories_enabled"), "0")

    def test_rag_index_survives_restart_without_copying_source_file(self):
        source = self.root / "notes.txt"
        source.write_text("BitNet utilise un index lexical local.\nLa mémoire doit rester privée.", encoding="utf-8")
        index = index_documents([source])
        self.assertTrue(index.chunks)
        warnings = self.store.save_rag_index(index)
        self.assertEqual(warnings, [])
        reopened = LocalStore(self.database)
        restored = reopened.load_rag_index()
        self.assertEqual(restored.indexed_files, [str(source.resolve())])
        self.assertEqual(restored.chunk_count, index.chunk_count)
        self.assertEqual(restored.chunks[0].text, index.chunks[0].text)
        self.assertTrue(source.is_file())

    def test_multipage_pdf_index_persists_and_can_be_saved_again(self):
        source = Path(__file__).parent / "fixtures" / "rag-stories-sample.pdf"
        index = index_documents([source])
        self.assertGreater(index.chunk_count, 1)
        self.assertEqual(
            len({chunk.chunk_index for chunk in index.chunks}), index.chunk_count
        )

        self.assertEqual(self.store.save_rag_index(index), [])
        # Saving the same complete index again must atomically replace the old one.
        self.assertEqual(self.store.save_rag_index(index), [])
        restored = LocalStore(self.database).load_rag_index()
        self.assertEqual(restored.document_count, 1)
        self.assertEqual(restored.chunk_count, index.chunk_count)
        self.assertEqual(
            [chunk.chunk_index for chunk in restored.chunks],
            [chunk.chunk_index for chunk in index.chunks],
        )

    def test_rag_storage_rejects_duplicate_chunk_indices_before_writing(self):
        from bitnet_windows.rag import DocumentChunk, RAGIndex

        source = self.root / "duplicate.txt"
        source.write_text("Contenu pour test des doublons.", encoding="utf-8")
        base = index_documents([source])
        first = base.chunks[0]
        duplicate = DocumentChunk(
            source_path=first.source_path,
            filename=first.filename,
            locator="Lignes supplémentaires",
            text="Passage de test dupliqué",
            chunk_index=first.chunk_index,
        )
        self.assertEqual(self.store.save_rag_index(base), [])
        bad_index = RAGIndex(
            chunks=[first, duplicate], indexed_files=base.indexed_files.copy()
        )
        with self.assertRaisesRegex(StorageError, "numéro de passage dupliqué"):
            self.store.save_rag_index(bad_index)
        # A rejected index must not replace or erase the previously saved index.
        restored = LocalStore(self.database).load_rag_index()
        self.assertEqual(restored.chunk_count, base.chunk_count)
        self.assertEqual(restored.chunks[0].text, base.chunks[0].text)

    def test_targeted_upsert_preserves_other_documents_and_conversations(self):
        first = self.root / "first.txt"
        second = self.root / "second.txt"
        first.write_text("Document alpha initial.", encoding="utf-8")
        second.write_text("Document beta stable.", encoding="utf-8")
        conversation_id = self.store.create_conversation("À conserver")
        self.store.add_message(conversation_id, "user", "Message qui doit rester")
        self.store.save_rag_documents(index_documents([first, second]))

        first.write_text("Document alpha modifié avec un contenu plus long.", encoding="utf-8")
        self.store.save_rag_documents(index_documents([first]))
        restored = LocalStore(self.database).load_rag_index()
        self.assertEqual(restored.document_count, 2)
        self.assertEqual({Path(p).name for p in restored.indexed_files}, {"first.txt", "second.txt"})
        by_name = {Path(c.source_path).name: c.text for c in restored.chunks}
        self.assertIn("modifié", by_name["first.txt"])
        self.assertIn("stable", by_name["second.txt"])
        self.assertEqual(self.store.list_messages(conversation_id)[0]["content"], "Message qui doit rester")

    def test_deleting_one_document_index_preserves_other_documents_and_source_files(self):
        first = self.root / "delete-me.txt"
        second = self.root / "keep-me.txt"
        first.write_text("Texte à retirer de l'index.", encoding="utf-8")
        second.write_text("Texte à garder dans l'index.", encoding="utf-8")
        self.store.save_rag_documents(index_documents([first, second]))

        deleted = self.store.delete_rag_documents([first])
        restored = LocalStore(self.database).load_rag_index()
        self.assertEqual(deleted, 1)
        self.assertEqual(restored.document_count, 1)
        self.assertEqual({Path(p).name for p in restored.indexed_files}, {"keep-me.txt"})
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertIn("garder", restored.chunks[0].text)

    def test_stale_rag_file_detection_after_source_modification(self):
        source = self.root / "change-during-session.txt"
        source.write_text("Texte initial.", encoding="utf-8")
        self.store.save_rag_documents(index_documents([source]))
        source.write_text("Texte modifié avec une taille différente.", encoding="utf-8")
        self.assertEqual(self.store.list_stale_rag_files(), [str(source.resolve())])

    def test_targeted_upsert_rejects_duplicate_indices_without_touching_other_docs(self):
        from bitnet_windows.rag import DocumentChunk, RAGIndex
        good = self.root / "good.txt"
        bad = self.root / "bad.txt"
        good.write_text("Document qui doit rester.", encoding="utf-8")
        bad.write_text("Document invalide.", encoding="utf-8")
        self.store.save_rag_documents(index_documents([good]))
        base = index_documents([bad])
        first = base.chunks[0]
        duplicate = DocumentChunk(first.source_path, first.filename, "autre passage", "doublon", first.chunk_index)
        with self.assertRaisesRegex(StorageError, "numéro de passage dupliqué"):
            self.store.save_rag_documents(RAGIndex(chunks=[first, duplicate], indexed_files=base.indexed_files))
        restored = LocalStore(self.database).load_rag_index()
        self.assertEqual({Path(p).name for p in restored.indexed_files}, {"good.txt"})

    def test_rag_index_refuses_to_reuse_chunks_after_source_file_changes(self):
        source = self.root / "notes.txt"
        source.write_text("Le contenu initial est court.", encoding="utf-8")
        self.store.save_rag_index(index_documents([source]))
        source.write_text("Le contenu a changé, avec plus de détails désormais présents dans le fichier.", encoding="utf-8")
        restored = LocalStore(self.database).load_rag_index()
        self.assertEqual(restored.chunk_count, 0)
        self.assertEqual(restored.indexed_files, [])
        self.assertIn(str(source.resolve()), restored.stale_files)
        self.assertTrue(any("a changé" in warning or "changé" in warning for warning in restored.warnings))

    def test_missing_source_is_visible_after_restart_and_index_is_preserved(self):
        source = self.root / "notes.txt"
        content = "Le contenu indexé reste conservé si le fichier est déplacé."
        source.write_text(content, encoding="utf-8")
        self.store.save_rag_index(index_documents([source]))
        original_stat = source.stat()
        source.unlink()

        reopened = LocalStore(self.database)
        restored = reopened.load_rag_index()
        self.assertEqual(restored.chunk_count, 0)
        self.assertEqual(restored.indexed_files, [])
        self.assertEqual(restored.stale_files, [str(source.resolve())])
        self.assertTrue(any("introuvable" in warning for warning in restored.warnings))
        self.assertEqual(reopened.list_stale_rag_files(), [str(source.resolve())])

        # Missing sources must not be silently deleted from SQLite. If the original
        # file is restored unchanged at its original path, the saved index is usable.
        source.write_text(content, encoding="utf-8")
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        restored_again = LocalStore(self.database).load_rag_index()
        self.assertEqual(restored_again.indexed_files, [str(source.resolve())])
        self.assertGreater(restored_again.chunk_count, 0)
        self.assertIn("conservé", restored_again.chunks[0].text)

    def test_database_path_can_be_overridden_for_isolated_local_testing(self):
        with patch.dict(os.environ, {"BITNET_ASSISTANT_DATA_DIR": str(self.root / "custom")}):
            self.assertEqual(default_database_path(), self.root / "custom" / "assistant_data.sqlite3")

    def test_empty_memory_is_rejected(self):
        with self.assertRaisesRegex(StorageError, "ne peut pas être vide"):
            self.store.create_memory("   ")


if __name__ == "__main__":
    unittest.main()

class LocalStoreBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "live.sqlite3"
        self.store = LocalStore(self.db_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_integrity_check_reports_ok_for_healthy_database(self):
        self.assertEqual(self.store.integrity_check(), ["ok"])

    def test_backup_is_consistent_and_preserves_chats_memories_and_settings(self):
        conversation = self.store.create_conversation("Sauvegarde de test")
        self.store.add_message(conversation, "user", "Message à conserver")
        self.store.create_memory("Souvenir de test", category="Préférence")
        self.store.set_setting("backup_test_setting", "preserved")

        backup_path = self.root / "backup.sqlite3"
        self.assertEqual(self.store.backup_to(backup_path), backup_path.resolve())
        self.assertTrue(backup_path.is_file())
        restored = LocalStore(backup_path)
        self.assertEqual(restored.integrity_check(), ["ok"])
        self.assertEqual(restored.list_messages(conversation)[0]["content"], "Message à conserver")
        self.assertEqual(restored.get_setting("backup_test_setting"), "preserved")
        self.assertEqual(len(restored.list_memories()), 1)

    def test_backup_refuses_to_overwrite_active_database(self):
        with self.assertRaisesRegex(StorageError, "autre fichier"):
            self.store.backup_to(self.db_path)
