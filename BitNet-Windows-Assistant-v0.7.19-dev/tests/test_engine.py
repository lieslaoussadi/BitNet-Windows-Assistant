import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bitnet_windows.engine import (
    EngineConfig, EngineError, build_command, config_from_mapping, generate_response, load_config, render_prompt, save_config, validate_config,
)


class EngineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.executable = self.root / ("llama-cli.exe" if __import__("os").name == "nt" else "llama-cli")
        self.model = self.root / "model.gguf"
        self.executable.write_bytes(b"placeholder executable for mocked tests")
        self.model.write_bytes(b"placeholder gguf for mocked tests")
        self.config = EngineConfig(
            executable=str(self.executable), model_path=str(self.model),
            context_size=1024, threads=4, gpu_layers=0, max_tokens=96, temperature=0.2,
        )

    def tearDown(self):
        self.temp.cleanup()


    def test_auto_start_defaults_to_disabled_for_legacy_settings(self):
        loaded = config_from_mapping({"executable": str(self.executable), "model_path": str(self.model)})
        self.assertFalse(loaded.auto_start)

    def test_auto_start_setting_round_trips_in_local_config(self):
        import dataclasses
        from unittest.mock import patch

        path = self.root / "BitNetWindowsAssistant" / "engine.json"
        enabled = dataclasses.replace(self.config, auto_start=True)
        with patch("bitnet_windows.engine.settings_path", return_value=path):
            save_config(enabled)
            loaded = load_config()
        self.assertTrue(loaded.auto_start)
        self.assertEqual(loaded.executable, self.config.executable)
        self.assertEqual(loaded.model_path, self.config.model_path)

    def test_auto_start_false_setting_round_trips_in_local_config(self):
        import dataclasses
        from unittest.mock import patch

        path = self.root / "BitNetWindowsAssistant" / "engine.json"
        disabled = dataclasses.replace(self.config, auto_start=False)
        with patch("bitnet_windows.engine.settings_path", return_value=path):
            save_config(disabled)
            loaded = load_config()
        self.assertFalse(loaded.auto_start)

    def test_render_prompt_matches_tested_user_assistant_template(self):
        prompt = render_prompt([
            {"role": "system", "content": "not included"},
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Salut"},
        ], "Que peux-tu faire ?")
        self.assertTrue(prompt.startswith("<|begin_of_text|>User: Bonjour<|eot_id|>Assistant: Salut<|eot_id|>"))
        self.assertTrue(prompt.endswith("User: Que peux-tu faire ?<|eot_id|>Assistant: "))
        self.assertNotIn("not included", prompt)

    def test_prompt_limits_previous_history_to_eight_messages(self):
        history = [{"role": "user", "content": f"message-{i}"} for i in range(12)]
        prompt = render_prompt(history, "dernier")
        self.assertNotIn("message-0", prompt)
        self.assertIn("message-11", prompt)
        self.assertIn("dernier", prompt)

    def test_command_uses_local_cli_no_shell_and_expected_model_controls(self):
        command = build_command(self.config, self.root / "prompt.txt")
        self.assertEqual(command[0], str(self.executable))
        self.assertIn("-m", command)
        self.assertIn(str(self.model), command)
        self.assertIn("-no-cnv", command)
        self.assertIn("--override-kv", command)
        self.assertIn("tokenizer.ggml.pre=str:llama-bpe", command)
        self.assertIn("--simple-io", command)
        # In the pinned llama.cpp revision, --no-show-timings is available to
        # llama-cli but not llama-completion; do not pass this CLI-specific flag.
        self.assertNotIn("--no-show-timings", command)

    def test_validate_rejects_missing_model(self):
        missing = EngineConfig(executable=str(self.executable), model_path=str(self.root / "missing.gguf"))
        with self.assertRaisesRegex(EngineError, "introuvable"):
            validate_config(missing)

    def test_validate_rejects_control_character_in_path(self):
        malformed = EngineConfig(executable="bad\npath.exe", model_path=str(self.model))
        with self.assertRaisesRegex(EngineError, "caractère de contrôle"):
            validate_config(malformed, require_files=False)

    def test_generation_cleans_prompt_file_and_returns_only_answer(self):
        captured = {}
        def fake_run(command, **kwargs):
            prompt_path = Path(command[command.index("-f") + 1])
            captured["prompt_path"] = prompt_path
            captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="  Paris  <|eot_id|>\n", stderr="")
        with patch("bitnet_windows.engine.subprocess.run", side_effect=fake_run):
            answer = generate_response(self.config, [{"role": "user", "content": "Salut"}], "Capitale de la France ?")
        self.assertEqual(answer, "Paris")
        self.assertIn("Capitale de la France ?", captured["prompt_text"])
        self.assertFalse(captured["prompt_path"].exists())

    def test_cleans_literal_end_of_text_marker(self):
        from bitnet_windows.engine import _clean_generation
        self.assertEqual(_clean_generation("  La capitale est Paris. [end of text]  "), "La capitale est Paris.")
        self.assertEqual(_clean_generation("Réponse <|end_of_text|>\n"), "Réponse")

    def test_generation_prompt_ends_with_direct_answer_instruction(self):
        captured = {}
        def fake_run(command, **kwargs):
            prompt_path = Path(command[command.index("-f") + 1])
            captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="La mémoire est réduite. [end of text]", stderr="")
        with patch("bitnet_windows.engine.subprocess.run", side_effect=fake_run):
            answer = generate_response(self.config, [], "Pourquoi ?", context="SOURCES LOCALES : [S1] BitNet utilise des poids ternaires.")
        self.assertEqual(answer, "La mémoire est réduite.")
        self.assertIn("[S1] BitNet utilise des poids ternaires", captured["prompt_text"])
        self.assertIn("QUESTION :\nPourquoi ?", captured["prompt_text"])
        self.assertIn("Ne recopie pas tout l'extrait", captured["prompt_text"])
        self.assertIn("Distingue les acteurs", captured["prompt_text"])
        self.assertTrue(captured["prompt_text"].endswith("RÉPONSE :<|eot_id|>Assistant: "))

    def test_rag_request_excludes_prior_history_to_avoid_answer_contamination(self):
        captured = {}
        history = [
            {"role": "user", "content": "Question précédente"},
            {"role": "assistant", "content": "Réponse antérieure potentiellement erronée"},
        ]
        def fake_run(command, **kwargs):
            prompt_path = Path(command[command.index("-f") + 1])
            captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="OK", stderr="")
        with patch("bitnet_windows.engine.subprocess.run", side_effect=fake_run):
            generate_response(self.config, history, "Question actuelle", context="Contexte RAG court.")
        self.assertNotIn("Question précédente", captured["prompt_text"])
        self.assertNotIn("Réponse antérieure potentiellement erronée", captured["prompt_text"])
        self.assertIn("Question actuelle", captured["prompt_text"])

    def test_memory_context_is_separated_from_rag_citations_and_evidence(self):
        captured = {}
        def fake_run(command, **kwargs):
            prompt_path = Path(command[command.index("-f") + 1])
            captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="Réponse en français.", stderr="")
        with patch("bitnet_windows.engine.subprocess.run", side_effect=fake_run):
            generate_response(
                self.config, [], "Que dit le document ?",
                context="SOURCES DOCUMENTAIRES : [S1] Le document décrit une méthode.",
                memory_context="[Préférence] Réponds en français.",
            )
        prompt = captured["prompt_text"]
        self.assertIn("SOUVENIRS PERSONNELS APPROUVÉS", prompt)
        self.assertIn("[Préférence] Réponds en français.", prompt)
        self.assertIn("SOURCES DOCUMENTAIRES : [S1] Le document décrit une méthode.", prompt)
        self.assertIn("Que dit le document ?", prompt)

    def test_memory_context_is_optional_in_normal_chat(self):
        captured = {}
        def fake_run(command, **kwargs):
            prompt_path = Path(command[command.index("-f") + 1])
            captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="OK", stderr="")
        with patch("bitnet_windows.engine.subprocess.run", side_effect=fake_run):
            generate_response(self.config, [], "Quelle heure est-il ?", memory_context="[Préférence] Réponses courtes.")
        self.assertIn("SOUVENIRS PERSONNELS APPROUVÉS", captured["prompt_text"])
        self.assertIn("QUESTION ACTUELLE :\nQuelle heure est-il ?", captured["prompt_text"])

    def test_nonzero_cli_exit_reports_stderr(self):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad model")
        with patch("bitnet_windows.engine.subprocess.run", side_effect=fake_run):
            with self.assertRaisesRegex(EngineError, "bad model"):
                generate_response(self.config, [], "Bonjour")

    def test_timeout_is_reported_clearly(self):
        with patch("bitnet_windows.engine.subprocess.run", side_effect=subprocess.TimeoutExpired("llama-cli", 5)):
            with self.assertRaisesRegex(EngineError, "n'a pas répondu"):
                generate_response(self.config, [], "Bonjour")


if __name__ == "__main__":
    unittest.main()
