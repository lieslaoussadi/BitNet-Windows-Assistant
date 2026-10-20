"""Small, dependency-free graphical front end for BitNet Windows Assistant."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SOURCE_DIR.parent


def parse_library_root(raw_path: str) -> Path:
    """Reject control characters before interpreting a user-selected path."""
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise ValueError(
            "Le chemin contient un retour à la ligne ou un caractère de contrôle. "
            "Efface le champ et saisis un chemin sur une seule ligne."
        )
    text_path = raw_path.strip().strip('"')
    if not text_path:
        raise ValueError("Indique un dossier racine.")
    return Path(text_path).expanduser()



class AssistantWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BitNet Windows Assistant")
        self.root.geometry("820x620")
        self.root.minsize(680, 480)
        self.busy = False

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="BitNet Windows Assistant",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Prototype v0.1 — outils de diagnostic locaux, sans installation automatique.",
        ).pack(anchor="w", pady=(2, 12))

        controls = ttk.LabelFrame(outer, text="Vérifications", padding=10)
        controls.pack(fill="x")
        self.doctor_button = ttk.Button(
            controls, text="Diagnostic rapide", command=self.run_doctor
        )
        self.doctor_button.grid(row=0, column=0, padx=(0, 8), pady=4, sticky="ew")
        self.build_button = ttk.Button(
            controls, text="Tester le compilateur C++", command=self.run_build_check
        )
        self.build_button.grid(row=0, column=1, padx=(0, 8), pady=4, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        lib_frame = ttk.LabelFrame(outer, text="Bibliothèque locale", padding=10)
        lib_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(lib_frame, text="Dossier racine choisi :").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.root_var = tk.StringVar(value=str(Path.home() / "BitNetAssistantLibrary"))
        self.root_entry = ttk.Entry(lib_frame, textvariable=self.root_var)
        self.root_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.library_button = ttk.Button(
            lib_frame, text="Initialiser…", command=self.initialize_library
        )
        self.library_button.grid(row=0, column=2, padx=(8, 0), pady=4)
        lib_frame.columnconfigure(1, weight=1)
        ttk.Label(
            lib_frame,
            text="Cette action crée des sous-dossiers à l'emplacement indiqué après confirmation.",
            wraplength=740,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        results_frame = ttk.LabelFrame(outer, text="Résultats", padding=8)
        results_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.output = tk.Text(
            results_frame, wrap="word", height=15, state="disabled",
            font=("Consolas", 9), undo=False,
        )
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.status_var = tk.StringVar(value="Prêt. Aucune action n'a été lancée.")
        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(
            fill="x", pady=(8, 0)
        )
        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(5, 3))
        ttk.Label(
            outer,
            text="Éditeur IA du projet : ChatGPT",
            anchor="e",
            font=("Segoe UI", 8),
        ).pack(fill="x")
        self._append(
            "Bienvenue.\n\n"
            "• Diagnostic rapide : lecture seule.\n"
            "• Test C++ : crée un petit programme temporaire, le compile, l'exécute puis nettoie les fichiers.\n"
            "• Bibliothèque : crée les dossiers proposés après confirmation.\n\n"
            f"Dossier du prototype : {PROJECT_DIR}\n"
        )

    def _append(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text.rstrip() + "\n\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.status_var.set(status)
        state = "disabled" if busy else "normal"
        self.doctor_button.configure(state=state)
        self.build_button.configure(state=state)
        self.library_button.configure(state=state)

    def _run(self, title: str, arguments: list[str]) -> None:
        if self.busy:
            return
        self._set_busy(True, f"En cours : {title}…")
        self._append(f"=== {title} ===\nCommande : {sys.executable} {' '.join(arguments)}")
        threading.Thread(
            target=self._worker, args=(title, arguments), daemon=True
        ).start()

    def _worker(self, title: str, arguments: list[str]) -> None:
        env = os.environ.copy()
        prior = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SOURCE_DIR) + (os.pathsep + prior if prior else "")
        try:
            result = subprocess.run(
                [sys.executable, *arguments],
                cwd=str(PROJECT_DIR),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if stdout:
                try:
                    parsed = json.loads(stdout)
                    stdout = json.dumps(parsed, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            self.root.after(
                0, self._finish, title, result.returncode, stdout, stderr, None
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.root.after(0, self._finish, title, None, "", "", str(exc))

    def _finish(
        self, title: str, code: int | None, stdout: str, stderr: str, error: str | None
    ) -> None:
        if error:
            self._append(f"ERREUR : {error}")
            self._set_busy(False, "Échec de la commande.")
            messagebox.showerror(title, error, parent=self.root)
            return
        chunks = []
        if stdout:
            chunks.append(stdout)
        if stderr:
            chunks.append("[stderr]\n" + stderr)
        chunks.append(f"Code retour : {code}")
        self._append("\n".join(chunks))
        self._set_busy(False, "Terminé." if code == 0 else f"Terminé avec le code {code}.")

    def run_doctor(self) -> None:
        self._run("Diagnostic rapide (lecture seule)", ["-m", "bitnet_windows", "doctor", "--json"])

    def run_build_check(self) -> None:
        confirmed = messagebox.askyesno(
            "Confirmer le test C++",
            "Ce test va compiler, lier et exécuter un petit programme temporaire.\n"
            "Les fichiers temporaires sont ensuite nettoyés. Aucun dépôt ou modèle BitNet ne sera modifié.\n\n"
            "Continuer ?",
            parent=self.root,
        )
        if confirmed:
            self._run(
                "Test C++ temporaire",
                ["-m", "bitnet_windows", "doctor", "--build-check", "--json"],
            )

    def initialize_library(self) -> None:
        try:
            target = parse_library_root(self.root_var.get())
        except ValueError as exc:
            messagebox.showwarning("Chemin invalide", str(exc), parent=self.root)
            return
        confirmed = messagebox.askyesno(
            "Confirmer la création",
            f"Créer les répertoires de la bibliothèque sous :\n\n{target}\n\n"
            "Aucun téléchargement ou logiciel ne sera installé. Continuer ?",
            parent=self.root,
        )
        if confirmed:
            self._run(
                "Initialisation explicite de la bibliothèque",
                ["-m", "bitnet_windows", "init-library", "--root", str(target)],
            )


def main() -> None:
    root = tk.Tk()
    AssistantWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
