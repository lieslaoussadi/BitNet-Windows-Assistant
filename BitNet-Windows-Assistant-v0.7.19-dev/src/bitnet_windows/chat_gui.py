"""Style-C local chat and lexical RAG interface for BitNet Windows Assistant.

Local chat and PDF/text retrieval are connected. OCR/image understanding and web
search remain unimplemented and are never represented as active features.
"""
from __future__ import annotations

import json
import os
import re
import webbrowser
import sys
import threading
import urllib.request
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from . import __version__
from .diagnostics import run_diagnostics, run_cpp_smoke_test
from .gui import parse_library_root
from .library import initialize_library
from .engine import (EngineConfig, EngineError, generate_response, load_config, save_config, suggested_config, validate_config)
from .rag import (DocumentChunk, RAGIndex, RetrievalHit, build_rag_context,
                  extract_faithful_sentence, format_hit_for_display,
                  index_documents, retrieve)
try:
    from .hnsw_backend import HNSW_AVAILABLE
except Exception:
    HNSW_AVAILABLE = False
from .summarize import SummaryError, SummaryResult, build_extractive_summary, estimate_bitnet_summary_calls, summarize_with_bitnet
from .storage import LocalStore, StorageError, default_database_path, default_backup_directory
from .web_search import search_web, format_web_context, format_web_references

APP_TITLE = "BitNet Windows Assistant"
APP_VERSION = __version__

DEFAULT_SYSTEM_PROMPT = "Réponds en français, de manière concise et factuelle. Pour les calculs simples, vérifie implicitement avant de répondre. Si une information n'est pas sûre, dis « Je ne sais pas ». Ne pas inventer date, calcul, ville ou fait."
ANSWER_MODE_GENERATIVE = "BitNet (génératif)"
ANSWER_MODE_EXTRACTIVE = "Extrait fidèle (sans reformulation)"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".txt", ".md", ".rst", ".csv", ".html", ".htm", ".log", ".json"}
SUPPORTED_ATTACHMENT_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | TEXT_SUFFIXES

COLORS = {
    "nav": "#0F172A",
    "nav_alt": "#1E293B",
    "nav_hover": "#334155",
    "blue": "#2563EB",
    "blue_hover": "#1D4ED8",
    "blue_light": "#EFF6FF",
    "purple": "#7C3AED",
    "bg": "#F8FAFC",
    "white": "#FFFFFF",
    "text": "#0F172A",
    "muted": "#64748B",
    "line": "#D1D5DB",
    "success": "#059669",
    "success_bg": "#D1FAE5",
    "warning": "#D97706",
    "warning_bg": "#FEF3C7",
    "assistant_bg": "#FBFCFD",
    "selection": "#2563EB",
}


WELCOME_MESSAGE = (
    "Bienvenue dans BitNet Windows Assistant.\n\n"
    "Le chat et l’index documentaire fonctionnent localement. Utilise Aide pour les commandes, "
    "Docs pour indexer des fichiers, et coche RAG ou Web uniquement lorsque tu en as besoin."
)


def attachment_kind(filename: str | Path) -> str:
    """Return the UI category for a candidate attachment (no parsing performed)."""
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "PDF"
    if suffix in IMAGE_SUFFIXES:
        return "Image"
    if suffix in TEXT_SUFFIXES:
        return "Texte"
    return "Non pris en charge"


def normalize_message(message: str) -> str:
    """Trim user input while preserving internal newlines."""
    return message.strip()



def normalize_clipboard_text(text: str) -> str:
    """Normalize copied text for Windows editors without altering its content."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def looks_like_supported_file_path(text: str) -> bool:
    """Identify a single absolute Windows path to a supported document/image."""
    candidate = text.strip().strip('"').strip()
    if not candidate or "\n" in candidate or "\r" in candidate:
        return False
    is_drive_path = (
        len(candidate) >= 3 and candidate[0].isalpha()
        and candidate[1] == ":" and candidate[2] in ("\\", "/")
    )
    is_unc_path = candidate.startswith("\\\\")
    suffix = Path(candidate.replace("\\", "/")).suffix.lower()
    return (is_drive_path or is_unc_path) and suffix in SUPPORTED_ATTACHMENT_SUFFIXES

def message_display_name(role: str, display_name: str | None = None) -> str:
    """Return a truthful visible label for the message's actual origin."""
    if display_name:
        return display_name
    names = {
        "system": "Information de BitNet Windows Assistant",
        "user": "Vous",
        "assistant": "BitNet 1.58",
    }
    return names.get(role, role)


def append_retrieved_source_references(answer: str, sources: list[RetrievalHit]) -> str:
    """Append deterministic document references when RAG supplied evidence.

    The model is still asked to cite [S1]-style IDs, but this UI-level footer
    guarantees that users can identify the passages actually retrieved even if
    the small model omits inline citations. It is explicitly labelled as the
    retrieval result, not as proof that every source detail was used perfectly.
    """
    cleaned = answer.strip()
    if not sources:
        return cleaned
    references = []
    for hit in sources:
        locator = hit.chunk.locator.strip()
        references.append(f"[{hit.source_id}] {hit.chunk.filename} — {locator}")
    footer = "Références documentaires retrouvées par le RAG : " + " ; ".join(references)
    if footer.casefold() in cleaned.casefold():
        return cleaned
    if not cleaned:
        return footer
    return cleaned + "\n\n" + footer


def append_summary_source_references(answer: str, sources: list[RetrievalHit]) -> str:
    """Append a provenance footer for all document locations included in a summary."""
    cleaned = answer.strip()
    if not sources:
        return cleaned
    references = [f"[{hit.source_id}] {hit.chunk.filename} — {hit.chunk.locator}" for hit in sources]
    footer = "Sources consultées pour ce résumé : " + "; ".join(references)
    if footer.casefold() in cleaned.casefold():
        return cleaned
    return cleaned + ("\n\n" if cleaned else "") + footer


def selected_mode_summary(rag_enabled: bool, web_enabled: bool, *, rag_ready: bool = False) -> str:
    """Describe requested modes without implying unavailable tools were used."""
    selected = []
    if rag_enabled:
        selected.append("RAG local actif" if rag_ready else "RAG local demandé mais index non prêt")
    if web_enabled:
        selected.append("recherche Web activée (résultats transmis au moteur local)")
    if not selected:
        selected.append("chat local demandé")
    return "Modes sélectionnés : " + ", ".join(selected) + ". La recherche Web ne démarre que si sa case est cochée."


def format_diagnostic_report(report: dict[str, Any]) -> str:
    """Format a diagnostic report as readable, self-contained UI text."""
    rows = [f"{report.get('app', APP_TITLE)} — {report.get('mode', 'diagnostic')}"]
    for check in report.get("checks", []):
        rows.append(f"[{str(check.get('status', 'unknown')).upper()}] {check.get('name', 'Vérification')}: {check.get('detail', '')}")
        if check.get("path"):
            rows.append(f"    Chemin : {check['path']}")
    if report.get("notice"):
        rows.append("")
        rows.append(str(report["notice"]))
    return "\n".join(rows)


class ChatAssistantWindow:
    """Desktop UI for the next chat + local RAG milestone."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_TITLE} v{APP_VERSION} — Chat + RAG")
        self.root.geometry("1600x920")
        self.root.minsize(1280, 780)
        self.root.configure(bg=COLORS["bg"])
        self.busy = False
        self.engine_config = load_config()
        self.engine_test_passed = False
        try:
            self.store: LocalStore | None = LocalStore()
            self.persistence_error: str | None = None
        except StorageError as exc:
            self.store = None
            self.persistence_error = str(exc)
        self.current_conversation_id: str | None = None
        self.conversation_ids: list[str] = []
        self.memory_ids: list[str] = []
        self.rag_enabled = tk.BooleanVar(value=False)
        self.answer_mode_var = tk.StringVar(value=ANSWER_MODE_GENERATIVE)
        self.web_enabled = tk.BooleanVar(value=False)
        self.ui_language = "fr"
        self.engine_status_var = tk.StringVar(value="")
        self.clock_status_var = tk.StringVar(value="Heure locale Windows")
        self._network_time_iso: str | None = None
        self.memories_enabled_var = tk.BooleanVar(value=True)
        self.selected_files: list[Path] = []
        self.rag_index = RAGIndex()
        self.last_rag_hits: list[RetrievalHit] = []
        self.history: list[dict[str, str]] = []
        self.last_assistant_message = ""
        self.active_section = "Conversations"
        self.show_recent_var = tk.BooleanVar(value=True)
        self.rag_index_engine_var = tk.StringVar(value="lexical")
        self.planning_mode_var = tk.BooleanVar(value=False)
        self.current_tab = "sources"
        self.theme_var = tk.StringVar(value="light")
        self._build_styles()
        self._build_layout()
        self._restore_persistent_state()
        if self.store is not None:
            try:
                saved_theme = self.store.get_setting("ui_theme", "light")
                self.theme_var.set(saved_theme if saved_theme in ("light", "dark") else "light")
            except Exception:
                self.theme_var.set("light")
        if self.theme_var.get() == "dark":
            self._apply_theme("dark", persist=False)
        self._refresh_mode_status()
        self._update_engine_status()
        self._refresh_clock_status()
        threading.Thread(target=self._fetch_network_time, daemon=True).start()
        self._last_document_status_snapshot = self._current_stale_document_keys()
        # Check source freshness while the Documents tab is visible. The poll
        # is lightweight (file metadata only) and never launches model work.
        self.root.after(2000, self._poll_document_changes)
        # Opt-in only. Delay until the complete UI has been built so startup
        # inference does not block first paint; the CLI runs in the task worker.
        if self.engine_config is not None and self.engine_config.auto_start:
            self.root.after(500, self._start_engine_automatically)

    def _refresh_clock_status(self) -> None:
        """Show the local Windows date/time and whether a network time was obtained."""
        now = datetime.now().astimezone()
        source = "heure réseau vérifiée" if self._network_time_iso else "heure locale Windows"
        self.clock_status_var.set(f"{now.strftime('%d/%m/%Y  %H:%M:%S')}  •  {source}")
        self.root.after(1000, self._refresh_clock_status)

    def _fetch_network_time(self) -> None:
        """Try a lightweight HTTPS Date header; never block startup or change the OS clock."""
        for url in ("https://www.google.com/generate_204", "https://www.microsoft.com/"):
            try:
                request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "BitNetWindowsAssistant/0.5.2"})
                with urllib.request.urlopen(request, timeout=2.5) as response:
                    date_header = response.headers.get("Date")
                if date_header:
                    parsed = parsedate_to_datetime(date_header)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    self._network_time_iso = parsed.astimezone().isoformat(timespec="seconds")
                    self.root.after(0, self._apply_network_time_status)
                    return
            except Exception:
                continue
        # Offline/blocked networks are normal: local Windows time remains authoritative.

    def _apply_network_time_status(self) -> None:
        if self._network_time_iso:
            self.clock_status_var.set(f"{datetime.now().astimezone().strftime('%d/%m/%Y  %H:%M:%S')}  •  réseau accessible")

    def _apply_theme(self, theme: str, *, persist: bool = True) -> None:
        """Apply a best-effort readable light/dark palette to existing Tk widgets."""
        theme = "dark" if theme == "dark" else "light"
        self.theme_var.set(theme)
        light_to_dark = {
            "#F1F5F9": "#111827", "#FFFFFF": "#1F2937", "#F8FAFC": "#273449",
            "#0F172A": "#F3F4F6", "#64748B": "#B6C2D2", "#D1D5DB": "#374151",
            "#EFF6FF": "#23395D", "#D1FAE5": "#174B38", "#FEF3C7": "#5B4218",
            "#2563EB": "#60A5FA",
        }
        dark_to_light = {v: k for k, v in light_to_dark.items()}
        mapping = light_to_dark if theme == "dark" else dark_to_light
        fg_map = {"#0F172A": "#F3F4F6", "#64748B": "#B6C2D2", "#374151": "#E5E7EB"} if theme == "dark" else {"#F3F4F6": "#0F172A", "#B6C2D2": "#64748B", "#E5E7EB": "#374151"}
        def recolor(widget):
            try:
                opts = widget.configure()
                for key in ("background", "bg", "highlightbackground", "insertbackground", "selectcolor"):
                    if key in opts:
                        old = str(widget.cget(key)).upper()
                        if old in mapping:
                            widget.configure(**{key: mapping[old]})
                for key in ("foreground", "fg", "highlightcolor"):
                    if key in opts:
                        old = str(widget.cget(key)).upper()
                        if old in fg_map:
                            widget.configure(**{key: fg_map[old]})
                if isinstance(widget, tk.Text):
                    widget.configure(insertbackground="#F9FAFB" if theme == "dark" else "#16223B")
                if isinstance(widget, tk.Listbox):
                    widget.configure(selectbackground="#315A9D" if theme == "dark" else "#2165F5", selectforeground="#FFFFFF")
            except (tk.TclError, TypeError):
                pass
            for child in widget.winfo_children():
                recolor(child)
        recolor(self.root)
        # Tk Text tag backgrounds are independent of the widget background. Reapply
        # a coordinated palette explicitly so dark mode never leaves white message
        # cards surrounded by a dark transcript gutter.
        chat_text = getattr(self, "chat_text", None)
        if chat_text is not None:
            try:
                if theme == "dark":
                    chat_text.configure(bg="#1F2937", fg="#F3F4F6", selectbackground="#315A9D", selectforeground="#FFFFFF", inactiveselectbackground="#315A9D")
                    chat_text.tag_configure("system_name", foreground="#A78BFA")
                    chat_text.tag_configure("system_body", foreground="#D1D9E6", background="#1F2937")
                    chat_text.tag_configure("user_name", foreground="#60A5FA")
                    chat_text.tag_configure("user_body", foreground="#F3F4F6", background="#1F2937")
                    chat_text.tag_configure("assistant_name", foreground="#F3F4F6")
                    chat_text.tag_configure("assistant_body", foreground="#F3F4F6", background="#1F2937")
                    chat_text.tag_configure("web_link", foreground="#93C5FD")
                    chat_text.tag_configure("sel", background="#315A9D", foreground="#FFFFFF")
                else:
                    chat_text.configure(bg="#FFFFFF", fg="#16223B", selectbackground="#245FDB", selectforeground="#FFFFFF", inactiveselectbackground="#AFC8FA")
                    chat_text.tag_configure("system_name", foreground="#7357D8")
                    chat_text.tag_configure("system_body", foreground="#687895", background="#FFFFFF")
                    chat_text.tag_configure("user_name", foreground="#2165F5")
                    chat_text.tag_configure("user_body", foreground="#16223B", background="#FFFFFF")
                    chat_text.tag_configure("assistant_name", foreground="#16223B")
                    chat_text.tag_configure("assistant_body", foreground="#16223B", background="#FFFFFF")
                    chat_text.tag_configure("web_link", foreground="#2165F5")
                    chat_text.tag_configure("sel", background="#245FDB", foreground="#FFFFFF")
            except tk.TclError:
                pass
        try:
            style = ttk.Style(self.root)
            bg = "#1F2937" if theme == "dark" else "#FFFFFF"
            fg = "#F3F4F6" if theme == "dark" else "#16223B"
            style.configure("TCombobox", fieldbackground=bg, background=bg, foreground=fg)
            style.configure("TCheckbutton", background=bg, foreground=fg)
            style.configure("TScrollbar", background=bg, troughcolor="#111827" if theme == "dark" else "#EEF2F8")
        except tk.TclError:
            pass
        if persist and self.store is not None:
            try:
                self.store.set_setting("ui_theme", theme)
            except Exception as exc:
                self.status_var.set(f"Thème appliqué, mais préférence non enregistrée : {exc}")
        if hasattr(self, "status_var"):
            self.status_var.set("Thème sombre activé." if theme == "dark" else "Thème clair activé.")

    def _toggle_theme(self) -> None:
        self._apply_theme("dark" if self.theme_var.get() != "dark" else "light")

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TSeparator", background=COLORS["line"])

    def _label(self, parent: tk.Widget, text: str, *, bg: str | None = None,
               fg: str | None = None, font: tuple[Any, ...] = ("Segoe UI", 10),
               **kwargs: Any) -> tk.Label:
        return tk.Label(parent, text=text, bg=bg or COLORS["white"],
                        fg=fg or COLORS["text"], font=font, **kwargs)

    def _button(self, parent: tk.Widget, text: str, command: Callable[[], None], *,
                primary: bool = False, compact: bool = False) -> tk.Button:
        """Create a button using one restrained accent color throughout the app."""
        bg = COLORS["blue"]
        fg = "#FFFFFF"
        active_bg = COLORS["blue_hover"]
        return tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=active_bg, activeforeground=fg,
            relief="flat", bd=0, cursor="hand2", padx=10 if compact else 14,
            pady=7 if compact else 9, font=("Segoe UI", 9 if compact else 10, "bold"),
            highlightthickness=0, highlightbackground=bg, highlightcolor=COLORS["blue_light"],
            overrelief="flat",
        )

    def _build_layout(self) -> None:
        self.sidebar = tk.Frame(self.root, bg=COLORS["nav"], width=244)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        self.main = tk.Frame(self.root, bg=COLORS["bg"])
        self.main.pack(side="left", fill="both", expand=True)
        self._build_topbar()

        self.work_area = tk.Frame(self.main, bg=COLORS["bg"])
        self.work_area.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._build_documents_panel()
        self._build_chat_panel()

    def _build_sidebar(self) -> None:
        brand = tk.Frame(self.sidebar, bg=COLORS["nav"], padx=12, pady=12)
        brand.pack(fill="x")
        self._label(brand, "◈", bg=COLORS["nav"], fg="#65A4FF", font=("Segoe UI Symbol", 25, "bold")).pack(side="left", padx=(0, 10))
        brand_text = tk.Frame(brand, bg=COLORS["nav"])
        brand_text.pack(side="left", fill="x", expand=True)
        self._label(brand_text, "BitNet", bg=COLORS["nav"], fg="white", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self._label(brand_text, "ASSISTANT", bg=COLORS["nav"], fg="#A8B9D9", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 0))


        self._button(self.sidebar, "+  Nouvelle session", self.new_conversation, primary=True).pack(fill="x", padx=14, pady=(2, 14))
        nav = tk.Frame(self.sidebar, bg=COLORS["nav"])
        nav.pack(fill="x", padx=8)
        # Conversation switching is handled by the dedicated Sessions picker;
        # memories are already available in the right-hand document panel.
        nav_items = [
            ("▣  RAG", "RAG"),
            ("⚙  Paramètres", "Paramètres"),
            ("⌁  Diagnostics", "Diagnostics"),
            ("ⓘ  À propos", "À propos"),
        ]
        self.nav_buttons: dict[str, tk.Button] = {}
        for visible, key in nav_items:
            button = tk.Button(
                nav, text=visible, anchor="w", command=lambda k=key: self.navigate(k),
                bg=COLORS["nav"], fg="#E8EFFC", activebackground=COLORS["nav_hover"],
                activeforeground="white", relief="flat", bd=0, cursor="hand2",
                padx=12, pady=5, font=("Segoe UI", 10),
            )
            button.pack(fill="x", pady=1)
            self.nav_buttons[key] = button
        self._set_active_nav("")

        tk.Frame(self.sidebar, bg="#263A5D", height=1).pack(fill="x", padx=16, pady=(10, 8))
        self.recent_section = tk.Frame(self.sidebar, bg=COLORS["nav"])
        self.recent_section.pack(fill="x")
        self.recent_heading = self._label(self.recent_section, "CHATS RÉCENTS", bg=COLORS["nav"], fg="#94A8CD", font=("Segoe UI", 8, "bold"))
        self.recent_heading.pack(anchor="w", padx=18, pady=(0, 6))
        search_row = tk.Frame(self.recent_section, bg=COLORS["nav"])
        search_row.pack(fill="x", padx=12, pady=(0, 4))
        self.session_search_var = tk.StringVar(value="")
        self.session_search_entry = tk.Entry(
            search_row, textvariable=self.session_search_var, bg="#172947", fg="#F1F5FC",
            insertbackground="#F1F5FC", relief="flat", bd=0, highlightthickness=0,
            highlightbackground="#3A5278", font=("Segoe UI", 9),
        )
        self.session_search_entry.pack(fill="x", padx=2, pady=2, ipady=4)
        self.session_search_entry.bind("<KeyRelease>", lambda _event: self._refresh_conversation_list())
        self.session_search_entry.insert(0, "Rechercher…")
        self.session_search_entry.bind("<FocusIn>", self._clear_search_placeholder)
        self.session_search_entry.bind("<FocusOut>", self._restore_search_placeholder)

        recent_wrap = tk.Frame(self.recent_section, bg=COLORS["nav"])
        recent_wrap.pack(fill="x", padx=12, pady=(0, 4))
        self.conversation_list = tk.Listbox(
            recent_wrap, height=7, bg="#172947", fg="#F1F5FC",
            selectbackground=COLORS["blue"], selectforeground="white",
            font=("Segoe UI", 10), relief="flat", bd=0, highlightthickness=0,
            highlightbackground="#3A5278", highlightcolor="#14B8A6",
            activestyle="none", exportselection=False,
        )
        self.conversation_list.pack(side="left", fill="both", expand=True)
        recent_scroll = tk.Scrollbar(recent_wrap, command=self.conversation_list.yview)
        recent_scroll.pack(side="right", fill="y")
        self.conversation_list.configure(yscrollcommand=recent_scroll.set)
        self.conversation_list.bind("<<ListboxSelect>>", self._on_conversation_list_select)
        self.conversation_list.bind("<Button-3>", self._show_conversation_context_menu)
        self._button(self.recent_section, "Supprimer la session", self.delete_selected_conversation, compact=True).pack(fill="x", padx=12, pady=(0, 4))
        self.conversation_context_menu = tk.Menu(self.root, tearoff=False)
        self.conversation_context_menu.add_command(
            label="Supprimer définitivement cette conversation",
            command=self.delete_selected_conversation,
        )
        self.recent_label = self._label(
            self.recent_section, "Les conversations sont enregistrées localement sur ce PC.",
            bg=COLORS["nav"], fg="#B8C5DC", font=("Segoe UI", 8),
            justify="left", wraplength=205,
        )
        self.recent_label.pack(anchor="w", padx=18, pady=(3, 6))
        self.sessions_button = self._button(self.sidebar, "Sessions (afficher/masquer)", self._toggle_sessions_panel, compact=True)
        self.sessions_button.pack(fill="x", padx=14, pady=(0, 6))
        spacer = tk.Frame(self.sidebar, bg=COLORS["nav"])
        spacer.pack(fill="both", expand=True)
        self._button(self.sidebar, "?  Aide", self.show_help, compact=True).pack(fill="x", padx=14, pady=(0, 7))
        tk.Frame(self.sidebar, bg="#263A5D", height=1).pack(fill="x", padx=16, pady=(6, 6))
        self._label(self.sidebar, "Éditeur IA du projet", bg=COLORS["nav"], fg="#B8C5DC", font=("Segoe UI", 8)).pack(anchor="w", padx=18)
        self._label(self.sidebar, "ChatGPT", bg=COLORS["nav"], fg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=18)
        self._label(self.sidebar, "Développé avec Beam et ChatGPT", bg=COLORS["nav"], fg="#A8B9D9", font=("Segoe UI", 8), wraplength=205, justify="left").pack(anchor="w", padx=18, pady=(0, 2))
        self._label(self.sidebar, "© 2026 — BitNet Windows Assistant", bg=COLORS["nav"], fg="#7F94B7", font=("Segoe UI", 7)).pack(anchor="w", padx=18, pady=(0, 7))

    def _build_topbar(self) -> None:
        top = tk.Frame(self.main, bg=COLORS["white"], height=80, highlightbackground=COLORS["line"], highlightthickness=0)
        top.pack(fill="x", padx=0, pady=(0, 12))
        top.pack_propagate(False)
        left = tk.Frame(top, bg=COLORS["white"])
        left.pack(side="left", fill="y", padx=20, pady=12)
        self.engine_status_label = tk.Label(left, textvariable=self.engine_status_var, bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 8), wraplength=230, justify="left")
        self.engine_status_label.pack(anchor="w", pady=(0, 4))
        self.clock_status_label = tk.Label(left, textvariable=self.clock_status_var, bg=COLORS["white"], fg=COLORS["blue"], font=("Segoe UI", 8, "bold"))
        self.clock_status_label.pack(anchor="w", pady=(2, 0))

        pills = tk.Frame(top, bg=COLORS["white"])
        pills.pack(side="left", fill="y", padx=(10, 4), pady=17)
        self.model_status = self._label(pills, "Cible : BitNet 1.58", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 9, "bold"), padx=10, pady=7)
        self.model_status.pack(side="left", padx=(0, 6))
        self.engine_status = self._label(pills, "Moteur non connecté", bg=COLORS["warning_bg"], fg=COLORS["warning"], font=("Segoe UI", 9, "bold"), padx=10, pady=7)
        self.engine_status.pack(side="left", padx=(0, 6))
        self.resource_status = self._label(pills, "CPU · 8 threads · BitNet 1.58", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 10), padx=10, pady=7)
        self.resource_status.pack(side="left")

        actions = tk.Frame(top, bg=COLORS["white"])
        actions.pack(side="right", fill="y", padx=12, pady=14)
        # The engine has one primary shortcut in the top bar. Diagnostics and
        # About live in the sidebar to avoid duplicate navigation targets.
        self._button(actions, "Moteur et modèles…", self.configure_engine, compact=True).pack(side="left", padx=3)

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg=COLORS["white"], highlightbackground=COLORS["line"], highlightthickness=0)

    def _build_chat_panel(self) -> None:
        center = tk.Frame(self.work_area, bg=COLORS["bg"])
        center.pack(side="left", fill="both", expand=True)
        self.chat_card = self._card(center)
        self.chat_card.pack(fill="both", expand=True)
        self.chat_card.grid_columnconfigure(0, weight=1)
        self.chat_card.grid_rowconfigure(2, weight=1, minsize=360)
        self.chat_card.grid_rowconfigure(3, weight=0)

        chat_header = tk.Frame(self.chat_card, bg=COLORS["white"], padx=16, pady=12)
        chat_header.grid(row=0, column=0, sticky="ew")
        header_left = tk.Frame(chat_header, bg=COLORS["white"])
        header_left.pack(side="left", fill="x", expand=True)
        self.conversation_title_var = tk.StringVar(value="Discussion générale")
        tk.Label(header_left, textvariable=self.conversation_title_var, bg=COLORS["white"], fg=COLORS["text"], font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.connection_summary_var = tk.StringVar(value="Configure le moteur local pour commencer à discuter.")
        tk.Label(header_left, textvariable=self.connection_summary_var, bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))
        header_actions = tk.Frame(chat_header, bg=COLORS["white"])
        header_actions.pack(side="right")
        self._button(header_actions, "Copier la réponse", self.copy_last_assistant_response, compact=True).pack(side="left", padx=(0, 5))
        self._button(header_actions, "Effacer", self.clear_conversation, compact=True).pack(side="left", padx=(0, 5))
        self._button(header_actions, "Exporter", self.export_conversation, compact=True).pack(side="left")
        tk.Frame(self.chat_card, bg=COLORS["line"], height=1).grid(row=1, column=0, sticky="ew")

        text_wrap = tk.Frame(self.chat_card, bg=COLORS["white"], padx=12, pady=10)
        text_wrap.grid(row=2, column=0, sticky="nsew")
        self.chat_text = tk.Text(
            text_wrap, wrap="word", state="normal", bg=COLORS["white"], fg=COLORS["text"],
            relief="flat", bd=0, padx=10, pady=8, font=("Segoe UI", 10),
            spacing1=3, spacing3=5, cursor="xterm", takefocus=True,
        )
        scrollbar = ttk.Scrollbar(text_wrap, orient="vertical", command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=scrollbar.set)
        self.chat_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.chat_text.tag_configure("system_name", foreground=COLORS["purple"], font=("Segoe UI", 10, "bold"), spacing1=14, spacing3=3)
        self.chat_text.tag_configure("system_body", foreground=COLORS["muted"], background=COLORS["white"], lmargin1=8, lmargin2=8, rmargin=8, spacing1=3, spacing3=12)
        self.chat_text.tag_configure("user_name", foreground=COLORS["blue"], font=("Segoe UI", 10, "bold"), justify="right", spacing1=12, spacing3=3)
        self.chat_text.tag_configure("user_body", foreground=COLORS["text"], background=COLORS["white"], lmargin1=10, lmargin2=10, rmargin=10, spacing1=3, spacing3=10)
        self.chat_text.tag_configure("assistant_name", foreground=COLORS["text"], font=("Segoe UI", 10, "bold"), spacing1=12, spacing3=3)
        self.chat_text.tag_configure("assistant_body", foreground=COLORS["text"], background=COLORS["white"], lmargin1=10, lmargin2=10, rmargin=10, spacing1=3, spacing3=10)
        # Message-body tags have their own backgrounds; raise the native selection
        # tag above them so the selected passage is visibly highlighted on Windows.
        self.chat_text.configure(
            selectbackground=COLORS["selection"], selectforeground="#FFFFFF",
            inactiveselectbackground="#AFC8FA",
        )
        self.chat_text.tag_configure("sel", background=COLORS["selection"], foreground="#FFFFFF")
        self.chat_text.tag_configure("web_link", foreground=COLORS["blue"], underline=True)
        self.chat_text.tag_bind("web_link", "<Button-1>", self._open_chat_link)
        self.chat_text.tag_bind("web_link", "<Enter>", lambda _event: self.chat_text.configure(cursor="hand2"))
        self.chat_text.tag_bind("web_link", "<Leave>", lambda _event: self.chat_text.configure(cursor=""))
        self.chat_text.tag_raise("sel")
        # Keyboard and mouse clipboard support for the read-only transcript.
        self.chat_text.bind("<Button-3>", self._show_chat_context_menu)
        self.chat_text.bind("<Control-c>", self._copy_selection_shortcut)
        self.chat_text.bind("<Control-C>", self._copy_selection_shortcut)
        self.chat_text.bind("<Control-a>", self._select_all_chat_shortcut)
        self.chat_text.bind("<Control-A>", self._select_all_chat_shortcut)
        self.chat_text.bind("<KeyPress>", self._protect_chat_transcript_edit)
        self.chat_text.bind("<Control-v>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-V>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-x>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-X>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-z>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-Z>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-y>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-Y>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Control-Insert>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<BackSpace>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Delete>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Return>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<KP_Enter>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Tab>", self._block_chat_edit_shortcut)
        self.chat_text.bind("<Insert>", self._block_chat_edit_shortcut)
        self.chat_context_menu = tk.Menu(self.root, tearoff=False)
        self.chat_context_menu.add_command(label="Copier la sélection", command=self.copy_chat_selection)
        self.chat_context_menu.add_command(label="Copier la dernière réponse", command=self.copy_last_assistant_response)
        self.chat_context_menu.add_command(label="Copier toute la discussion", command=self.copy_entire_chat)
        self.chat_context_menu.add_separator()
        self.chat_context_menu.add_command(label="Créer un souvenir depuis la sélection…", command=self.save_selection_as_memory)
        self.chat_context_menu.add_separator()
        self.chat_context_menu.add_command(label="Tout sélectionner", command=self.select_all_chat)

        divider = tk.Frame(self.chat_card, bg=COLORS["line"], height=1)
        divider.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        composer = tk.Frame(self.chat_card, bg=COLORS["white"], highlightbackground=COLORS["line"], highlightthickness=0)
        composer.grid(row=4, column=0, sticky="ew", pady=(8, 12))
        self.message_entry = tk.Text(composer, height=3, wrap="word", font=("Segoe UI", 10), bg=COLORS["white"], fg=COLORS["text"], relief="flat", bd=0, padx=14, pady=12, undo=True)
        self.message_entry.pack(fill="x", padx=2, pady=(2, 0))
        self._label(
            composer, "Entrée : envoyer  •  Maj+Entrée : nouvelle ligne",
            fg=COLORS["muted"], font=("Segoe UI", 8), anchor="e",
        ).pack(fill="x", padx=14, pady=(0, 4))
        self.message_entry.insert("1.0", "Écrivez votre message ici…")
        self.message_entry.configure(fg="#8B98AE")
        self.message_entry.bind("<FocusIn>", self._composer_focus_in)
        self.message_entry.bind("<FocusOut>", self._composer_focus_out)
        self.message_entry.bind("<Return>", self._composer_return_shortcut)
        self.message_entry.bind("<KP_Enter>", self._composer_return_shortcut)
        self.message_entry.bind("<Control-Return>", self._send_shortcut)
        self.message_entry.bind("<Control-v>", self._paste_shortcut)
        self.message_entry.bind("<Control-V>", self._paste_shortcut)
        self.message_entry.bind("<Shift-Insert>", self._paste_shortcut)
        self.message_entry.bind("<Control-c>", self._copy_composer_shortcut)
        self.message_entry.bind("<Control-C>", self._copy_composer_shortcut)
        self.message_entry.bind("<Control-x>", self._cut_composer_shortcut)
        self.message_entry.bind("<Control-X>", self._cut_composer_shortcut)
        self.message_entry.bind("<Button-3>", self._show_composer_context_menu)
        self.composer_context_menu = tk.Menu(self.root, tearoff=False)
        self.composer_context_menu.add_command(label="Couper", command=self.cut_composer_selection)
        self.composer_context_menu.add_command(label="Copier", command=self.copy_composer_selection)
        self.composer_context_menu.add_command(label="Coller", command=self.paste_into_composer)
        self.composer_context_menu.add_separator()
        self.composer_context_menu.add_command(label="Tout sélectionner", command=self.select_all_composer)
        tk.Frame(composer, bg=COLORS["line"], height=1).pack(fill="x", padx=10)

        controls = tk.Frame(composer, bg=COLORS["white"], padx=10, pady=8)
        controls.pack(fill="x")
        self._button(controls, "＋ Joindre PDF / image / texte", self.choose_files, compact=True).pack(side="left", padx=(0, 4))
        self.rag_check = tk.Checkbutton(
            controls, text="RAG", variable=self.rag_enabled, command=self._refresh_mode_status,
            bg=COLORS["white"], activebackground=COLORS["white"], fg=COLORS["text"],
            selectcolor=COLORS["blue_light"], font=("Segoe UI", 10), cursor="hand2",
        )
        self.rag_check.pack(side="left", padx=5)
        self.web_check = tk.Checkbutton(
            controls, text="Web", variable=self.web_enabled, command=self._refresh_mode_status,
            bg=COLORS["white"], activebackground=COLORS["white"], fg=COLORS["text"],
            selectcolor=COLORS["blue_light"], font=("Segoe UI", 10), cursor="hand2",
        )
        self.web_check.pack(side="left", padx=5)
        self.plan_check = tk.Checkbutton(
            controls,
            text="Plan",
            variable=self.planning_mode_var,
            command=self._save_planning_mode,
            bg=COLORS["white"],
            activebackground=COLORS["white"],
            fg=COLORS["text"],
            selectcolor=COLORS["blue_light"],
            font=("Segoe UI", 10),
            cursor="hand2",
        )
        self.plan_check.pack(side="left", padx=5)
        self.send_button = self._button(controls, "➤ Envoyer", self.send_message, primary=True, compact=True)
        self.send_button.configure(cursor="hand2")
        self.send_button.pack(side="right", padx=(5, 0))

        self.mode_note_var = tk.StringVar(value="RAG : désactivé • Web : désactivé")
        tk.Label(composer, textvariable=self.mode_note_var, bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=14, pady=(0, 3))
        self.status_var = tk.StringVar(value="Prêt. Moteur de chat non connecté.")
        self.status_bar = tk.Label(center, textvariable=self.status_var, bg=COLORS["bg"], fg=COLORS["muted"], anchor="w", font=("Segoe UI", 8), padx=3, pady=5)
        self.status_bar.pack(fill="x", pady=(2, 0))

    def _build_documents_panel(self) -> None:
        panel = tk.Frame(self.work_area, bg=COLORS["bg"], width=310)
        panel.pack(side="right", fill="y", padx=(12, 0))
        panel.pack_propagate(False)
        self.doc_card = self._card(panel)
        self.doc_card.pack(fill="both", expand=True)

        title_bar = tk.Frame(self.doc_card, bg=COLORS["white"], padx=12, pady=12)
        title_bar.pack(fill="x")
        self._label(title_bar, "▣  Sources et documents", font=("Segoe UI", 11, "bold")).pack(side="left")
        self._button(title_bar, "+", self.choose_files, primary=True, compact=True).pack(side="right")
        tabs = tk.Frame(self.doc_card, bg=COLORS["white"], padx=8)
        tabs.pack(fill="x")
        self.sources_tab_button = self._button(tabs, "Sources", lambda: self.show_document_tab("sources"), compact=True)
        self.sources_tab_button.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.files_tab_button = self._button(tabs, "Docs", lambda: self.show_document_tab("documents"), compact=True)
        self.files_tab_button.pack(side="left", fill="x", expand=True, padx=2)
        self.memories_tab_button = self._button(tabs, "Mém.", lambda: self.show_document_tab("memories"), compact=True)
        self.memories_tab_button.pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Frame(self.doc_card, bg=COLORS["line"], height=1).pack(fill="x", padx=10, pady=(8, 0))
        mode_box = tk.Frame(self.doc_card, bg=COLORS["bg"], padx=10, pady=8, highlightbackground=COLORS["blue"], highlightthickness=0)
        mode_box.pack(fill="x", padx=8, pady=(8, 0))
        self._label(mode_box, "Mode pour la prochaine réponse RAG", bg=COLORS["bg"], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        self.answer_mode_combo = ttk.Combobox(
            mode_box, textvariable=self.answer_mode_var, state="readonly", width=29,
            values=(ANSWER_MODE_GENERATIVE, ANSWER_MODE_EXTRACTIVE),
        )
        self.answer_mode_combo.pack(fill="x")
        self.answer_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_mode_status())

        self.sources_frame = tk.Frame(self.doc_card, bg=COLORS["white"], padx=13, pady=12)
        self.sources_frame.pack(fill="both", expand=True)
        self.sources_heading = self._label(self.sources_frame, "Aucune source utilisée", font=("Segoe UI", 10, "bold"))
        self.sources_heading.pack(anchor="w", pady=(2, 8))
        self.sources_info = self._label(self.sources_frame, "Les passages retrouvés apparaissent ici lorsque RAG est activé.", fg=COLORS["muted"], font=("Segoe UI", 10), wraplength=255, justify="left")
        self.sources_info.pack(anchor="w", pady=(0, 8))
        # Keep the heading and explanation fixed while allowing the source cards
        # to scroll independently inside the right-hand panel.
        sources_list = tk.Frame(self.sources_frame, bg=COLORS["white"])
        sources_list.pack(fill="both", expand=True, anchor="n")
        self.sources_canvas = tk.Canvas(
            sources_list, bg=COLORS["white"], highlightthickness=0, borderwidth=0,
        )
        self.sources_scrollbar = ttk.Scrollbar(
            sources_list, orient="vertical", command=self.sources_canvas.yview,
        )
        self.sources_canvas.configure(yscrollcommand=self.sources_scrollbar.set)
        self.sources_scrollbar.pack(side="right", fill="y")
        self.sources_canvas.pack(side="left", fill="both", expand=True)
        self.sources_cards = tk.Frame(self.sources_canvas, bg=COLORS["white"])
        self._sources_canvas_window = self.sources_canvas.create_window(
            (0, 0), window=self.sources_cards, anchor="nw",
        )
        self.sources_cards.bind("<Configure>", self._update_sources_scrollregion)
        self.sources_canvas.bind("<Configure>", self._resize_sources_canvas_window)
        self.sources_canvas.bind("<Enter>", self._bind_sources_mousewheel)
        self.sources_canvas.bind("<Leave>", self._unbind_sources_mousewheel)

        self.documents_frame = tk.Frame(self.doc_card, bg=COLORS["white"], padx=10, pady=10)
        self.documents_heading = self._label(self.documents_frame, "Fichiers sélectionnés (en attente d'indexation)", fg=COLORS["muted"], font=("Segoe UI", 9, "bold"))
        self.documents_heading.pack(anchor="w", pady=(0, 7))
        self.documents_list = tk.Listbox(self.documents_frame, bg="#FAFBFE", fg=COLORS["text"], selectbackground=COLORS["blue"], selectforeground="white", font=("Segoe UI", 10), relief="flat", highlightthickness=0, highlightbackground=COLORS["line"], activestyle="none")
        self.documents_list.pack(fill="both", expand=True)
        doc_actions = tk.Frame(self.documents_frame, bg=COLORS["white"])
        doc_actions.pack(fill="x", pady=(8, 0))
        self._button(doc_actions, "Indexer la sélection", self.index_selected_documents, primary=True, compact=True).pack(side="left")
        self._button(doc_actions, "Supprimer l’index", self.remove_selected_files, compact=True).pack(side="right")

        summary_actions = tk.Frame(self.documents_frame, bg=COLORS["white"])
        summary_actions.pack(fill="x", pady=(8, 0))
        self._button(summary_actions, "Résumé rapide (sans modèle)", self.summarize_selected_extractive, compact=True).pack(fill="x", pady=(0, 4))
        self._button(summary_actions, "Résumer avec BitNet…", self.summarize_selected_with_bitnet, compact=True).pack(fill="x")
        self._label(self.documents_frame, "Sélectionne un document indexé ci-dessus. Le résumé rapide reprend des phrases originales sans appel au modèle ; le résumé BitNet peut lancer plusieurs générations locales et être plus lent. Aucun fichier n'est modifié ni envoyé sur Internet.", fg=COLORS["muted"], font=("Segoe UI", 8), wraplength=255, justify="left").pack(anchor="w", pady=(9, 2))

        self.memories_frame = tk.Frame(self.doc_card, bg=COLORS["white"], padx=10, pady=10)
        self._label(self.memories_frame, "Souvenirs locaux", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 3))
        self._label(
            self.memories_frame,
            "Ajoute volontairement une préférence, un fait ou une information utile. Les souvenirs sont modifiables et supprimables ; rien n'est mémorisé automatiquement.",
            fg=COLORS["muted"], font=("Segoe UI", 8), wraplength=260, justify="left",
        ).pack(anchor="w", pady=(0, 7))
        self.memories_use_check = tk.Checkbutton(
            self.memories_frame, text="Utiliser les souvenirs actifs dans le chat",
            variable=self.memories_enabled_var, command=self._set_memories_usage,
            bg=COLORS["white"], activebackground=COLORS["white"], fg=COLORS["text"],
            selectcolor=COLORS["blue_light"], font=("Segoe UI", 8), cursor="hand2",
            wraplength=260, justify="left",
        )
        self.memories_use_check.pack(anchor="w", pady=(0, 7))
        self.memories_list = tk.Listbox(
            self.memories_frame, bg="#FAFBFE", fg=COLORS["text"],
            selectbackground=COLORS["blue"], selectforeground="white",
            font=("Segoe UI", 10), relief="flat", highlightthickness=0,
            highlightbackground=COLORS["line"], activestyle="none", exportselection=False,
        )
        self.memories_list.pack(fill="both", expand=True)
        memory_actions = tk.Frame(self.memories_frame, bg=COLORS["white"])
        memory_actions.pack(fill="x", pady=(8, 0))
        self._button(memory_actions, "+ Ajouter", self.add_memory, primary=True, compact=True).pack(side="left", padx=(0, 3))
        self._button(memory_actions, "Modifier", self.edit_selected_memory, compact=True).pack(side="left", padx=3)
        self._button(memory_actions, "Activer/désactiver", self.toggle_selected_memory, compact=True).pack(side="left", padx=3)
        self._button(self.memories_frame, "Supprimer le souvenir sélectionné", self.delete_selected_memory, compact=True).pack(fill="x", pady=(5, 0))
        self._label(
            self.memories_frame,
            "Astuce : sélectionne un passage de la discussion, puis clic droit → « Créer un souvenir depuis la sélection… ».",
            fg=COLORS["muted"], font=("Segoe UI", 8), wraplength=260, justify="left",
        ).pack(anchor="w", pady=(8, 0))

        pipeline_box = tk.LabelFrame(
            self.documents_frame, text="Pipeline local", bg=COLORS["white"], fg=COLORS["text"],
            font=("Segoe UI", 9, "bold"), padx=7, pady=5,
        )
        pipeline_box.pack(fill="x", pady=(8, 0))
        self._label(
            pipeline_box,
            "PDF textuels et fichiers texte → extraction → index local → recherche RAG.\n"
            "OCR des images/PDF scannés non disponible. Recherche Web uniquement sur demande.",
            fg=COLORS["muted"], font=("Segoe UI", 8), wraplength=245, justify="left",
        ).pack(anchor="w", fill="x")

        self.show_document_tab("sources")

    def _set_active_nav(self, section: str) -> None:
        for key, button in self.nav_buttons.items():
            selected = key == section
            button.configure(bg=COLORS["nav_hover"] if selected else COLORS["nav"], fg="white" if selected else "#E8EFFC")

    def _restore_persistent_state(self) -> None:
        """Restore the last local conversation, saved index and explicit memory settings."""
        if self.store is None:
            self.status_var.set("Stockage local indisponible : " + str(self.persistence_error))
            self.recent_label.configure(text="La sauvegarde locale est indisponible. Les données de cette session ne seront pas conservées.")
            return

        try:
            self.memories_enabled_var.set(self.store.get_setting("memories_enabled", "1") != "0")
            self.show_recent_var.set(self.store.get_setting("show_recent_chats", "1") != "0")
            self.planning_mode_var.set(self.store.get_setting("planning_mode", "0") == "1")
            saved_engine = self.store.get_setting("rag_index_engine", "lexical")
            if saved_engine not in ("lexical", "hnsw"):
                saved_engine = "lexical"
            self.rag_index_engine_var.set(saved_engine)
        except StorageError as exc:
            self.status_var.set(str(exc))

        try:
            self.rag_index = self.store.load_rag_index()
            restored_paths = list(dict.fromkeys(self.rag_index.indexed_files + self.rag_index.stale_files))
            self.selected_files = [Path(path) for path in restored_paths]
            self._refresh_documents_list()
            if self.rag_index.chunk_count:
                self.status_var.set(
                    f"Index documentaire restauré : {self.rag_index.document_count} document(s), "
                    f"{self.rag_index.chunk_count} passage(s)."
                )
            elif self.rag_index.warnings:
                self.status_var.set("Index documentaire partiellement obsolète : réindexe les fichiers concernés.")

            if not self.rag_index.chunks:
                default_chunks = self._load_default_rag_chunks()
                if default_chunks:
                    self.rag_index = RAGIndex(
                        chunks=default_chunks,
                        indexed_files=["Base documentaire interne"],
                        warnings=[],
                        stale_files=[],
                    )
                    self._refresh_documents_list()
                    self.status_var.set(
                        f"Base de référence chargée : {self.rag_index.document_count} document(s), "
                        f"{self.rag_index.chunk_count} passage(s)."
                    )
            if self.rag_index_engine_var.get() == "hnsw" and self.rag_index.chunks:
                try:
                    from .hnsw_backend import HNSW_AVAILABLE, HnswIndex
                    if HNSW_AVAILABLE:
                        candidate = HnswIndex()
                        if candidate.build(self.rag_index.chunks):
                            self.rag_index.hnsw_index = candidate
                        else:
                            self.rag_index.hnsw_index = None
                except Exception:
                    self.rag_index.hnsw_index = None
        except StorageError as exc:
            self.rag_index = RAGIndex(warnings=[str(exc)])
            self.selected_files = []
            self.status_var.set(str(exc))

        self._refresh_memories_list()
        try:
            conversations = self.store.list_conversations(limit=100)
            if conversations:
                self._load_conversation(str(conversations[0]["id"]), refresh_list=False)
            else:
                self.current_conversation_id = self.store.create_conversation("Discussion générale")
                self.conversation_title_var.set("Discussion générale")
                
            self._refresh_conversation_list()
            self._apply_recent_visibility()
            if self.persistence_error is None:
                self.recent_label.configure(text="Conversations et souvenirs stockés localement sur ce PC.")
        except StorageError as exc:
            self.persistence_error = str(exc)
            self.current_conversation_id = None
            self.status_var.set("Stockage local indisponible : " + str(exc))

        if self.rag_index.warnings:
            self.status_var.set("Certains documents ne sont plus à jour ; ouvre Documents et réindexe-les.")

    def _open_chat_link(self, event: tk.Event) -> str:
        index = self.chat_text.index(f"@{event.x},{event.y}")
        for tag in self.chat_text.tag_names(index):
            if tag == "web_link":
                ranges = self.chat_text.tag_prevrange("web_link", f"{index}+1c")
                if ranges:
                    url = self.chat_text.get(ranges[0], ranges[1]).strip().rstrip(".,;:!")
                    if url.startswith(("http://", "https://")):
                        try:
                            webbrowser.open(url, new=2)
                        except Exception as exc:
                            self.status_var.set("Impossible d'ouvrir le lien : " + str(exc))
                break
        return "break"

    def _render_message(self, role: str, message: str, display_name: str | None = None) -> None:
        if role == "assistant":
            self.last_assistant_message = message.rstrip()
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", message_display_name(role, display_name) + "\n", f"{role}_name")
        body_start = self.chat_text.index("end-1c")
        body = message.rstrip()
        self.chat_text.insert("end", body + "\n\n", f"{role}_body")
        for match in re.finditer(r"https?://[^\s<>\]\[()\"']+", body):
            start = f"{body_start}+{match.start()}c"
            end = f"{body_start}+{match.end()}c"
            self.chat_text.tag_add("web_link", start, end)
        self.chat_text.tag_raise("sel")
        self.chat_text.see("end")

    def _refresh_conversation_list(self, search_term: str | None = None) -> None:
        if self.store is None:
            return
        if search_term is None and hasattr(self, "session_search_var"):
            term = self.session_search_var.get().strip()
            if term in ("", "Rechercher…"):
                search_term = None
            else:
                search_term = term
        needle = (search_term or "").casefold().strip()
        try:
            limit = 500 if needle else 100
            conversations = self.store.list_conversations(limit=limit)
        except StorageError as exc:
            self.status_var.set(str(exc))
            return
        if needle:
            conversations = [
                item for item in conversations
                if needle in " ".join(str(item.get("title", "")).split()).casefold()
                or needle in str(item.get("updated_at", "")).casefold()
            ]
        self.conversation_list.delete(0, "end")
        self.conversation_ids = []
        current_index: int | None = None
        for entry in conversations:
            conversation_id = str(entry["id"])
            title = " ".join(str(entry["title"]).split()) or "Nouvelle session"
            title = title if len(title) <= 21 else title[:20] + "…"
            stamp = str(entry.get("updated_at", ""))[:10]
            count = int(entry.get("message_count", 0))
            label = f"{title}  ·  {count} msg  ·  {stamp}" if stamp else title
            self.conversation_list.insert("end", label)
            self.conversation_ids.append(conversation_id)
            if conversation_id == self.current_conversation_id:
                current_index = len(self.conversation_ids) - 1
        if current_index is not None:
            self.conversation_list.selection_clear(0, "end")
            self.conversation_list.selection_set(current_index)
            self.conversation_list.activate(current_index)
        if needle:
            self.status_var.set(f"{len(conversations)} sessions correspondant à « {search_term} ».")

    def _clear_search_placeholder(self, _event=None) -> None:
        if self.session_search_entry.get() == "Rechercher…":
            self.session_search_entry.delete(0, "end")
            self.session_search_entry.configure(fg="#F1F5FC")

    def _restore_search_placeholder(self, _event=None) -> None:
        if not self.session_search_entry.get():
            self.session_search_entry.insert(0, "Rechercher…")
            self.session_search_entry.configure(fg="#8A9AB3")

    def _on_conversation_list_select(self, _event: tk.Event | None = None) -> None:
        selection = self.conversation_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index < len(self.conversation_ids):
            self._load_conversation(self.conversation_ids[index])

    def _show_conversation_context_menu(self, event: tk.Event) -> str:
        index = self.conversation_list.nearest(event.y)
        if index >= 0 and index < len(self.conversation_ids):
            self.conversation_list.selection_clear(0, "end")
            self.conversation_list.selection_set(index)
            self.conversation_list.activate(index)
            try:
                self.conversation_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.conversation_context_menu.grab_release()
        return "break"

    def _toggle_sessions_panel(self) -> None:
        if not hasattr(self, "recent_section") or self.recent_section.winfo_ismapped():
            self.recent_section.pack_forget()
            self.sessions_button.configure(text="↔  Sessions ▸")
            self.status_var.set("Liste des sessions masquée.")
        else:
            self.recent_section.pack(fill="x", before=self.sessions_button)
            self.sessions_button.configure(text="↔  Sessions ▾")
            self._refresh_conversation_list()
            self.status_var.set("Liste des sessions affichée.")

    def delete_selected_conversation(self) -> None:
        if self.busy:
            self.status_var.set("Attends la fin de la génération avant de supprimer une conversation.")
            return
        selected = self.conversation_list.curselection()
        if not selected or self.store is None:
            self.status_var.set("Sélectionne une session dans la liste pour la supprimer.")
            return
        index = int(selected[0])
        if index >= len(self.conversation_ids):
            return
        conversation_id = self.conversation_ids[index]
        try:
            record = self.store.get_conversation(conversation_id)
        except StorageError as exc:
            messagebox.showerror("Supprimer la conversation", str(exc), parent=self.root)
            return
        title = str(record["title"]) if record else "cette conversation"
        if not messagebox.askyesno(
            "Supprimer la conversation",
            f"Supprimer définitivement « {title} » et tous ses messages locaux ? Cette action ne peut pas être annulée.",
            parent=self.root,
        ):
            return
        try:
            self.store.delete_conversation(conversation_id)
        except StorageError as exc:
            messagebox.showerror("Supprimer la conversation", str(exc), parent=self.root)
            return
        if conversation_id == self.current_conversation_id:
            self.current_conversation_id = None
            self.history.clear()
            self.last_assistant_message = ""
            self.chat_text.delete("1.0", "end")
            self.conversation_title_var.set("Aucune session")
            self._refresh_sources_panel([], active=False)
            self.status_var.set("Session supprimée. Aucune session active.")
        else:
            self._refresh_conversation_list()
            self.status_var.set("Conversation supprimée de la base locale.")
        self._refresh_conversation_list()

    def _load_conversation(self, conversation_id: str, *, refresh_list: bool = True) -> None:
        if self.store is None:
            return
        if self.busy:
            self.status_var.set("Attends la fin de la génération avant de changer de conversation.")
            self._refresh_conversation_list()
            return
        if conversation_id == self.current_conversation_id:
            return
        try:
            record = self.store.get_conversation(conversation_id)
            if record is None:
                return
            messages = self.store.list_messages(conversation_id)
        except StorageError as exc:
            self.status_var.set(str(exc))
            return
        self.current_conversation_id = conversation_id
        self.history = [
            {
                "role": str(item["role"]),
                "content": str(item["content"]),
                "timestamp": str(item.get("timestamp") or ""),
                "display_name": str(item["display_name"]) if item.get("display_name") else "",
            }
            for item in messages
        ]
        self.last_assistant_message = ""
        self.chat_text.delete("1.0", "end")
        for item in self.history:
            self._render_message(item["role"], item["content"], item.get("display_name") or None)
        self.conversation_title_var.set(str(record["title"]))
        if not self.history:
            self.status_var.set("Conversation restaurée, sans message.")
        self._refresh_sources_panel([], active=False)
        self.status_var.set("Conversation locale restaurée.")
        if refresh_list:
            self._refresh_conversation_list()

    def _rename_current_conversation(self, title: str) -> None:
        clean_title = " ".join(title.split()).strip()[:120] or "Nouvelle session"
        self.conversation_title_var.set(clean_title)
        if self.store is not None and self.current_conversation_id:
            try:
                self.store.update_conversation_title(self.current_conversation_id, clean_title)
            except StorageError as exc:
                self.status_var.set(str(exc))
        self._refresh_conversation_list()

    def _append_message(
        self, role: str, message: str, *, save: bool = True,
        display_name: str | None = None,
    ) -> None:
        if save:
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            is_first_user_message = role == "user" and not any(item.get("role") == "user" for item in self.history)
            entry = {
                "role": role, "content": message, "timestamp": stamp,
                "display_name": display_name or "",
            }
            self.history.append(entry)
            if self.store is not None and self.current_conversation_id:
                try:
                    self.store.add_message(self.current_conversation_id, role, message, stamp, display_name)
                    if is_first_user_message:
                        title = " ".join(message.split())[:38]
                        if len(" ".join(message.split())) > 38:
                            title += "…"
                        self.store.update_conversation_title(self.current_conversation_id, title)
                        self.conversation_title_var.set(title)
                        self._refresh_conversation_list()
                    self._refresh_conversation_list()
                except StorageError as exc:
                    self.status_var.set("Message affiché, mais sauvegarde locale impossible : " + str(exc))
        self._render_message(role, message, display_name)

    def _copy_text_to_clipboard(self, text: str, label: str) -> bool:
        cleaned = normalize_clipboard_text(text)
        if not cleaned.strip():
            self.status_var.set("Aucun texte à copier.")
            return False
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(cleaned)
            self.root.update_idletasks()
        except tk.TclError as exc:
            self.status_var.set(f"Impossible d'accéder au presse-papiers : {exc}")
            return False
        self.status_var.set(f"{label} copié dans le presse-papiers.")
        return True

    def copy_chat_selection(self) -> None:
        try:
            selected = self.chat_text.get("sel.first", "sel.last")
        except tk.TclError:
            selected = ""
        self._copy_text_to_clipboard(selected, "Sélection")

    def copy_last_assistant_response(self) -> None:
        self._copy_text_to_clipboard(self.last_assistant_message, "Dernière réponse")

    def copy_entire_chat(self) -> None:
        self._copy_text_to_clipboard(self.chat_text.get("1.0", "end-1c").rstrip("\n"), "Discussion")

    def select_all_chat(self) -> None:
        self.chat_text.focus_set()
        self.chat_text.tag_add("sel", "1.0", "end-1c")
        self.status_var.set("Toute la discussion est sélectionnée. Utilise Ctrl+C ou le menu contextuel pour copier.")

    def _copy_selection_shortcut(self, _event: tk.Event) -> str:
        self.copy_chat_selection()
        return "break"

    def _select_all_chat_shortcut(self, _event: tk.Event) -> str:
        self.select_all_chat()
        return "break"

    def _block_chat_edit_shortcut(self, _event: tk.Event) -> str:
        """Keep the transcript read-only while preserving mouse/keyboard selection."""
        return "break"

    def _protect_chat_transcript_edit(self, event: tk.Event) -> str | None:
        """Block typing but allow native caret navigation and Shift-selection.

        Using a normal Text widget (instead of Tk's disabled state) allows reliable
        drag-selection of arbitrary passages on Windows. All non-navigation keypresses
        are stopped here, with common clipboard/edit shortcuts handled separately.
        """
        navigation_keys = {
            "Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next",
            "KP_Left", "KP_Right", "KP_Up", "KP_Down", "KP_Home", "KP_End",
            "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
            "Caps_Lock", "Num_Lock", "Scroll_Lock", "Escape",
        }
        editing_keys = {"BackSpace", "Delete", "Return", "KP_Enter", "Tab", "Insert"}
        if event.keysym in navigation_keys:
            return None
        if event.keysym in editing_keys:
            return "break"
        # Block printable text input but let function keys and OS/window shortcuts
        # (for example Alt+F4) continue through the remaining Tk binding tags.
        if getattr(event, "char", "") and event.char.isprintable():
            return "break"
        return None

    def _show_chat_context_menu(self, event: tk.Event) -> str:
        try:
            self.chat_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.chat_context_menu.grab_release()
        return "break"

    def _composer_selection(self) -> str:
        try:
            return self.message_entry.get("sel.first", "sel.last")
        except tk.TclError:
            return ""

    def copy_composer_selection(self) -> None:
        self._copy_text_to_clipboard(self._composer_selection(), "Sélection")

    def cut_composer_selection(self) -> None:
        selected = self._composer_selection()
        if not selected:
            return
        if self._copy_text_to_clipboard(selected, "Texte"):
            self.message_entry.delete("sel.first", "sel.last")
            self.message_entry.configure(fg=COLORS["text"])

    def paste_into_composer(self) -> None:
        try:
            clipboard_text = self.root.clipboard_get()
        except tk.TclError:
            self.status_var.set("Le presse-papiers ne contient pas de texte accessible.")
            return
        if not clipboard_text:
            self.status_var.set("Le presse-papiers est vide.")
            return
        self.message_entry.focus_set()
        if self.message_entry.get("1.0", "end-1c") == "Écrivez votre message ici…":
            self.message_entry.delete("1.0", "end")
        try:
            self.message_entry.delete("sel.first", "sel.last")
        except tk.TclError:
            pass
        self.message_entry.insert("insert", normalize_clipboard_text(clipboard_text))
        self.message_entry.configure(fg=COLORS["text"])
        if looks_like_supported_file_path(clipboard_text):
            self.status_var.set(
                "Chemin de fichier collé, pas le contenu du fichier. Pour l'indexer, utilise « Joindre PDF / image / texte », puis « Indexer la sélection »."
            )
        else:
            self.status_var.set("Texte collé dans le champ du message. Vérifie-le avant de l'envoyer.")

    def select_all_composer(self) -> None:
        self.message_entry.focus_set()
        self._composer_focus_in(None)
        self.message_entry.tag_add("sel", "1.0", "end-1c")

    def _paste_shortcut(self, _event: tk.Event) -> str:
        self.paste_into_composer()
        return "break"

    def _copy_composer_shortcut(self, _event: tk.Event) -> str:
        self.copy_composer_selection()
        return "break"

    def _cut_composer_shortcut(self, _event: tk.Event) -> str:
        self.cut_composer_selection()
        return "break"

    def _show_composer_context_menu(self, event: tk.Event) -> str:
        self.message_entry.focus_set()
        try:
            self.composer_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.composer_context_menu.grab_release()
        return "break"

    def _composer_focus_in(self, _event: tk.Event) -> None:
        if self.message_entry.get("1.0", "end-1c") == "Écrivez votre message ici…":
            self.message_entry.delete("1.0", "end")
            self.message_entry.configure(fg=COLORS["text"])

    def _composer_focus_out(self, _event: tk.Event) -> None:
        if not self.message_entry.get("1.0", "end-1c").strip():
            self.message_entry.delete("1.0", "end")
            self.message_entry.insert("1.0", "Écrivez votre message ici…")
            self.message_entry.configure(fg="#8B98AE")

    def _composer_return_shortcut(self, event: tk.Event) -> str | None:
        """Enter sends; Shift+Enter keeps Tk's standard newline insertion."""
        # Tk's ShiftMask bit is 0x0001 on Windows and standard Tk platforms.
        if int(getattr(event, "state", 0)) & 0x0001:
            return None
        self.send_message()
        return "break"

    def _send_shortcut(self, _event: tk.Event) -> str:
        self.send_message()
        return "break"

    def _refresh_mode_status(self) -> None:
        if self.rag_enabled.get():
            rag_state = f"RAG : actif ({self.rag_index.chunk_count} passages)" if self.rag_index.chunks else "RAG : activé, index vide"
        else:
            rag_state = "RAG : désactivé"
        web_state = "Web : demandé, non connecté" if self.web_enabled.get() else "Web : désactivé"
        answer_state = "Prochaine réponse : extrait fidèle" if self.answer_mode_var.get() == ANSWER_MODE_EXTRACTIVE else "Prochaine réponse : BitNet"
        self.mode_note_var.set(rag_state + " • " + web_state + " • " + answer_state)
        if hasattr(self, "model_status"):
            self.model_status.configure(
                text=f"BitNet 1.58 • RAG {'ON' if self.rag_enabled.get() else 'OFF'}",
                bg=COLORS["bg"], fg=COLORS["muted"],
            )

    def send_message(self) -> None:
        if self.store is not None and self.current_conversation_id is None:
            try:
                self.current_conversation_id = self.store.create_conversation("Nouvelle session")
                self.conversation_title_var.set("Nouvelle session")
                self._refresh_conversation_list()
            except StorageError as exc:
                self.status_var.set(str(exc))
                return
        if self.busy:
            self.status_var.set("Une opération est déjà en cours. Attends qu'elle se termine.")
            return
        raw = self.message_entry.get("1.0", "end-1c")
        message = normalize_message(raw)
        if not message or message == "Écrivez votre message ici…":
            return
        extractive_mode = self.answer_mode_var.get() == ANSWER_MODE_EXTRACTIVE
        if extractive_mode and not self.rag_enabled.get():
            messagebox.showinfo(
                "Mode fidèle au document",
                "Active « Utiliser le RAG » et indexe au moins un document pour extraire une phrase exacte. Ce mode n'utilise pas le modèle génératif.",
                parent=self.root,
            )
            return
        if not extractive_mode:
            if self.engine_config is None:
                messagebox.showinfo(
                    "Moteur local non configuré",
                    "Configure d'abord llama-completion.exe (recommandé) ou llama-cli.exe et le fichier modèle GGUF avec le bouton « Moteur et modèles… ». "
                    "Ton message reste dans le champ jusqu'à ce que tu l'envoies.",
                    parent=self.root,
                )
                self.configure_engine()
                return
            try:
                validate_config(self.engine_config)
            except EngineError as exc:
                messagebox.showwarning("Configuration du moteur", str(exc), parent=self.root)
                self.configure_engine()
                return
        rag_hits: list[RetrievalHit] = []
        rag_context: str | None = None
        if self.rag_enabled.get():
            # Re-check source metadata before every retrieval so modified files never silently feed stale passages.
            self._refresh_documents_list()
            if not self.rag_index.chunks:
                messagebox.showwarning(
                    "Bibliothèque RAG vide",
                    "Ajoute des PDF textuels ou des fichiers texte, ouvre l'onglet Documents et clique sur « Indexer la sélection » avant d'utiliser le RAG.",
                    parent=self.root,
                )
                return
            context_size = self.engine_config.context_size if self.engine_config is not None else 1024
            rag_top_k = 1 if context_size < 2048 else (2 if context_size < 4096 else 3)
            hnsw_index = getattr(self.rag_index, "hnsw_index", None)
            if getattr(self.rag_index, "engine", "lexical") == "hnsw" and hnsw_index is not None:
                try:
                    rag_hits = hnsw_index.search(message, rag_top_k)
                except Exception:
                    self.rag_index.engine = "lexical"
                    rag_hits = retrieve(message, self.rag_index.chunks, top_k=rag_top_k)
            else:
                rag_hits = retrieve(message, self.rag_index.chunks, top_k=rag_top_k)
            self._refresh_sources_panel(rag_hits, active=True)
            if not rag_hits:
                self.message_entry.delete("1.0", "end")
                self.message_entry.configure(fg=COLORS["text"])
                self._append_message("user", message)
                self._append_message(
                    "assistant",
                    "Je n'ai trouvé aucun passage pertinent dans les documents indexés. Essaie d'autres mots-clés ou vérifie que le document a bien été indexé.",
                    display_name="RAG local — aucun passage trouvé",
                )
                self.status_var.set("Aucun passage pertinent trouvé ; le moteur n'a pas été interrogé.")
                return
            rag_context = build_rag_context(rag_hits, query=message)
        else:
            self._refresh_sources_panel([], active=False)

        prior_history = [
            {"role": item["role"], "content": item["content"]}
            for item in self.history
            if item.get("role") in ("user", "assistant")
        ]
        memory_context = self._build_memory_context(message, document_grounded=bool(rag_context))
        self.message_entry.delete("1.0", "end")
        self.message_entry.configure(fg=COLORS["text"])
        self._append_message("user", message)
        self.recent_label.configure(text="Conversations et souvenirs stockés localement sur ce PC.")
        if extractive_mode:
            selected = extract_faithful_sentence(rag_hits, message)
            if selected is None:
                answer = (
                    "Je n'ai pas trouvé de phrase suffisamment correspondante pour répondre fidèlement. "
                    "Consulte les extraits affichés ou reformule la question avec des mots présents dans le document."
                )
                answer_sources = rag_hits
            else:
                sentence, source_hit = selected
                answer = "Extrait fidèle du document :\n« " + sentence + " »"
                answer_sources = [source_hit]
            answer = append_retrieved_source_references(answer, answer_sources)
            self._append_message(
                "assistant", answer, display_name="Extrait fidèle — RAG local"
            )
            self._refresh_sources_panel(answer_sources, active=bool(answer_sources))
            self.status_var.set("Réponse extraite littéralement du document — aucun appel au moteur génératif.")
            return

        config = self.engine_config
        assert config is not None  # validated above for the generative mode
        # Web search is strictly opt-in. Search snippets become context for the
        # configured local model; no remote AI service is used.
        self._run_task(
            "Réponse BitNet avec recherche Web" if self.web_enabled.get() else "Réponse BitNet",
            lambda: self._generate_answer_task(
                config, prior_history, message, rag_context, memory_context, rag_hits,
                use_web=self.web_enabled.get(),
            ),
        )

    def _generate_answer_task(self, config, history, message, rag_context, memory_context, rag_hits, *, use_web=False):
        web_results = search_web(message, limit=5) if use_web else []
        context_parts = [part for part in (rag_context, format_web_context(web_results) if web_results else None) if part]
        combined_context = "\n\n---\n\n".join(context_parts) or None
        # The clock remains a UI-only indicator. Do not inject date/time into model prompts.
        system_prompt = (getattr(config, "system_prompt", "") or "").strip()
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt = (f"Consignes de comportement : {system_prompt}\n\n" if system_prompt else "") + message
        if web_results:
            prompt += "\n\nConsigne : réponds dans la langue de la question. Utilise uniquement les informations réellement présentes dans les résultats ci-dessous, signale les incertitudes, ne prétends pas avoir lu les pages entières et cite les références [W1], [W2], etc."
            if any(term in message.casefold() for term in ("météo", "meteo", "weather", "prévision météo", "forecast")):
                prompt += " Pour une question météo, distingue clairement la date demandée, la ville, les températures minimale et maximale, les conditions et le risque de précipitations. Ne transforme pas des températures isolées en prévision horaire et n'invente aucune valeur si les extraits ne la donnent pas."
        if self.planning_mode_var.get():
            planning_message = "Mode plan (experimental) : etablis 3 sous-etapes courtes, puis une proposition de reponse finale.\nQuestion : " + message
            plan = generate_response(
                config,
                history,
                (system_prompt + "\n\n" if system_prompt else "") + planning_message,
                context=combined_context,
                memory_context=memory_context,
            )
            prompt = (system_prompt + "\n\n" if system_prompt else "") + message + "\n\nPlan interne : " + plan + "\nReponds maintenant de facon concise, fondree, et sans inventer."
            answer = generate_response(config, history, prompt, context=combined_context, memory_context=memory_context)
            answer = answer.rstrip() + "\n\n---\nPlan agent local (experimental) :\n" + plan
        else:
            answer = generate_response(config, history, prompt, context=combined_context, memory_context=memory_context)
        if web_results:
            answer = answer.rstrip() + "\n\n" + format_web_references(web_results)
        return {"answer": answer, "sources": rag_hits, "web_results": web_results}

    def choose_files(self) -> None:
        if self.busy:
            messagebox.showinfo("Opération en cours", "Attends la fin de l'opération avant de modifier la sélection de documents.", parent=self.root)
            return
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Sélectionner des documents ou images",
            filetypes=[
                ("PDF, texte et images", "*.pdf *.txt *.md *.rst *.csv *.html *.htm *.log *.json *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("Documents PDF", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("Documents texte", "*.txt *.md *.rst *.csv *.html *.htm *.log *.json"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not selected:
            return
        rejected: list[str] = []
        existing = {str(path.resolve()).casefold() for path in self.selected_files}
        for raw in selected:
            path = Path(raw)
            if path.suffix.lower() not in SUPPORTED_ATTACHMENT_SUFFIXES:
                rejected.append(path.name)
                continue
            resolved_key = str(path.resolve()).casefold()
            if resolved_key not in existing:
                self.selected_files.append(path)
                existing.add(resolved_key)
        # Selecting another file must not invalidate the already indexed library.
        self._refresh_documents_list()
        self.show_document_tab("documents")
        if rejected:
            messagebox.showwarning("Fichiers non pris en charge", "Ces fichiers n'ont pas été ajoutés :\n" + "\n".join(rejected), parent=self.root)
        if self.selected_files:
            self.status_var.set(f"{len(self.selected_files)} fichier(s) sélectionné(s) — clique sur « Indexer la sélection » pour les lire localement.")

    def _current_stale_document_keys(self) -> frozenset[str]:
        """Return normalized paths whose persisted or in-memory index is stale."""
        stale = {str(Path(p).expanduser().resolve()).casefold() for p in self.rag_index.stale_files}
        if self.store is not None:
            try:
                stale.update(str(Path(p).expanduser().resolve()).casefold() for p in self.store.list_stale_rag_files())
            except StorageError:
                # Keep the last visible state when the database cannot be checked.
                pass
        return frozenset(stale)

    def _poll_document_changes(self) -> None:
        """Refresh stale-source status periodically while the Documents tab is open."""
        try:
            if self.current_tab == "documents" and not self.busy:
                current = self._current_stale_document_keys()
                previous = getattr(self, "_last_document_status_snapshot", frozenset())
                if current != previous:
                    newly_stale = current - previous
                    self._refresh_documents_list()
                    if newly_stale and not self.busy:
                        changed_name = next((
                            path.name for path in self.selected_files
                            if str(path.expanduser().resolve()).casefold() in newly_stale
                        ), "un document")
                        changed_path = next((
                            path for path in self.selected_files
                            if str(path.expanduser().resolve()).casefold() in newly_stale
                        ), None)
                        if changed_path is not None and not changed_path.is_file():
                            self.status_var.set(
                                f"Source introuvable : {changed_name}. Son index est conservé, mais ses passages sont exclus du RAG. Restaure le fichier ou supprime explicitement son index."
                            )
                        else:
                            self.status_var.set(
                                f"Modification détectée : {changed_name}. Réindexe ce document pour actualiser le RAG ; ses anciens passages sont exclus des recherches."
                            )
            self.root.after(2000, self._poll_document_changes)
        except tk.TclError:
            # The window may have been closed while the scheduled callback waited.
            return

    def _refresh_documents_list(self) -> None:
        # Detect changes when the list is opened/refreshed and keep selection stable.
        selected_keys = set()
        try:
            selected_keys = {
                str(self.selected_files[int(index)].expanduser().resolve()).casefold()
                for index in self.documents_list.curselection()
                if 0 <= int(index) < len(self.selected_files)
            }
        except (AttributeError, TypeError, ValueError):
            pass
        stale_paths = set()
        if self.store is not None:
            try:
                stale_paths = {str(Path(p).resolve()).casefold() for p in self.store.list_stale_rag_files()}
            except StorageError as exc:
                self.status_var.set(f"Vérification des documents impossible : {exc}")
        if stale_paths:
            self.rag_index.stale_files = list(dict.fromkeys(
                self.rag_index.stale_files + [p for p in self.rag_index.indexed_files if str(Path(p).resolve()).casefold() in stale_paths]
            ))
            self.rag_index.chunks = [c for c in self.rag_index.chunks if str(Path(c.source_path).resolve()).casefold() not in stale_paths]
            self.rag_index.indexed_files = [p for p in self.rag_index.indexed_files if str(Path(p).resolve()).casefold() not in stale_paths]
        indexed = {str(Path(p).resolve()).casefold() for p in self.rag_index.indexed_files}
        stale = {str(Path(p).resolve()).casefold() for p in self.rag_index.stale_files} | stale_paths
        self.documents_list.delete(0, "end")
        for index, path in enumerate(self.selected_files):
            key = str(path.resolve()).casefold()
            if key in stale:
                marker = "⚠ source introuvable — index conservé" if not path.is_file() else "⚠ modifié — à réindexer"
            elif key in indexed:
                marker = "✓ indexé"
            else:
                marker = "à indexer"
            self.documents_list.insert("end", f"{attachment_kind(path)}   •   {path.name}   [{marker}]")
            if key in selected_keys:
                self.documents_list.selection_set(index)
        self._last_document_status_snapshot = frozenset(stale)
        self.documents_heading.configure(text=f"Documents sélectionnés ({len(self.selected_files)}) — {self.rag_index.chunk_count} passages valides")

    def _get_selected_indexed_document(self) -> tuple[Path, list[DocumentChunk]] | None:
        """Resolve a single indexed document from the Documents list, with a safe single-file convenience."""
        if not self.rag_index.chunks:
            messagebox.showwarning(
                "Aucun document indexé",
                "Ajoute un document puis clique sur « Indexer la sélection » avant de le résumer.",
                parent=self.root,
            )
            return None
        selected_indices = list(self.documents_list.curselection())
        if len(selected_indices) > 1:
            messagebox.showinfo("Résumer un document", "Sélectionne un seul document à la fois dans la liste.", parent=self.root)
            return None
        if selected_indices:
            index = selected_indices[0]
            if index >= len(self.selected_files):
                messagebox.showwarning("Document introuvable", "La sélection de documents a changé. Actualise la liste puis réessaie.", parent=self.root)
                return None
            path = self.selected_files[index]
        else:
            indexed_paths = list(dict.fromkeys(str(Path(chunk.source_path).resolve()) for chunk in self.rag_index.chunks))
            if len(indexed_paths) != 1:
                self.show_document_tab("documents")
                messagebox.showinfo(
                    "Choisir un document",
                    "Sélectionne une seule ligne dans la liste des documents à droite, puis relance le résumé.",
                    parent=self.root,
                )
                return None
            path = Path(indexed_paths[0])
        key = str(path.expanduser().resolve()).casefold()
        chunks = [chunk for chunk in self.rag_index.chunks if str(Path(chunk.source_path).expanduser().resolve()).casefold() == key]
        if not chunks:
            messagebox.showwarning(
                "Document non indexé",
                "Ce fichier ne possède aucun passage dans l'index actuel. Clique sur « Indexer la sélection » puis réessaie.",
                parent=self.root,
            )
            return None
        return path, chunks

    def summarize_selected_extractive(self) -> None:
        if self.busy:
            self.status_var.set("Attends la fin de l'opération en cours avant de lancer un résumé.")
            return
        selected = self._get_selected_indexed_document()
        if selected is None:
            return
        path, chunks = selected
        self._run_task("Résumé rapide", lambda: build_extractive_summary(chunks, max_sentences=8))
        self.show_document_tab("documents")

    def summarize_selected_with_bitnet(self) -> None:
        if self.busy:
            self.status_var.set("Attends la fin de l'opération en cours avant de lancer un résumé.")
            return
        selected = self._get_selected_indexed_document()
        if selected is None:
            return
        path, chunks = selected
        config = self.engine_config
        if config is None:
            messagebox.showinfo("Moteur local requis", "Configure d'abord le moteur BitNet local avec « Moteur et modèles… ».", parent=self.root)
            self.configure_engine()
            return
        try:
            validate_config(config)
            calls = estimate_bitnet_summary_calls(chunks, config.context_size)
        except (EngineError, SummaryError) as exc:
            messagebox.showwarning("Résumé BitNet impossible", str(exc), parent=self.root)
            return
        call_word = "génération locale" if calls == 1 else f"générations locales (environ {calls} lancements)"
        confirmed = messagebox.askyesno(
            "Confirmer le résumé BitNet",
            f"Résumer le document :\n{path.name}\n\n"
            f"Estimation : {calls} {call_word}. Le moteur actuel se relance pour chaque génération ; "
            "un document long peut donc demander plusieurs minutes.\n\n"
            "Le traitement reste local : aucune API distante n'est utilisée. Continuer ?",
            parent=self.root,
        )
        if not confirmed:
            return
        def progress(done: int, total: int, detail: str) -> None:
            self.root.after(0, self._set_summary_progress, path.name, done, total, detail)
        self._run_task(
            "Résumé avec BitNet",
            lambda: summarize_with_bitnet(config, chunks, progress=progress),
        )
        self.show_document_tab("documents")

    def _set_summary_progress(self, filename: str, done: int, total: int, detail: str) -> None:
        if self.busy:
            self.status_var.set(f"Résumé BitNet — {filename} : {detail} ({min(done, total)}/{total})…")

    def remove_selected_files(self) -> None:
        if self.busy:
            messagebox.showinfo("Opération en cours", "Attends la fin de l'opération avant de supprimer un index.", parent=self.root)
            return
        selected_indices = list(self.documents_list.curselection())
        if not selected_indices:
            messagebox.showinfo("Supprimer un index", "Sélectionne d'abord un document dans la liste.", parent=self.root)
            return
        paths = [self.selected_files[i] for i in selected_indices if i < len(self.selected_files)]
        if not paths:
            return
        names = "\n".join(f"• {path.name}" for path in paths)
        if not messagebox.askyesno(
            "Supprimer l’index documentaire",
            "Supprimer les passages indexés pour ce(s) document(s) ?\n\n" + names +
            "\n\nLes fichiers originaux et les conversations ne seront pas supprimés.",
            parent=self.root,
        ):
            return
        try:
            if self.store is not None:
                self.store.delete_rag_documents(paths)
        except StorageError as exc:
            messagebox.showerror("Supprimer l’index", str(exc), parent=self.root)
            return
        keys = {str(Path(p).resolve()).casefold() for p in paths}
        self.selected_files = [p for p in self.selected_files if str(Path(p).resolve()).casefold() not in keys]
        self.rag_index.chunks = [c for c in self.rag_index.chunks if str(Path(c.source_path).resolve()).casefold() not in keys]
        self.rag_index.indexed_files = [p for p in self.rag_index.indexed_files if str(Path(p).resolve()).casefold() not in keys]
        self.rag_index.stale_files = [p for p in self.rag_index.stale_files if str(Path(p).resolve()).casefold() not in keys]
        self.last_rag_hits = []
        self._refresh_documents_list()
        self._refresh_sources_panel([], active=False, note="Index du document supprimé. Le fichier original est conservé sur le disque.")
        self._refresh_mode_status()
        self.status_var.set("Index supprimé pour la sélection. Les fichiers originaux, autres index et conversations sont conservés.")

    def _invalidate_index(self, reason: str) -> None:
        # Kept for explicit full-library reset only; ordinary selection changes must not call it.
        self.rag_index = RAGIndex()
        self.last_rag_hits = []
        if self.store is not None:
            try:
                self.store.save_rag_index(self.rag_index)
            except StorageError as exc:
                self.status_var.set("Index retiré de la session, mais pas de la base locale : " + str(exc))
        self._refresh_sources_panel([], active=False, note=reason)
        self._refresh_mode_status()

    def index_selected_documents(self) -> None:
        if self.busy:
            messagebox.showinfo("Opération en cours", "Attends la fin de l'opération en cours.", parent=self.root)
            return
        if not self.selected_files:
            messagebox.showinfo("Indexer les documents", "Clique d'abord sur « Joindre PDF / image / texte » pour sélectionner des fichiers.", parent=self.root)
            return
        paths = list(self.selected_files)
        self._run_task("Indexer les documents", lambda: index_documents(paths))

    def _update_sources_scrollregion(self, _event=None) -> None:
        """Keep the source-card canvas scroll region in sync with its content."""
        if hasattr(self, "sources_canvas"):
            self.sources_canvas.configure(scrollregion=self.sources_canvas.bbox("all"))

    def _resize_sources_canvas_window(self, event) -> None:
        """Make source cards fit the available panel width without horizontal scrolling."""
        if hasattr(self, "sources_canvas") and hasattr(self, "_sources_canvas_window"):
            self.sources_canvas.itemconfigure(self._sources_canvas_window, width=max(1, event.width))

    def _bind_sources_mousewheel(self, _event=None) -> None:
        """Enable wheel scrolling only while the pointer is over the source list."""
        self.root.bind_all("<MouseWheel>", self._scroll_sources_mousewheel)
        self.root.bind_all("<Button-4>", self._scroll_sources_mousewheel)
        self.root.bind_all("<Button-5>", self._scroll_sources_mousewheel)

    def _unbind_sources_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _scroll_sources_mousewheel(self, event):
        """Scroll the source excerpts independently of the chat transcript."""
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = getattr(event, "delta", 0)
            units = -1 if delta > 0 else 1 if delta < 0 else 0
        if units:
            self.sources_canvas.yview_scroll(units, "units")
        return "break"

    def _refresh_sources_panel(self, hits: list[RetrievalHit], *, active: bool, note: str | None = None) -> None:
        self.last_rag_hits = list(hits)
        for child in self.sources_cards.winfo_children():
            child.destroy()
        if hits:
            self.sources_heading.configure(text=f"Sources utilisées ({len(hits)})")
            self.sources_info.configure(text=note or "Passages récupérés localement pour cette réponse. Les identifiants [S1], [S2]… correspondent à ces extraits.")
            for hit in hits:
                card = tk.Frame(self.sources_cards, bg=COLORS["blue_light"], highlightbackground=COLORS["line"], highlightthickness=0, padx=8, pady=7)
                card.pack(fill="x", pady=(0, 7))
                tk.Label(card, text=f"[{hit.source_id}] {hit.chunk.filename}", bg=COLORS["blue_light"], fg=COLORS["text"], font=("Segoe UI", 9, "bold"), wraplength=245, justify="left").pack(anchor="w")
                tk.Label(card, text=hit.chunk.locator, bg=COLORS["blue_light"], fg=COLORS["muted"], font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 4))
                excerpt = hit.chunk.text.strip()
                if len(excerpt) > 420:
                    excerpt = excerpt[:420].rstrip() + "…"
                tk.Label(card, text=excerpt, bg=COLORS["blue_light"], fg=COLORS["text"], font=("Segoe UI", 8), wraplength=245, justify="left", anchor="w").pack(fill="x")
        else:
            self.sources_heading.configure(text="Aucune source utilisée")
            if note:
                detail = note
            elif active:
                detail = "Aucun passage pertinent retrouvé dans les documents indexés. Le modèle n'a pas été interrogé pour cette question."
            else:
                detail = "Active « Utiliser le RAG » après l'indexation pour voir ici les passages réellement utilisés. Sans RAG, aucune source documentaire ne sera affichée."
            self.sources_info.configure(text=detail)

    def show_document_tab(self, tab: str) -> None:
        self.current_tab = tab
        self.sources_frame.pack_forget()
        self.documents_frame.pack_forget()
        self.memories_frame.pack_forget()
        inactive = {"bg": COLORS["white"], "fg": COLORS["text"]}
        active = {"bg": COLORS["blue_light"], "fg": COLORS["blue"]}
        if tab == "documents":
            self.documents_frame.pack(fill="both", expand=True)
            self._refresh_documents_list()
            self.files_tab_button.configure(**active)
            self.sources_tab_button.configure(**inactive)
            self.memories_tab_button.configure(**inactive)
        elif tab == "memories":
            self.memories_frame.pack(fill="both", expand=True)
            self.memories_tab_button.configure(**active)
            self.sources_tab_button.configure(**inactive)
            self.files_tab_button.configure(**inactive)
            self._refresh_memories_list()
        else:
            self.sources_frame.pack(fill="both", expand=True)
            self.sources_tab_button.configure(**active)
            self.files_tab_button.configure(**inactive)
            self.memories_tab_button.configure(**inactive)

    def _refresh_memories_list(self) -> None:
        if not hasattr(self, "memories_list"):
            return
        self.memories_list.delete(0, "end")
        self.memory_ids = []
        if self.store is None:
            self.memories_list.insert("end", "Stockage local indisponible")
            return
        try:
            memories = self.store.list_memories()
        except StorageError as exc:
            self.memories_list.insert("end", "Erreur de lecture des souvenirs")
            self.status_var.set(str(exc))
            return
        for item in memories:
            self.memory_ids.append(str(item["id"]))
            state = "✓" if item["enabled"] else "○"
            category = str(item.get("category", "Général"))
            preview = " ".join(str(item["content"]).split())
            if len(preview) > 38:
                preview = preview[:37] + "…"
            self.memories_list.insert("end", f"{state} [{category}] {preview}")
        if not memories:
            self.memories_list.insert("end", "Aucun souvenir enregistré")

    def _set_memories_usage(self) -> None:
        enabled = bool(self.memories_enabled_var.get())
        if self.store is not None:
            try:
                self.store.set_setting("memories_enabled", "1" if enabled else "0")
            except StorageError as exc:
                self.status_var.set(str(exc))
        self.status_var.set(
            "Souvenirs activés pour le chat local." if enabled
            else "Utilisation des souvenirs désactivée ; les souvenirs restent conservés sur ce PC."
        )

    def _selected_memory(self) -> dict[str, Any] | None:
        if self.store is None:
            messagebox.showwarning("Souvenirs locaux", "Le stockage local est indisponible.", parent=self.root)
            return None
        selected = self.memories_list.curselection()
        if not selected:
            messagebox.showinfo("Souvenirs locaux", "Sélectionne d'abord un souvenir dans la liste.", parent=self.root)
            return None
        index = int(selected[0])
        if index >= len(self.memory_ids):
            return None
        try:
            return self.store.get_memory(self.memory_ids[index])
        except StorageError as exc:
            messagebox.showerror("Souvenirs locaux", str(exc), parent=self.root)
            return None

    def add_memory(self) -> None:
        self._show_memory_editor()

    def edit_selected_memory(self) -> None:
        memory = self._selected_memory()
        if memory is not None:
            self._show_memory_editor(memory=memory)

    def _show_memory_editor(
        self,
        *,
        memory: dict[str, Any] | None = None,
        initial_content: str = "",
        source: str = "manuel",
    ) -> None:
        if self.store is None:
            messagebox.showwarning("Souvenirs locaux", "Le stockage local est indisponible.", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Modifier un souvenir" if memory else "Ajouter un souvenir")
        window.configure(bg=COLORS["bg"])
        window.transient(self.root)
        window.geometry("540x430")
        window.minsize(440, 340)
        body = tk.Frame(window, bg=COLORS["bg"], padx=16, pady=14)
        body.pack(fill="both", expand=True)
        self._label(body, "Information à retenir", bg=COLORS["bg"], font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self._label(
            body,
            "Enregistre uniquement ce que tu souhaites retrouver dans de futures conversations. Le souvenir sera stocké sur ce PC.",
            bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 10), wraplength=490, justify="left",
        ).pack(anchor="w", pady=(3, 8))
        category_row = tk.Frame(body, bg=COLORS["bg"])
        category_row.pack(fill="x", pady=(0, 8))
        self._label(category_row, "Catégorie :", bg=COLORS["bg"], font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
        category_var = tk.StringVar(value=str((memory or {}).get("category", "Général")))
        category_combo = ttk.Combobox(
            category_row, textvariable=category_var, state="readonly", width=18,
            values=("Préférence", "Projet", "Fait", "Règle", "Général", "Autre"),
        )
        category_combo.pack(side="left")
        content = tk.Text(
            body, wrap="word", font=("Segoe UI", 10), bg=COLORS["white"], fg=COLORS["text"],
            relief="flat", bd=0, padx=10, pady=8, undo=True,
        )
        content.pack(fill="both", expand=True)
        initial = str(memory["content"]) if memory else initial_content
        if initial:
            content.insert("1.0", initial)
        content.focus_set()

        buttons = tk.Frame(body, bg=COLORS["bg"])
        buttons.pack(fill="x", pady=(10, 0))

        def cancel() -> None:
            window.destroy()

        def save() -> None:
            value = content.get("1.0", "end-1c").strip()
            if not value:
                messagebox.showwarning("Souvenir vide", "Écris une information avant de l'enregistrer.", parent=window)
                return
            try:
                if memory:
                    self.store.update_memory(str(memory["id"]), value, category=category_var.get())
                    message = "Souvenir modifié et enregistré localement."
                else:
                    self.store.create_memory(value, category=category_var.get(), source=source)
                    message = "Souvenir ajouté. Il sera utilisé dans le chat si les souvenirs sont activés."
            except StorageError as exc:
                messagebox.showerror("Impossible d'enregistrer le souvenir", str(exc), parent=window)
                return
            window.destroy()
            self._refresh_memories_list()
            self.show_document_tab("memories")
            self.status_var.set(message)

        self._button(buttons, "Annuler", cancel, compact=True).pack(side="right", padx=(6, 0))
        self._button(buttons, "Enregistrer le souvenir", save, primary=True, compact=True).pack(side="right")
        window.bind("<Escape>", lambda _event: cancel())
        window.bind("<Control-Return>", lambda _event: save())
        window.grab_set()

    def save_selection_as_memory(self) -> None:
        try:
            selected_text = self.chat_text.get("sel.first", "sel.last").strip()
        except tk.TclError:
            selected_text = ""
        if not selected_text:
            self.status_var.set("Sélectionne d'abord un passage dans la discussion, puis ouvre le menu contextuel.")
            return
        self._show_memory_editor(initial_content=selected_text, source="sélection")

    def toggle_selected_memory(self) -> None:
        memory = self._selected_memory()
        if memory is None or self.store is None:
            return
        try:
            enabled = not bool(memory["enabled"])
            self.store.set_memory_enabled(str(memory["id"]), enabled)
        except StorageError as exc:
            messagebox.showerror("Souvenirs locaux", str(exc), parent=self.root)
            return
        self._refresh_memories_list()
        self.status_var.set("Souvenir activé." if enabled else "Souvenir désactivé ; il est conservé mais ne sera pas transmis au modèle.")

    def delete_selected_memory(self) -> None:
        memory = self._selected_memory()
        if memory is None or self.store is None:
            return
        preview = " ".join(str(memory["content"]).split())[:90]
        if not messagebox.askyesno("Supprimer le souvenir", f"Supprimer définitivement ce souvenir local ?\n\n{preview}", parent=self.root):
            return
        try:
            self.store.delete_memory(str(memory["id"]))
        except StorageError as exc:
            messagebox.showerror("Souvenirs locaux", str(exc), parent=self.root)
            return
        self._refresh_memories_list()
        self.status_var.set("Souvenir supprimé de la base locale.")

    def _build_memory_context(self, query: str, *, document_grounded: bool) -> str | None:
        if not self.memories_enabled_var.get() or self.store is None:
            return None
        try:
            memories = self.store.enabled_memories_for_query(query, limit=5)
        except StorageError as exc:
            self.status_var.set("Impossible de lire les souvenirs : " + str(exc))
            return None
        if document_grounded:
            # Only stylistic preferences/rules may supplement source-grounded RAG
            # answers; personal factual memories must not become false citations.
            memories = [
                item for item in memories
                if any(word in str(item.get("category", "")).casefold() for word in ("préférence", "preference", "règle", "regle"))
            ]
        if not memories:
            return None
        lines = []
        for item in memories:
            category = str(item.get("category", "Général"))
            content = " ".join(str(item["content"]).split())
            lines.append(f"- [{category}] {content}")
        return "\n".join(lines)

    def prepare_library(self) -> None:
        target = filedialog.askdirectory(parent=self.root, title="Choisir le dossier racine de la bibliothèque", mustexist=False)
        if not target:
            return
        try:
            parsed = parse_library_root(target)
        except ValueError as exc:
            messagebox.showwarning("Chemin invalide", str(exc), parent=self.root)
            return
        if not messagebox.askyesno("Confirmer la création", f"Créer les répertoires de la bibliothèque sous :\n\n{parsed}\n\nAucun téléchargement ou logiciel ne sera installé. Continuer ?", parent=self.root):
            return
        self._run_task("Initialisation de la bibliothèque", lambda: initialize_library(parsed))

    def new_conversation(self) -> None:
        if self.busy:
            messagebox.showinfo("Opération en cours", "Attends la fin de la génération avant de créer une nouvelle conversation.", parent=self.root)
            return
        if self.store is not None:
            try:
                self.current_conversation_id = self.store.create_conversation("Nouvelle session")
            except StorageError as exc:
                messagebox.showerror("Nouvelle session", str(exc), parent=self.root)
                return
        else:
            self.current_conversation_id = None
        self.history.clear()
        self.last_assistant_message = ""
        self.chat_text.delete("1.0", "end")
        self.conversation_title_var.set("Nouvelle session")
        self._refresh_sources_panel([], active=False)
        self.status_var.set("Nouvelle session locale prête.")
        self._refresh_conversation_list()
        self.status_var.set("Nouvelle session créée et sauvegardée localement." if self.store is not None else "Nouvelle session en mémoire uniquement.")
        self._set_active_nav("Conversations")

    def clear_conversation(self) -> None:
        if self.busy:
            self.status_var.set("Attends la fin de la génération avant d'effacer la conversation.")
            return
        if not messagebox.askyesno(
            "Effacer la conversation",
            "Supprimer définitivement les messages de cette conversation locale ? Cette action ne peut pas être annulée.",
            parent=self.root,
        ):
            return
        if self.store is not None and self.current_conversation_id:
            try:
                self.store.clear_conversation(self.current_conversation_id)
                self.store.update_conversation_title(self.current_conversation_id, "Nouvelle session")
            except StorageError as exc:
                messagebox.showerror("Effacer la conversation", str(exc), parent=self.root)
                return
        self.history.clear()
        self.last_assistant_message = ""
        self.chat_text.delete("1.0", "end")
        self.conversation_title_var.set("Nouvelle session")
        self._refresh_sources_panel([], active=False)
        self.status_var.set("Nouvelle session locale prête.")
        self._refresh_conversation_list()
        self.status_var.set("Messages de cette conversation effacés.")

    def export_conversation(self) -> None:
        if not self.history:
            messagebox.showinfo("Exporter", "Aucun message à exporter pour le moment.", parent=self.root)
            return
        target = filedialog.asksaveasfilename(
            parent=self.root, title="Exporter la conversation", defaultextension=".txt",
            filetypes=[("Texte", "*.txt"), ("JSON", "*.json")],
            initialfile="conversation-bitnet.txt",
        )
        if not target:
            return
        out = Path(target)
        try:
            if out.suffix.lower() == ".json":
                out.write_text(json.dumps({"app": APP_TITLE, "version": APP_VERSION, "messages": self.history}, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                lines = [f"{APP_TITLE} — export de conversation", ""]
                lines.extend(f"[{entry['timestamp']}] {entry['role'].upper()}\n{entry['content']}\n" for entry in self.history)
                out.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Erreur d'export", str(exc), parent=self.root)
            return
        self.status_var.set(f"Conversation exportée : {out}")
        self._append_message("system", f"Export créé : {out}", save=False)

    def show_help(self) -> None:
        """Show concise, on-demand help instead of persistent instructions in the chat."""
        is_en = self.ui_language == "en"
        title = "BitNet Assistant — Help" if is_en else "BitNet Assistant — Aide"
        if is_en:
            body = (
                "1. Configure the engine\n"
                "Open ‘Engine and models…’, choose llama-completion.exe (recommended) "
                "or another compatible llama.cpp executable, then select a GGUF model. "
                "Save and run ‘Save and test’; it should reply OK.\n\n"
                "2. Chat locally\n"
                "Type a message and press Enter. Use Shift+Enter for a newline. "
                "The message history remains local and is restored on reopening the app.\n\n"
                "3. Create sessions\n"
                "‘+  Nouvelle session’ creates a separate session. "
                "‘Sessions…’ opens your saved sessions. You can reopen, search, select, or delete "
                "a session with the right-click menu.\n\n"
                "4. Attach documents\n"
                "Use ‘＋ Joindre PDF / image / texte’, then ‘Indexer la sélection’. "
                "PDF text is read locally and split into chunks. Changed source files "
                "are shown as needing reindexing.\n\n"
                "5. Use document RAG\n"
                "Enable ‘Utiliser le RAG’, then ask a question. Retrieved sources appear "
                "on the right. The answer is grounded in those passages.\n\n"
                "6. Extractive mode\n"
                "‘Extrait fidèle’ selects one sentence from a matching source without calling "
                "the model; useful for direct quotes and facts.\n\n"
                "7. Summaries\n"
                "‘Résumé rapide (sans modèle)’ is extractive and source-faithful. "
                "‘Résumer avec BitNet’ calls the local model.\n\n"
                "8. Memories\n"
                "Memories are never saved without an explicit click. You can edit, enable, "
                "disable, or delete them. They supplement normal chat, not document evidence.\n\n"
                "9. Settings\n"
                "Use ‘Moteur et modèles…’ for paths, threads, context, temperature, and "
                "automatic startup testing. Use ‘Paramètres…’ for the optional HNSW index, "
                "memory toggle, and theme. HNSW stays off by default on lower-RAM systems.\n\n"
                "10. Privacy\n"
                "No cloud AI API is used. Documents stay on this PC. "
                "Optional web search is always explicit and shows its sources. The Plan toggle is local, does not expose the UI clock, and should stay disabled for normal answers."
            )
        else:
            body = (
                "1. Configuration du moteur\n"
                "Ouvre « Moteur et modèles… », choisis llama-completion.exe (recommandé) "
                "ou un autre exécutable llama.cpp compatible, puis sélectionne un modèle GGUF. "
                "Enregistre et lance « Enregistrer et tester »; le moteur doit répondre OK.\n\n"
                "2. Conversation locale\n"
                "Écris dans le champ et appuie sur Entrée. Maj+Entrée ajoute une ligne. "
                "L’historique est sauvegardé localement et restauré à l’ouverture.\n\n"
                "3. Gérer les sessions\n"
                "Le bouton « +  Nouvelle session » crée une session séparée. "
                "« Sessions… » ouvre les sessions sauvegardées. Tu peux les rechercher, rouvrir "
                "ou supprimer avec le menu contextuel (clic droit).\n\n"
                "4. Joindre et indexer des documents\n"
                "Utilise « ＋ Joindre PDF / image / texte », puis « Indexer la sélection ». "
                "Le PDF est lu localement et découpé en passages. Une source modifiée est "
                "affichée avec « à réindexer ».\n\n"
                "5. Réponses avec RAG\n"
                "Active « Utiliser le RAG » puis pose ta question. Les passages utilisés "
                "apparaissent dans le panneau Sources. La réponse s’appuie sur ces passages.\n\n"
                "6. Mode extrait fidèle\n"
                "Ce mode sélectionne une phrase exacte depuis un passage pertinent sans "
                "appeler le modèle; utile pour vérifier une information ou citer.\n\n"
                "7. Résumé\n"
                "« Résumé rapide (sans modèle) » conserve la formulation source. "
                "« Résumer avec BitNet » appelle le modèle local et peut prendre plus longtemps.\n\n"
                "8. Souvenirs\n"
                "Aucun souvenir n’est créé automatiquement. Tu peux les ajouter, "
                "modifier, activer, désactiver ou supprimer. Ils complètent le chat "
                "normal, mais ne remplacent pas les sources documentaires.\n\n"
                "9. Paramètres et moteur\n"
                "« Moteur et modèles… » gère les chemins, threads, contexte, température et test. "
                "« Paramètres… » gère le moteur d’index (lexique/HNSW optionnel), le thème et le diagnostic. "
                "HNSW reste volontaire et se replie sur lexical si le PC est faible.\n\n"
                "10. Vie privée et sécurité\n"
                "Aucune API IA distante n’est appelée. Les données restent sur ce PC. "
                "La recherche Web est toujours volontaire et affiche ses références. La case Plan est locale, l'horloge UI n'est pas transmise, elle devrait rester désactivée pour les réponses normales. "
                "Vérifie les nombres et données critiques avant utilisation."
            )
        messagebox.showinfo(title, body, parent=self.root)
    def navigate(self, section: str) -> None:
        self.active_section = section
        self._set_active_nav(section)
        if section == "Conversations":
            self.status_var.set("Conversations sauvegardées localement. Sélectionne une ancienne discussion dans la barre latérale.")
        elif section == "Souvenirs":
            self.show_document_tab("memories")
            self.status_var.set("Souvenirs locaux : ajoute ou modifie uniquement les informations que tu souhaites conserver.")
        elif section == "RAG":
            self.show_document_tab("documents")
            indexed_summary = f"{self.rag_index.document_count} document(s), {self.rag_index.chunk_count} passage(s) indexé(s) localement. " if self.rag_index.chunks else "Aucun index documentaire valide. "
            self._append_message("system", "Bibliothèque documentaire locale : " + indexed_summary, save=False)
        elif section == "Paramètres":
            self.show_settings()
        elif section == "Diagnostics":
            self.run_diagnostics()
        elif section == "À propos":
            messagebox.showinfo("À propos de BitNet Windows Assistant", f"{APP_TITLE} {APP_VERSION}\n\nPrototype open source en développement.\n\nÉditeur IA du projet : ChatGPT\n\nChatGPT a contribué à la conception technique et au développement assisté. Le mainteneur humain reste responsable des choix, des tests et de la publication.\n\nLe chat local utilise le connecteur llama.cpp configuré par l'utilisateur. Le RAG lexical local pour PDF textuels et fichiers texte est intégré. L'OCR/vision et la recherche Web restent à intégrer.", parent=self.root)

    def show_settings(self) -> None:
        """Show general application settings without duplicating engine setup."""
        window = tk.Toplevel(self.root)
        is_en = self.ui_language == "en"
        window.title("General settings" if is_en else "Paramètres généraux")
        window.transient(self.root)
        # The settings dialog must remain usable on smaller screens and at high DPI.
        window.resizable(True, True)
        window.configure(bg=COLORS["bg"])
        window.geometry("760x780")
        window.minsize(620, 460)

        # Scrollable content prevents lower settings and action buttons from being clipped.
        shell = tk.Frame(window, bg=COLORS["bg"])
        shell.pack(fill="both", expand=True)
        canvas = tk.Canvas(shell, bg=COLORS["bg"], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        card = self._card(canvas)
        card_window = canvas.create_window((12, 12), window=card, anchor="nw")
        card.pack_propagate(True)

        def _resize_settings_content(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            width = max(300, canvas.winfo_width() - 28)
            canvas.itemconfigure(card_window, width=width)
        canvas.bind("<Configure>", _resize_settings_content)
        card.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        # Mouse-wheel scrolling is scoped to this dialog only.
        def _settings_mousewheel(event):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            return "break"
        canvas.bind("<MouseWheel>", _settings_mousewheel)
        canvas.bind("<Button-4>", _settings_mousewheel)
        canvas.bind("<Button-5>", _settings_mousewheel)
        card.bind("<MouseWheel>", _settings_mousewheel)
        card.bind("<Button-4>", _settings_mousewheel)
        card.bind("<Button-5>", _settings_mousewheel)
        self._label(card, "General settings" if is_en else "Paramètres généraux", font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=18, pady=(12, 5))
        self._label(
            card,
            ("Engine and model settings are available from ‘Engine and models…’ at the top.\n"
             "Conversations, memories and document indexes remain stored locally.") if is_en else
            ("Les réglages du moteur et du modèle se trouvent dans « Moteur et modèles… » en haut.\n"
             "Les conversations, souvenirs et index documentaires restent enregistrés localement."),
            fg=COLORS["muted"], justify="left", wraplength=500,
        ).pack(anchor="w", padx=18, pady=(0, 12))

        self._label(card, "Interface language" if is_en else "Langue UI", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(0, 4))
        language_row = tk.Frame(card, bg=COLORS["white"])
        language_row.pack(anchor="w", padx=18, pady=(0, 10))
        self.language_combo = ttk.Combobox(language_row, state="readonly", width=22, values=("Français", "English"))
        self.language_combo.set("English" if self.ui_language == "en" else "Français")
        self.language_combo.pack(side="left")
        self._button(language_row, "Apply" if is_en else "Appliquer", self._apply_ui_language, compact=True).pack(side="left", padx=8)

        self._label(card, "Appearance" if is_en else "Apparence", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(4, 4))
        theme_row = tk.Frame(card, bg=COLORS["white"])
        theme_row.pack(anchor="w", padx=18, pady=(0, 10))
        self.theme_combo = ttk.Combobox(theme_row, state="readonly", width=18, values=("Clair", "Sombre") if not is_en else ("Light", "Dark"))
        self.theme_combo.set(("Sombre" if self.theme_var.get() == "dark" else "Clair") if not is_en else ("Dark" if self.theme_var.get() == "dark" else "Light"))
        self.theme_combo.pack(side="left")
        def apply_theme_choice():
            choice = self.theme_combo.get().casefold()
            self._apply_theme("dark" if choice in ("sombre", "dark") else "light")
        self._button(theme_row, "Apply" if is_en else "Appliquer", apply_theme_choice, compact=True).pack(side="left", padx=8)

        self.show_recent_var = tk.BooleanVar(value=self.show_recent_var.get())
        recent_check = ttk.Checkbutton(
            card,
            text=("Show recent chats in sidebar" if is_en else "Afficher les chats récents dans la barre latérale"),
            variable=self.show_recent_var,
            command=self._save_recent_visibility,
        )
        recent_check.pack(anchor="w", padx=18, pady=(0, 10))

        self.planning_check = ttk.Checkbutton(
            card,
            text=("Plan mode agent (experimental)" if is_en else "Mode agent plan (expérimental, BitNet local)"),
            variable=self.planning_mode_var,
            command=self._save_planning_mode,
        )
        self.planning_check.pack(anchor="w", padx=18, pady=(0, 10))

        self._label(card, "Moteur d’index RAG" if not is_en else "RAG Index Engine", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(8, 4))
        engine_row = tk.Frame(card, bg=COLORS["white"])
        engine_row.pack(anchor="w", padx=18, pady=(0, 8))
        try:
            from .hnsw_backend import HNSW_AVAILABLE
        except Exception:
            HNSW_AVAILABLE = False
        engine_labels = {"lexical": "Lexical (defaut)", "hnsw": "HNSW hybride (experimental)"}
        engine_values = [engine_labels[v] for v in (["lexical", "hnsw"] if HNSW_AVAILABLE else ["lexical"])]
        self.rag_engine_combo = ttk.Combobox(engine_row, state="readonly", values=engine_values)
        current_engine = "hnsw" if (self.rag_index_engine_var.get() == "hnsw" and HNSW_AVAILABLE) else "lexical"
        self.rag_engine_combo.set(engine_labels[current_engine])
        self.rag_engine_combo.pack(side="left", padx=(0, 10))
        self._button(engine_row, "Appliquer" if not is_en else "Apply", lambda: self._apply_rag_engine(HNSW_AVAILABLE), compact=True).pack(side="left", padx=6)
        if not HNSW_AVAILABLE:
            self._label(card, "HNSW indisponible sur cette machine. Mode lexical par défaut." if not is_en else "HNSW unavailable on this machine. Defaulting to lexical.", fg=COLORS["muted"], font=("Segoe UI", 8), wraplength=300, justify="left").pack(anchor="w", padx=18, pady=(0, 8))

        self._button(card, "Show/hide sessions panel" if is_en else "Afficher/masquer les sessions…", getattr(self, "_toggle_sessions_panel", self.show_sessions), compact=True).pack(anchor="w", padx=18, pady=(0, 10))

        self._label(card, "Assistant instructions (local)" if is_en else "Instructions de l’assistant (locales)", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(2, 4))
        self.system_prompt_text = tk.Text(
            card, height=6, width=70, wrap="word", font=("Segoe UI", 10), relief="flat", bd=0,
            padx=10, pady=8, undo=True,
            bg=COLORS.get("input", COLORS.get("white", "#ffffff")),
            fg=COLORS.get("text", "#172033"), insertbackground=COLORS.get("text", "#172033"),
            selectbackground=COLORS.get("accent", "#167d78"),
        )
        self.system_prompt_text.pack(fill="x", padx=18, pady=(0, 5))
        default_prompt = ((getattr(self.engine_config, "system_prompt", "") if self.engine_config else "") or DEFAULT_SYSTEM_PROMPT)
        self.system_prompt_text.insert("1.0", default_prompt)
        
        self._button(card, "Save assistant instructions" if is_en else "Enregistrer les instructions", lambda: self._save_system_prompt(), primary=True).pack(anchor="w", padx=18, pady=(0, 10))

        db_path = default_database_path()
        self._label(card, "Local DB path" if is_en else "Base locale", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(0, 4))
        active_db_path = self.store.path if self.store is not None else db_path
        path_label = self._label(card, str(active_db_path), fg=COLORS["text"], justify="left", wraplength=540)
        path_label.pack(anchor="w", padx=18, pady=(0, 8))

        backup_row = tk.Frame(card, bg=COLORS["white"])
        backup_row.pack(fill="x", padx=18, pady=(0, 8))
        self._button(backup_row, "Create backup…" if is_en else "Créer une sauvegarde…", lambda: self._create_database_backup(active_db_path), primary=True).pack(side="left", padx=(0, 8))
        self._button(backup_row, "Check database" if is_en else "Vérifier la base", lambda: self._check_database_integrity(active_db_path)).pack(side="left")

        buttons = tk.Frame(card, bg=COLORS["white"])
        buttons.pack(fill="x", padx=18, pady=(0, 16))
        self._button(buttons, "Copy path" if is_en else "Copier le chemin", lambda: self._copy_settings_path(active_db_path)).pack(side="left", padx=(0, 8))
        self._button(buttons, "Open folder" if is_en else "Ouvrir le dossier", lambda: self._open_settings_folder(active_db_path)).pack(side="left", padx=(0, 8))
        self._button(buttons, "Close" if is_en else "Fermer", window.destroy, primary=True).pack(side="right")
        window.grab_set()

    def _save_system_prompt(self) -> None:
        """Persist the user's local assistant instructions in the engine settings."""
        if self.engine_config is None or not hasattr(self, "system_prompt_text"):
            messagebox.showwarning("BitNet", "Configure le moteur avant d’enregistrer les instructions.", parent=self.root)
            return
        from dataclasses import replace
        value = self.system_prompt_text.get("1.0", "end-1c").strip()
        if len(value) > 6000:
            messagebox.showwarning("BitNet", "Les instructions doivent faire 6 000 caractères maximum.", parent=self.root)
            return
        try:
            updated = replace(self.engine_config, system_prompt=value)
            save_config(updated)
            self.engine_config = updated
            messagebox.showinfo("BitNet", "Instructions enregistrées localement.", parent=self.root)
        except Exception as exc:
            messagebox.showerror("BitNet", f"Impossible d’enregistrer les instructions : {exc}", parent=self.root)

    def _apply_rag_engine(self, available: bool) -> None:
        value = self.rag_engine_combo.get()
        selected = "hnsw" if value.startswith("HNSW") else "lexical"
        if selected == "hnsw" and not available:
            selected = "lexical"
            messagebox.showwarning("HNSW indisponible", "HNSW n'est pas disponible sur cette machine. Mode lexical conserve.", parent=self.root)
        if selected == "hnsw":
            try:
                import ctypes

                class _MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64)]

                st = _MS()
                st.dwLength = ctypes.sizeof(_MS)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
                ram_mb = int(st.ullTotalPhys / (1024 * 1024))
            except Exception:
                ram_mb = 8192
            if ram_mb < 8192:
                messagebox.showwarning("Attention HNSW", f"Mémoire détectée : {ram_mb} MB. Les machines à moins de 8 Go seront plus lentes. L'option reste activable.", parent=self.root)
        self.rag_index_engine_var.set(selected)
        if self.store is not None:
            try:
                self.store.set_setting("rag_index_engine", selected)
            except Exception:
                pass
        self.status_var.set(("Index RAG : HNSW sélectionné (expérimental)" if selected == "hnsw" else "Index RAG : lexical (par défaut)"))

    def _load_default_rag_chunks(self) -> list[DocumentChunk]:
        path = Path(__file__).with_name("default_rag.json")
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        source_name = str(data.get("default_source", "Base documentaire interne")).strip() or "Base documentaire interne"
        source_path = f"internal:{source_name}"
        chunks: list[DocumentChunk] = []
        for index, entry in enumerate(data.get("chunks", [])):
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            topic = str(entry.get("topic", "Référence")).strip() or "Référence"
            if not text:
                continue
            chunks.append(
                DocumentChunk(
                    source_path=source_path,
                    filename=source_name,
                    locator=f"{topic} - passage {index + 1}",
                    text=text,
                    chunk_index=index,
                )
            )
        return chunks

    def _save_planning_mode(self) -> None:
        if self.store is not None:
            try:
                self.store.set_setting("planning_mode", "1" if self.planning_mode_var.get() else "0")
            except Exception:
                pass

    def _apply_recent_visibility(self) -> None:
        """Show or hide the optional recent-chat list without deleting any sessions."""
        if self.show_recent_var.get():
            self.recent_section.pack(fill="x", before=self.sessions_button)
        else:
            self.recent_section.pack_forget()
        # The sessions shortcut remains visible even when the recent list is hidden.

    def _save_recent_visibility(self) -> None:
        """Persist the sidebar preference locally and apply it immediately."""
        self._apply_recent_visibility()
        if self.store is not None:
            try:
                self.store.set_setting("show_recent_chats", "1" if self.show_recent_var.get() else "0")
            except StorageError as exc:
                self.status_var.set(f"Préférence non enregistrée : {exc}")
                return
        is_en = self.ui_language == "en"
        self.status_var.set("Recent chats visibility saved." if is_en else "Affichage des chats récents enregistré.")

    def show_sessions(self) -> None:
        """Open a compact picker for saved local conversations, including hidden recent chats."""
        if self.store is None:
            messagebox.showwarning("Sessions", "Stockage local indisponible.", parent=self.root)
            return
        try:
            conversations = self.store.list_conversations(limit=500)
        except StorageError as exc:
            messagebox.showerror("Sessions", str(exc), parent=self.root)
            return
        is_en = self.ui_language == "en"
        window = tk.Toplevel(self.root)
        window.title("Saved sessions" if is_en else "Sessions sauvegardées")
        window.transient(self.root)
        window.geometry("520x420")
        window.minsize(420, 300)
        window.configure(bg=COLORS["bg"])
        card = self._card(window)
        card.pack(fill="both", expand=True, padx=12, pady=12)
        self._label(card, "Saved sessions" if is_en else "Sessions sauvegardées", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 8))
        listing = tk.Listbox(card, bg=COLORS["white"], fg=COLORS["text"], font=("Segoe UI", 10), relief="flat", exportselection=False)
        listing.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        ids: list[str] = []
        for entry in conversations:
            title = " ".join(str(entry.get("title", "")).split()) or ("New chat" if is_en else "Nouveau chat")
            title = title if len(title) <= 48 else title[:47] + "…"
            count = int(entry.get("message_count", 0))
            stamp = str(entry.get("updated_at", ""))[:10]
            listing.insert("end", f"{title}  ·  {count} msg  ·  {stamp}")
            ids.append(str(entry["id"]))
        if self.current_conversation_id in ids:
            listing.selection_set(ids.index(self.current_conversation_id))
        def open_selected(_event=None):
            selected = listing.curselection()
            if not selected:
                return
            self._load_conversation(ids[int(selected[0])])
            window.destroy()
        listing.bind("<Double-Button-1>", open_selected)
        row = tk.Frame(card, bg=COLORS["white"])
        row.pack(fill="x", padx=12, pady=(0, 12))
        self._button(row, "Open" if is_en else "Ouvrir", open_selected, primary=True, compact=True).pack(side="right", padx=(6, 0))
        self._button(row, "Close" if is_en else "Fermer", window.destroy, compact=True).pack(side="right")
        listing.focus_set()
        window.grab_set()

    def _apply_ui_language(self) -> None:
        """Switch the most-used navigation/composer controls between French and English."""
        if not hasattr(self, "language_combo"):
            return
        self.ui_language = "en" if self.language_combo.get() == "English" else "fr"
        translations = {
            "fr": {"+  Nouvelle session":"+  New conversation", "▤  Conversations":"Chats", "✦  Souvenirs":"Mém.", "▣  RAG":"RAG", "⚙  Paramètres":"Régl.", "⌁  Diagnostics":"Diag.", "ⓘ  À propos":"À prop.", "CHATS RÉCENTS":"RECENT CHATS", "Nouvelle session":"New conversation", "Moteur et modèles…":"Engine and models…", "＋ Joindre PDF / image / texte":"＋ Attach PDF / image / text", "RAG":"Use local RAG", "Web":"Web search", "Envoyer  ➤":"Send  ➤", "Sources et documents":"Sources and documents", "Sources utilisées":"Used sources", "Documents":"Documents", "Aucune source utilisée":"No sources used", "Entrée : envoyer  •  Maj+Entrée : nouvelle ligne":"Enter: send  •  Shift+Enter: new line", "Écrivez votre message ici…":"Type your message here…", "Discussion générale":"General chat", "Moteur non connecté":"Engine not connected", "Prêt. Moteur de chat non connecté.":"Ready. Chat engine not connected."},
            "en": {"+  New conversation":"+  Nouvelle session", "▤  Chats":"Chats", "✦  Memories":"Mem.", "▣  RAG":"RAG", "⚙  Settings":"Settings", "⌁  Diagnostics":"Diag.", "ⓘ  About":"About", "RECENT CHATS":"CHATS RÉCENTS", "New conversation":"Nouvelle session", "Engine and models…":"Moteur et modèles…", "＋ Attach PDF / image / text":"＋ Joindre PDF / image / texte", "Use local RAG":"RAG", "Web search":"Web", "Send  ➤":"Envoyer  ➤", "Sources and documents":"Sources et documents", "Used sources":"Sources utilisées", "No sources used":"Aucune source utilisée", "Enter: send  •  Shift+Enter: new line":"Entrée : envoyer  •  Maj+Entrée : nouvelle ligne", "Type your message here…":"Écrivez votre message ici…", "General chat":"Discussion générale", "Engine not connected":"Moteur non connecté", "Ready. Chat engine not connected.":"Prêt. Moteur de chat non connecté."}
        }
        mapping = translations["fr" if self.ui_language == "en" else "en"]
        # Toggle the labels in place. Store the original label once per widget so
        # switching repeatedly never compounds translations.
        def walk(widget):
            try:
                text = widget.cget("text")
                if text and text in mapping:
                    widget.configure(text=mapping[text])
            except (tk.TclError, TypeError):
                pass
            for child in widget.winfo_children(): walk(child)
        walk(self.root)
        if self.ui_language == "en":
            self.root.title(f"{APP_TITLE} v{APP_VERSION} — Chat + RAG")
            self.status_var.set("Interface language set to English. Reopen Settings to refresh its labels.")
        else:
            self.root.title(f"{APP_TITLE} v{APP_VERSION} — Chat + RAG")
            self.status_var.set("Langue de l’interface réglée sur le français. Rouvre Paramètres pour actualiser les libellés.")
        # Navigation callbacks use stable French keys, while visible labels are translated.

    def _create_database_backup(self, database_path: Path) -> None:
        """Let the user choose a destination for a consistent SQLite backup."""
        from datetime import datetime
        is_en = self.ui_language == "en"
        suggested = f"BitNet-Assistant-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.sqlite3"
        backup_dir = default_backup_directory()
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            initial_dir = str(backup_dir)
        except OSError:
            # Restricted installation folders must not prevent manual backups.
            initial_dir = str(Path.home())
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save local database backup" if is_en else "Enregistrer une sauvegarde de la base locale",
            initialdir=initial_dir,
            initialfile=suggested,
            defaultextension=".sqlite3",
            filetypes=[("SQLite database", "*.sqlite3"), ("All files", "*.*")],
        )
        if not destination:
            return
        try:
            if self.store is None:
                raise StorageError("La base locale n'est pas disponible." if not is_en else "The local database is unavailable.")
            created = self.store.backup_to(destination)
            messagebox.showinfo(
                "Backup created" if is_en else "Sauvegarde créée",
                (f"A consistent backup was created and verified:\n{created}" if is_en else f"Une sauvegarde cohérente a été créée et vérifiée :\n{created}"),
                parent=self.root,
            )
            self.status_var.set("Backup created and verified." if is_en else "Sauvegarde créée et vérifiée.")
        except StorageError as exc:
            messagebox.showerror("Backup failed" if is_en else "Échec de la sauvegarde", str(exc), parent=self.root)

    def _check_database_integrity(self, database_path: Path) -> None:
        is_en = self.ui_language == "en"
        try:
            if self.store is None:
                raise StorageError("La base locale n'est pas disponible." if not is_en else "The local database is unavailable.")
            results = self.store.integrity_check()
            good = results == ["ok"]
            message = ("Database integrity check passed (SQLite: ok)." if is_en else "Vérification d'intégrité réussie (SQLite : ok).") if good else "\n".join(results)
            if good:
                messagebox.showinfo("Database check" if is_en else "Vérification de la base", message, parent=self.root)
            else:
                messagebox.showwarning("Database check" if is_en else "Vérification de la base", message, parent=self.root)
        except StorageError as exc:
            messagebox.showerror("Database check failed" if is_en else "Échec de vérification", str(exc), parent=self.root)

    def _copy_settings_path(self, path: Path) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(str(path))
            self.status_var.set("Chemin de la base locale copié dans le presse-papiers.")
        except tk.TclError as exc:
            messagebox.showerror("Presse-papiers indisponible", str(exc), parent=self.root)

    def _open_settings_folder(self, path: Path) -> None:
        folder = path.parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(folder)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(folder)])
        except (OSError, tk.TclError) as exc:
            messagebox.showerror("Impossible d'ouvrir le dossier", str(exc), parent=self.root)

    def _update_engine_status(self) -> None:
        configured = self.engine_config is not None
        if configured:
            try:
                validate_config(self.engine_config)
                if self.engine_test_passed:
                    self.engine_status.configure(text="Moteur opérationnel", bg=COLORS["success_bg"], fg=COLORS["success"])
                    self.connection_summary_var.set("Moteur local testé — prêt à générer des réponses.")
                else:
                    self.engine_status.configure(text="Moteur configuré — à tester", bg=COLORS["warning_bg"], fg=COLORS["warning"])
                    self.connection_summary_var.set("Exécutable et modèle trouvés ; lance un test pour confirmer le chargement.")
                return
            except EngineError:
                pass
        self.engine_status.configure(text="Moteur non connecté", bg=COLORS["warning_bg"], fg=COLORS["warning"])
        self.connection_summary_var.set("Configure llama-completion.exe (recommandé) ou llama-cli.exe et le fichier modèle GGUF pour commencer.")

    def _start_engine_automatically(self) -> None:
        """Run a local startup inference only when the user opted in."""
        config = self.engine_config
        if config is None or not config.auto_start:
            return
        try:
            validate_config(config)
        except EngineError as exc:
            self.engine_test_passed = False
            self._update_engine_status()
            message = (
                "Démarrage automatique ignoré : la configuration du moteur local "
                "n'est pas valide. Ouvre « Moteur et modèles… » pour corriger les chemins.\n\n"
                + str(exc)
            )
            self._append_message("system", message, save=False)
            self.status_var.set("Démarrage automatique impossible — configuration à vérifier.")
            return
        self._run_task(
            "Démarrage automatique du moteur BitNet",
            lambda: generate_response(config, [], "Réponds uniquement par le mot OK."),
        )

    def configure_engine(self) -> None:
        if self.busy:
            messagebox.showinfo("Opération en cours", "Attends la fin de l'opération en cours avant de configurer le moteur.", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Configurer le moteur BitNet local")
        window.transient(self.root)
        window.grab_set()
        window.configure(bg=COLORS["bg"])
        window.resizable(False, False)

        current = self.engine_config or suggested_config()
        body = tk.Frame(window, bg=COLORS["bg"], padx=16, pady=14)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Connexion au moteur local", bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))
        tk.Label(body, text="Aucun service distant n'est utilisé. Sélectionne l'exécutable llama.cpp et un modèle GGUF déjà présents sur ce PC.", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 10), wraplength=620, justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        executable_var = tk.StringVar(value=current.executable)
        model_var = tk.StringVar(value=current.model_path)
        threads_var = tk.StringVar(value=str(current.threads))
        context_var = tk.StringVar(value=str(current.context_size))
        gpu_var = tk.StringVar(value=str(current.gpu_layers))
        tokens_var = tk.StringVar(value=str(current.max_tokens))
        temp_var = tk.StringVar(value=str(current.temperature))
        override_var = tk.BooleanVar(value=current.tokenizer_override)
        auto_start_var = tk.BooleanVar(value=current.auto_start)

        def path_row(row: int, label: str, var: tk.StringVar, browse_command: Callable[[], None]) -> None:
            tk.Label(body, text=label, bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
            tk.Entry(body, textvariable=var, width=64).grid(row=row, column=1, sticky="ew", pady=5)
            tk.Button(body, text="Parcourir…", command=browse_command).grid(row=row, column=2, padx=(8, 0), pady=5)

        def browse_executable() -> None:
            value = filedialog.askopenfilename(parent=window, title="Choisir llama-completion.exe ou llama-cli.exe", filetypes=[("Exécutables llama.cpp", "*.exe"), ("Tous les fichiers", "*.*")])
            if value:
                executable_var.set(value)

        def browse_model() -> None:
            value = filedialog.askopenfilename(parent=window, title="Choisir le modèle BitNet GGUF", filetypes=[("Modèles GGUF", "*.gguf"), ("Tous les fichiers", "*.*")])
            if value:
                model_var.set(value)

        path_row(2, "Exécutable llama.cpp", executable_var, browse_executable)
        path_row(3, "Modèle GGUF", model_var, browse_model)
        fields = [
            ("Threads CPU", threads_var, 1, 256),
            ("Contexte (tokens)", context_var, 256, 131072),
            ("Couches GPU", gpu_var, 0, 4096),
            ("Tokens générés max.", tokens_var, 1, 32768),
        ]
        for index, (label, var, low, high) in enumerate(fields):
            row = 4 + index // 2
            col = (index % 2) * 2
            tk.Label(body, text=label, bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI", 9)).grid(row=row, column=col, sticky="w", pady=7, padx=(0, 8))
            tk.Spinbox(body, from_=low, to=high, textvariable=var, width=10).grid(row=row, column=col + 1, sticky="w", pady=7, padx=(0, 18))
        tk.Label(body, text="Température", bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI", 9)).grid(row=6, column=0, sticky="w", pady=7)
        tk.Spinbox(body, from_=0.0, to=2.0, increment=0.1, textvariable=temp_var, width=10).grid(row=6, column=1, sticky="w", pady=7)
        tk.Checkbutton(body, text="Forcer tokenizer.ggml.pre = llama-bpe (modèle BitNet testé)", variable=override_var, bg=COLORS["bg"], activebackground=COLORS["bg"], fg=COLORS["text"]).grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 5))
        tk.Checkbutton(
            body,
            text="Tester automatiquement le moteur local à l'ouverture de l'application",
            variable=auto_start_var,
            bg=COLORS["bg"], activebackground=COLORS["bg"], fg=COLORS["text"],
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 2))
        tk.Label(
            body,
            text="Désactivé par défaut. Coche puis enregistre pour effectuer un test local au prochain démarrage. Avec llama-completion, le modèle est chargé pour ce test puis le processus se ferme ; il ne reste pas en mémoire.",
            bg=COLORS["bg"], fg=COLORS["muted"], wraplength=620, justify="left", font=("Segoe UI", 8),
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(0, 8))
        tk.Label(body, text="Format de conversation de ce connecteur : User / Assistant / <|eot_id|>, identique au modèle testé. Pour un autre modèle, il faudra vérifier son format de chat.", bg=COLORS["bg"], fg=COLORS["muted"], wraplength=620, justify="left", font=("Segoe UI", 8)).grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 10))

        def read_config() -> EngineConfig:
            try:
                return EngineConfig(
                    executable=executable_var.get().strip().strip('"'), model_path=model_var.get().strip().strip('"'),
                    context_size=int(context_var.get()), threads=int(threads_var.get()),
                    gpu_layers=int(gpu_var.get()), max_tokens=int(tokens_var.get()),
                    temperature=float(temp_var.get()), tokenizer_override=bool(override_var.get()),
                    auto_start=bool(auto_start_var.get()),
                )
            except ValueError as exc:
                raise EngineError("Un des réglages numériques n'est pas valide.") from exc

        def commit() -> EngineConfig | None:
            try:
                config = read_config()
                validate_config(config)
                save_config(config)
            except (EngineError, OSError) as exc:
                messagebox.showerror("Configuration invalide", str(exc), parent=window)
                return None
            self.engine_config = config
            self.engine_test_passed = False
            self._update_engine_status()
            self.status_var.set("Configuration locale enregistrée ; le moteur n'a pas encore été testé.")
            # Keep transient configuration state in the status bar, not in chat history;
            # otherwise an obsolete "test still required" message remains after success.
            return config

        def save_and_close() -> None:
            if commit() is not None:
                window.destroy()

        def test_engine() -> None:
            config = commit()
            if config is None:
                return
            window.destroy()
            self._run_task("Test du moteur BitNet", lambda: generate_response(config, [], "Réponds uniquement par le mot OK."))

        actions = tk.Frame(body, bg=COLORS["bg"])
        actions.grid(row=11, column=0, columnspan=3, sticky="e", pady=(6, 0))
        tk.Button(actions, text="Enregistrer", command=save_and_close, padx=12, pady=6).pack(side="left", padx=4)
        tk.Button(actions, text="Enregistrer et tester", command=test_engine, bg=COLORS["blue"], fg="white", activebackground=COLORS["blue_hover"], activeforeground="white", relief="flat", padx=12, pady=6).pack(side="left", padx=4)
        tk.Button(actions, text="Annuler", command=window.destroy, padx=12, pady=6).pack(side="left", padx=4)
        body.columnconfigure(1, weight=1)
        window.update_idletasks()
        window.geometry(f"{max(740, window.winfo_reqwidth())}x{window.winfo_reqheight()}")
        window.wait_visibility()
        window.focus_set()

    def run_diagnostics(self) -> None:
        """Run diagnostics and show results in a separate window, not in chat."""
        self._run_task("Diagnostic rapide", run_diagnostics)

    def _show_diagnostics_view(self, title: str, text: str) -> None:
        """Open a dedicated diagnostics view with a direct return to conversation."""
        window = tk.Toplevel(self.root)
        is_dark = self.theme_var.get() == "dark"
        bg = "#1F2937" if is_dark else "#F5F7FA"
        panel = "#1F2937" if is_dark else "#FFFFFF"
        fg = "#F3F4F6" if is_dark else "#16223B"
        muted = "#B6C2D2" if is_dark else "#687895"
        window.title("Diagnostics BitNet")
        window.transient(self.root)
        window.geometry("820x620")
        window.minsize(620, 420)
        window.configure(bg=bg)
        header = tk.Frame(window, bg=bg, padx=18, pady=14)
        header.pack(fill="x")
        tk.Label(header, text=title, bg=bg, fg=fg, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(header, text="Résultats techniques séparés de la conversation.", bg=bg, fg=muted,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
        body = tk.Frame(window, bg=panel, padx=10, pady=10)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        report = tk.Text(body, wrap="word", bg=panel, fg=fg, insertbackground=fg,
                         selectbackground="#315A9D" if is_dark else "#BBD0F5",
                         relief="flat", padx=10, pady=10, font=("Consolas", 10))
        scrollbar = tk.Scrollbar(body, command=report.yview)
        report.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        report.pack(side="left", fill="both", expand=True)
        report.insert("1.0", text)
        report.configure(state="disabled")
        footer = tk.Frame(window, bg=bg, padx=16, pady=10)
        footer.pack(fill="x")
        self._button(footer, "↩ Retour à la conversation", window.destroy, primary=True, compact=True).pack(side="right")

    def run_cpp_test(self) -> None:
        self._run_task("Test C++ temporaire", run_cpp_smoke_test)

    def _run_task(self, title: str, task: Callable[[], Any]) -> None:
        if self.busy:
            self.status_var.set("Une opération est déjà en cours.")
            return
        self.busy = True
        self.status_var.set(f"En cours : {title}…")
        self.send_button.configure(state="disabled")
        self.root.update_idletasks()

        def worker() -> None:
            try:
                result = task()
                error = None
            except Exception as exc:  # surface actionable errors in the GUI
                result, error = None, str(exc)
            self.root.after(0, self._finish_task, title, result, error)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_task(self, title: str, result: Any, error: str | None) -> None:
        self.busy = False
        self.send_button.configure(state="normal")
        if error:
            self._append_message("system", f"{title} — erreur\n{error}", save=False)
            self.status_var.set(f"Échec : {title}")
            return
        if title == "Indexer les documents" and isinstance(result, RAGIndex):
            updated_keys = {str(Path(p).resolve()).casefold() for p in result.indexed_files}
            # Merge only successfully indexed documents; keep all other valid documents intact.
            kept_chunks = [c for c in self.rag_index.chunks if str(Path(c.source_path).resolve()).casefold() not in updated_keys]
            kept_paths = [p for p in self.rag_index.indexed_files if str(Path(p).resolve()).casefold() not in updated_keys]
            stale_keys = {str(Path(p).resolve()).casefold() for p in self.rag_index.stale_files}
            kept_chunks = [c for c in kept_chunks if str(Path(c.source_path).resolve()).casefold() not in stale_keys]
            self.rag_index = RAGIndex(
                chunks=kept_chunks + result.chunks,
                indexed_files=kept_paths + result.indexed_files,
                warnings=list(dict.fromkeys(self.rag_index.warnings + result.warnings)),
                stale_files=[p for p in self.rag_index.stale_files if str(Path(p).resolve()).casefold() not in updated_keys],
            )
            requested_engine = "hnsw" if self.rag_index_engine_var.get() == "hnsw" else "lexical"
            self.rag_index.engine = requested_engine
            if requested_engine == "hnsw":
                try:
                    from .hnsw_backend import HNSW_AVAILABLE, HnswIndex
                    if HNSW_AVAILABLE:
                        hnsw = HnswIndex()
                        if hnsw.build(self.rag_index.chunks):
                            self.rag_index.hnsw_index = hnsw
                            result.warnings.append("Moteur d'index HNSW local activé pour la recherche (expérimental).")
                        else:
                            self.rag_index.engine = "lexical"
                            result.warnings.append("Échec de l'index HNSW : repli automatique sur l'index lexical.")
                    else:
                        self.rag_index.engine = "lexical"
                        result.warnings.append("HNSW indisponible : installez hnswlib (experimental). Utilisation de l'index lexical.")
                except Exception as exc:
                    self.rag_index.engine = "lexical"
                    result.warnings.append("HNSW indisponible : " + str(exc) + ". Utilisation de l'index lexical.")
            else:
                self.rag_index.hnsw_index = None
            if self.store is not None:
                try:
                    persist_warnings = self.store.save_rag_documents(result)
                    if persist_warnings:
                        result.warnings.extend(persist_warnings)
                except StorageError as exc:
                    result.warnings.append("L'index est actif pour cette session mais n'a pas pu être conservé après fermeture : " + str(exc))
            self._refresh_documents_list()
            self._refresh_mode_status()
            self._refresh_sources_panel([], active=False, note="Index local actualisé. Active « Utiliser le RAG » pour rechercher les passages pertinents lors d'une question.")
            message = f"Indexation terminée : {result.document_count} document(s), {result.chunk_count} passage(s). L'index et les extraits sont conservés dans la base locale ; les fichiers sources restent à leur emplacement d'origine. Aucune donnée n'a été envoyée sur Internet."
            if result.warnings:
                message += "\n\nRemarques :\n• " + "\n• ".join(result.warnings[:20])
                if len(result.warnings) > 20:
                    message += f"\n• … et {len(result.warnings) - 20} autre(s) remarque(s)."
            self._append_message("system", message, save=False)
            self.status_var.set(f"RAG local prêt : {result.chunk_count} passages dans {result.document_count} document(s).")
            return
        if title in ("Résumé rapide", "Résumé avec BitNet") and isinstance(result, SummaryResult):
            answer = append_summary_source_references(result.text, result.source_hits)
            self._append_message("assistant", answer, display_name=result.mode)
            source_note = (
                "Sources consultées localement pour le résumé. Les identifiants [S1], [S2]… correspondent aux fichiers et pages/lignes visibles ci-dessous. "
                "La présence d'une source indique son utilisation par le pipeline, pas une garantie d'exactitude absolue du résumé."
            )
            self._refresh_sources_panel(result.source_hits, active=bool(result.source_hits), note=source_note)
            self.show_document_tab("sources")
            if title == "Résumé avec BitNet":
                self.engine_test_passed = True
                self._update_engine_status()
                self.status_var.set(f"Résumé BitNet terminé : {result.batch_count} groupe(s), {result.model_calls} génération(s) locale(s). Vérifie les sources et le contenu.")
            else:
                self.status_var.set("Résumé rapide terminé — phrases originales extraites localement, sans appel au modèle.")
            return
        if title.startswith("Réponse BitNet") and isinstance(result, dict) and "answer" in result:
            sources = result.get("sources", [])
            answer = append_retrieved_source_references(str(result["answer"]), sources)
            self._append_message("assistant", answer)
            self._refresh_sources_panel(sources, active=bool(sources))
            self.engine_test_passed = True
            self._update_engine_status()
            self.status_var.set("Réponse reçue du moteur BitNet local." + (" avec des sources Web." if result.get("web_results") else ""))
            return
        if title == "Réponse BitNet" and isinstance(result, str):
            self._append_message("assistant", result)
            self._refresh_sources_panel([], active=False)
            self.engine_test_passed = True
            self._update_engine_status()
            self.status_var.set("Réponse reçue du moteur BitNet local.")
            return
        if title in ("Test du moteur BitNet", "Démarrage automatique du moteur BitNet") and isinstance(result, str):
            self.engine_test_passed = True
            self._update_engine_status()
            if title == "Démarrage automatique du moteur BitNet":
                self.engine_status_var.set("Moteur BitNet : auto-test OK.")
                self.status_var.set("Démarrage automatique réussi — moteur BitNet testé.")
            else:
                self.engine_status_var.set("Moteur BitNet : test OK.")
                self.status_var.set("Moteur BitNet opérationnel — test d'inférence réussi.")
            return
        if title == "Diagnostic rapide":
            if isinstance(result, dict) and "checks" in result:
                report_text = format_diagnostic_report(result)
            elif isinstance(result, dict):
                report_text = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                report_text = str(result)
            self._show_diagnostics_view("Diagnostic rapide", report_text)
            self.status_var.set("Diagnostic terminé — résultats ouverts dans la vue Diagnostics.")
            return
        if isinstance(result, dict) and "checks" in result:
            text = format_diagnostic_report(result)
            if "build_check" in result:
                text += "\n\nTest C++ : " + json.dumps(result["build_check"], ensure_ascii=False, indent=2)
        elif isinstance(result, dict):
            text = json.dumps(result, ensure_ascii=False, indent=2)
        elif isinstance(result, (list, tuple)):
            text = "Opération terminée.\n" + "\n".join(str(path) for path in result)
            text += "\nAucun téléchargement, installation ou réglage Windows n'a été effectué."
        else:
            text = str(result)
        self._append_message("system", f"=== {title} ===\n{text}", save=False)
        status = "Terminé"
        if isinstance(result, dict) and result.get("status") not in (None, "passed", "found"):
            status += f" — {result.get('status')}"
        self.status_var.set(f"{status} : {title}")


def _enable_windows_dpi_awareness() -> bool:
    """Ask Windows for per-monitor DPI awareness before creating Tk widgets.

    Returns False on non-Windows platforms or when Windows refuses the setting.
    This avoids Windows bitmap-scaling blur where supported; actual rendering
    still needs verification on the user's Windows/display configuration.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (HANDLE)-4
        setter = getattr(ctypes.windll.user32, "SetProcessDpiAwarenessContext", None)
        if setter is not None and setter(ctypes.c_void_p(-4)):
            return True
        # Older Windows fallback: system DPI awareness.
        legacy = getattr(ctypes.windll.user32, "SetProcessDPIAware", None)
        if legacy is not None:
            return bool(legacy())
    except (AttributeError, OSError, ValueError):
        return False
    return False


def main() -> None:
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    ChatAssistantWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()