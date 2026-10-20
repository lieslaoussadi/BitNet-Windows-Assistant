from pathlib import Path
import time
from bitnet_windows.engine import generate_response, suggested_config

from bitnet_windows.engine import config_from_mapping
cfg = config_from_mapping({
    "threads": 8,
    "context_size": 2048,
    "max_tokens": 48,
    "temperature": 0.1,
    "tokenizer_override": True,
})
report = Path(r"C:\AI\bitnet\BitNet-TestReport.md")
prompts = [
    "Bonjour.",
    "Que vaut 2+2 ? Reponds tres court.",
    "Qui est le president inconnu d'une planete lointaine ?",
    "Reponds 'OUI' sans explication.",
    "Donne une seule phrase exacte pour verifier ton comportement.",
]
with report.open("w", encoding="utf-8") as f:
    f.write("# Rapport de test BitNet (local)\n\n")
    f.write("## Parametres utilises\n")
    f.write(f"- Executable: `{cfg.executable}`\n")
    f.write(f"- Modele: `{cfg.model_path}`\n")
    f.write(f"- Contexte: `{cfg.context_size}`\n")
    f.write(f"- Threads: `{cfg.threads}`\n")
    f.write(f"- Tokens max: `{cfg.max_tokens}`\n")
    f.write(f"- Temperature: `{cfg.temperature}`\n\n")
    f.write("## Resultats\n")
    for i, prompt in enumerate(prompts, 1):
        started = time.time()
        response = generate_response(cfg, [], prompt)
        elapsed = time.time() - started
        f.write(f"### Cas {i}: {prompt}\n")
        f.write(f"- duree: {elapsed:.1f}s\n")
        f.write(f"- reponse: {response}\n\n")
print("Report written:", report)