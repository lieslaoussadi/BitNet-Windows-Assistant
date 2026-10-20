import sys, time
sys.path.insert(0, "src")
from bitnet_windows.engine import EngineConfig, validate_config, generate_response

rt = r"C:\AI\bitnet\BitNet-runtime"
cfg = EngineConfig(
    executable=rt + r"\bin\llama-completion.exe",
    model_path=rt + r"\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf",
    context_size=1024, threads=8, gpu_layers=0, max_tokens=48,
    temperature=0.1, tokenizer_override=True, auto_start=False,
)
validate_config(cfg, require_files=True)

prompts = {
    "factual_terse": "Réponds en français, de manière concise et factuelle. Si tu n'es pas sûr, dis « Je ne sais pas ».",
    "strict_rag": "Réponds uniquement à partir de fait simple et vérifiable. Pas d'explication longue. Si incertitude, « Je ne sais pas ».",
    "calculation_focus": "Vérification mathématique : donne uniquement le résultat numérique (racine carrée, addition).",
}

questions = [
    "Quelle est la racine carrée de 144 ?",
    "Donne la capitale exacte de la Finlande.",
]

for pname, system in prompts.items():
    print(f"\n=== PROMPT: {pname} ===")
    for q in questions:
        full = f"Consignes de comportement : {system}\n\n{q}" if system else q
        t = time.perf_counter()
        out = generate_response(cfg, [], full)
        dt = time.perf_counter() - t
        print(f" Q: {q}")
        print(f" A: {out.strip()}")
        print(f" t={dt:.2f}s")