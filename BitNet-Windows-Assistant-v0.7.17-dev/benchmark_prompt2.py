import sys, time
sys.path.insert(0, "src")
from bitnet_windows.engine import EngineConfig, validate_config, generate_response
rt = r"C:\AI\bitnet\BitNet-runtime"
cfg = EngineConfig(executable=rt+r"\bin\llama-completion.exe", model_path=rt+r"\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf", context_size=1024, threads=8, gpu_layers=0, max_tokens=40, temperature=0.05, tokenizer_override=True, auto_start=False)
validate_config(cfg, require_files=True)
tests = [
 "Consignes : Réponds en français. sqrt(144)=12 et 12^2=144. Calcule correctement et donne le résultat. Question : Quelle est la racine carrée de 144 ?",
 "Consignes : réponds juste \"12\". Question : racine carrée de 144.",
 "Consignes : vérifie mentalement et assure-toi de ne pas confondre racine et carré. Question : Quelle est la racine carrée de 144 ?",
]
for p in tests:
    t=time.perf_counter()
    out=generate_response(cfg, [], p)
    print("PROMPT:", p)
    print("A:", out.strip(), f"({time.perf_counter()-t:.2f}s)")