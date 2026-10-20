# BitNet Windows Assistant (v0.7.17-dev)

Application locale pour chat autour de **BitNet-b1.58-2B-4T**, avec RAG documents, souvenirs, résumés, export local et moteur **llama.cpp**.

## Composants indispensables
- `BitNet-runtime\` (pour BitNet)  
  - `bin\llama-completion.exe`
  - `models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf`
- `BitNet-Windows-Assistant-v0.7.17-dev\dist\BitNetWindowsAssistant\BitNetWindowsAssistant.exe` (l'application)

## Vérification moteur (une fois)
1. Ouvrir `Moteur et modèles...`
2. Vérifier :
   - Exécutable : `C:\AI\bitnet\BitNet-runtime\bin\llama-completion.exe`
   - Modèle : `C:\AI\bitnet\BitNet-runtime\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf`
3. Paramètres conseillés :
   - Threads CPU : `8`
   - Contexte : `2048`
   - Tokens max : `48` à `128`
   - Température : `0.1`
   - TOKENIZER forcé : `llama-bpe` (coché)
4. Cliquer **Enregistrer et tester** → doit s'afficher **OK**.

## Réglages locaux conseillés
- Les données et l’historique sont locaux.
- `RAG` reste volontaire (pas permanent).
- `Index moteur` : **Lexical (défaut)** pour la plupart des PC.
- **HNSW (expérimental)** : option visible uniquement si la machine dispose de mémoire suffisante (>= 8 Go RAM conseillée). Repli automatique sur lexical si erreur.

## Aide et docs
- Utiliser `Aide (?)` dans l’application pour rappel.
- Aucun cloud IA obligatoire par défaut.
- Documents indexés localement sans transfert de données.

## Installation rapide et sans problème
1. **System expectation**  
   Windows 10/11 64-bit, machine x86_64, **RAM conseillée ≥8 Go** (lexical fonctionne plus doucement avec 4 Go).
2. **Composants indispensables**  
   `BitNet-runtime\bin\llama-completion.exe`  
   `BitNet-runtime\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf`  
   `BitNet-Windows-Assistant-v0.7.17-dev\dist\BitNetWindowsAssistant\BitNetWindowsAssistant.exe`
3. **Vérification une fois**
   - *Moteur et modèles...*
   - importer les chemins ci-dessus
   - Threads 8, contexte 2048, tokens max 48/128, température 0.1
   - tokeniazer forcé (llama-bpe)
   - **Enregistrer et tester** → OK.
4. **Usage**
   - aucune API cloud obligatoire
   - no Python à installer
   - index **Lexical (default)**
   - index **HNSW (optional, >=8GB)** avec fallback lexical
   - tested current smoke : chat + RAG + HNSW

## Limites observées (honnêteté)
- Dans un cas réel, dire `merci bitnet` a conduit le modèle à une réponse hors sujet.
- Pour les faits critiques, utiliser RAG documentaire + extraction fidele.

## Attribution
BitNet Windows Assistant — contributions Beam & ChatGPT selon les discussions de projet (open source spirit). 
