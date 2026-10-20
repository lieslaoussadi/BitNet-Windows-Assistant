# Rapport de test BitNet (local)

## Parametres utilises
- Executable: `C:\AI\bitnet\BitNet-runtime\bin\llama-completion.exe`
- Modele: `C:\AI\bitnet\BitNet-runtime\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf`
- Contexte: `2048`
- Threads: `8`
- Tokens max: `48`
- Temperature: `0.1`

## Resultats
### Cas 1: Bonjour.
- duree: 1.7s
- reponse: Bonjour! Comment puis-je vous aider aujourd'hui?

### Cas 2: Que vaut 2+2 ? Reponds tres court.
- duree: 2.1s
- reponse: La réponse est simple : 4. C'est parce que 2+2 est un nombre.

### Cas 3: Qui est le president inconnu d'une planete lointaine ?
- duree: 3.2s
- reponse: Il n'y a pas de président de la planète lointaine. La planète est appelée Terre et elle est la seule planète dans le système de l'Univers que nous vivons. La planète est orbite

### Cas 4: Reponds 'OUI' sans explication.
- duree: 1.3s
- reponse: OUI.

### Cas 5: Donne une seule phrase exacte pour verifier ton comportement.
- duree: 3.2s
- reponse: Je ne peux pas vous fournir une phrase exacte pour verifier ton comportement, car cela necessite une anthropomorphisation de l'artificial intelligence, qui n'est possible. Cependant, je peux vous fourn

## Conclusion honnete
- Le modele repond court, vite et correctement aux questions simples/factuelles.
- Il peut **halluciner** des details pour des questions speculatives hors contexte.
- **RAG documentaire + extraction fidele** reste le veritable support pour avancer des faits.
- A recommander comme moteur generaliste de consultation, pas comme oracle sans verification.

