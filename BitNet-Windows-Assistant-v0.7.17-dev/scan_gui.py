import pathlib
p=pathlib.Path(r"C:\AI\bitnet\BitNet-Windows-Assistant-v0.7.3-dev\src\bitnet_windows\chat_gui.py")
s=p.read_text(encoding="utf-8")
tokens=['COLORS["blue_light"]','#EFF6FF','highlightthickness','highlightbackground','bg=COLORS["white"]','bg="#172947','bg=COLORS["blue"]','mode_note_var','mode_box','tk.Frame(self.doc_card']
for token in tokens:
    print(f"{token}: {s.count(token)}")