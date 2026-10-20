from pathlib import Path

stream = b"BT /F1 12 Tf 72 720 Td (Bonjour! Ce PDF de test contient la phrase cle: trois pommes et deux oranges.) Tj ET\n"

objects = [
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n",
    b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    b"5 0 obj\n<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream\nendobj\n",
]

positions = []
offset = 8
buf = b"%PDF-1.4\n"
for o in objects:
    positions.append(offset)
    buf += o
    offset += len(o)

xref = b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n0000000000 65535 f \n"
for pos in positions:
    xref += f"{pos:010d} 00000 n \n".encode("ascii")
buf += xref
buf += b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii") + b" /Root 1 0 R >>\n"
buf += b"startxref\n" + str(offset).encode("ascii") + b"\n%%EOF\n"

out = Path(r"C:\AI\bitnet\BitNet-Windows-Assistant-v0.7.18-dev\rag-corpus\test_simple.pdf")
out.write_bytes(buf)
print("Created", out)