import json
import pathlib
import sys

import pymupdf

P2 = pathlib.Path(sys.argv[1])
out = P2 / "lectura"
out.mkdir(exist_ok=True)
for t, f in [("A", "reporte.md"), ("B", "solucion.ipynb"), ("C", "reporte.pdf"),
             ("S1", "reporte.md"), ("I1", "reporte.md"), ("P1", "reporte.md")]:
    p = P2 / "base" / "corridas" / f"tarea-{t}" / f
    if f.endswith(".ipynb"):
        nb = json.loads(p.read_text(encoding="utf-8"))
        partes = []
        for c in nb["cells"]:
            partes.append(f"--- [{c['cell_type']}] ---\n" + "".join(c["source"]))
            if c["cell_type"] == "code":
                for o in c.get("outputs", []):
                    txt = "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
                    if txt:
                        partes.append("[salida]\n" + txt)
        txt = "\n".join(partes)
    elif f.endswith(".pdf"):
        txt = "\n".join(pg.get_text() for pg in pymupdf.open(p))
    else:
        txt = p.read_text(encoding="utf-8")
    (out / f"{t}.txt").write_text(txt, encoding="utf-8")
    print(t, len(txt))