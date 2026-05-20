"""
Genera app/rag/horarios.json desde el Excel de horarios.
Requiere: pip install openpyxl

Uso:
    python generar_horarios.py horarios-v5.xlsx
"""
import json
import sys
from pathlib import Path

def generar(excel_path: str):
    try:
        import openpyxl
    except ImportError:
        print("Instala openpyxl: pip install openpyxl")
        return

    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    profesores = {}
    headers = None

    for row in ws.iter_rows(values_only=True):
        if headers is None:
            # Buscar fila de encabezados
            if row and "Profesor" in str(row):
                headers = [str(c).strip() if c else "" for c in row]
            continue

        if not row or not row[0]:
            continue

        try:
            # Mapear columnas por nombre
            idx = {h: i for i, h in enumerate(headers)}
            profesor = str(row[idx.get("Profesor", 0)] or "").strip().upper()
            dia      = str(row[idx.get("Día", 1)] or "").strip()
            entrada  = str(row[idx.get("Hora Entrada", 2)] or "").strip()
            salida   = str(row[idx.get("Hora Salida", 3)] or "").strip()
            materia  = str(row[idx.get("Materia", 4)] or "").strip()
            salon    = str(row[idx.get("Salón", 5)] or "").strip()

            if not profesor or not dia:
                continue

            if profesor not in profesores:
                profesores[profesor] = []

            profesores[profesor].append({
                "dia": dia,
                "entrada": entrada,
                "salida": salida,
                "materia": materia,
                "salon": salon
            })
        except Exception as e:
            continue

    output = Path("app/rag/horarios.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"profesores": profesores}, f, ensure_ascii=False, indent=2)

    print(f"✅ Generado: {output}")
    print(f"   Profesores: {len(profesores)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python generar_horarios.py horarios-v5.xlsx")
    else:
        generar(sys.argv[1])
