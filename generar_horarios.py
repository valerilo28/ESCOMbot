"""
Genera horarios.json desde el Excel horarios_v5.xlsx
Uso: python generar_horarios.py
Coloca este script en la raíz del proyecto (junto a rebuild_index.py)
"""
import openpyxl
import json
from pathlib import Path
from collections import defaultdict

EXCEL_PATH = Path(__file__).resolve().parent / "horarios_v5.xlsx"
OUTPUT_PATH = Path(__file__).resolve().parent / "app" / "rag" / "horarios.json"

def fmt_hora(t):
    if t is None:
        return "?"
    if hasattr(t, 'strftime'):
        return t.strftime("%H:%M")
    return str(t).strip()

def generar():
    if not EXCEL_PATH.exists():
        print(f"❌ No se encontró el Excel en: {EXCEL_PATH}")
        print("   Asegúrate de que horarios_v5.xlsx esté en la raíz del proyecto.")
        return

    wb = openpyxl.load_workbook(str(EXCEL_PATH), read_only=True)
    ws = wb['Horarios']

    profesores = defaultdict(list)
    total = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        profesor, dia, entrada, salida, materia, salon = row
        profesores[profesor.strip()].append({
            "dia": str(dia).strip() if dia else "?",
            "entrada": fmt_hora(entrada),
            "salida": fmt_hora(salida),
            "materia": str(materia).strip() if materia else "?",
            "salon": str(salon).strip() if salon else "?"
        })
        total += 1

    result = {"profesores": dict(profesores)}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ horarios.json generado")
    print(f"   Profesores: {len(profesores)}")
    print(f"   Registros:  {total}")
    print(f"   Guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    generar()