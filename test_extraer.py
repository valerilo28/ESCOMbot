from app.rag.loader import load_documents 

try:
    documentos = load_documents()
    for i, doc in enumerate(documentos[:3]):
        print(f"\n--- PÁGINA {i+1} ---")
        print(f"Fuente: {doc.metadata['source']}")
        print(f"Texto extraído:\n{doc.page_content[:500]}...")
except Exception as e:
    print(f"Error: {e}")
