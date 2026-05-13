import os
from dotenv import load_dotenv

load_dotenv()

_supabase_client = None

def get_supabase():
    """Lazy init — solo conecta cuando se necesita, no al importar."""
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Faltan variables de entorno SUPABASE_URL o SUPABASE_KEY")
        _supabase_client = create_client(url, key)
    return _supabase_client

# Compatibilidad con código que importa `supabase` directamente
class _LazySupabase:
    def __getattr__(self, name):
        return getattr(get_supabase(), name)

supabase = _LazySupabase()
