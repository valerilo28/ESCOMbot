from supabase import create_client

SUPABASE_URL = "https://fgmimudjthrxkqcitejp.supabase.co"
SUPABASE_KEY = "sb_publishable_IzERztqBdnkvWCEnI7HzsQ_A9Vohu-N"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)