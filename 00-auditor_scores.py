import sqlite3
import pandas as pd

DB_PATH = "/lab/visualdata-ia/db/registry.db"

def auditar_scores():
    conn = sqlite3.connect(DB_PATH)
    
    # Ajustes visuales para que la tabla se vea bien en consola
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 20)

    print("📊 AUDITORÍA DE SCORES Y VALIDACIÓN")
    print("=" * 80)

    # 1. MUESTRA DE ACEPTADAS (is_valid = 1)
    print("\n✅ EJEMPLOS DE IMÁGENES ACEPTADAS:")
    df_ok = pd.read_sql_query("""
        SELECT url_hash, is_valid, confidence, 
               score_category, score_product, 
               score_watermark, score_quality, 
               image_suggest_category as 'CLIP_TEXT'
        FROM downloads 
        WHERE is_valid = 1 
        ORDER BY RANDOM() 
        LIMIT 5
    """, conn)
    
    if not df_ok.empty:
        # Formatear hash para que no ocupe tanto
        df_ok['url_hash'] = df_ok['url_hash'].apply(lambda x: x[:8] + '...')
        print(df_ok.to_string(index=False))
    else:
        print("⚠️ No hay imágenes aceptadas aún.")

    # 2. MUESTRA DE RECHAZADAS (is_valid = 0)
    print("\n❌ EJEMPLOS DE IMÁGENES RECHAZADAS:")
    df_ko = pd.read_sql_query("""
        SELECT url_hash, is_valid, confidence, 
               score_category, score_product, 
               score_watermark, score_placeholder, 
               score_quality
        FROM downloads 
        WHERE is_valid = 0 
        ORDER BY RANDOM() 
        LIMIT 5
    """, conn)
    
    if not df_ko.empty:
        df_ko['url_hash'] = df_ko['url_hash'].apply(lambda x: x[:8] + '...')
        print(df_ko.to_string(index=False))
    else:
        print("🎉 No hay imágenes rechazadas (o no se han procesado malas aún).")

    # 3. VERIFICACIÓN DE INTEGRIDAD
    print("\n🔎 ESTADÍSTICAS GLOBALES:")
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(score_product), AVG(confidence) FROM downloads WHERE is_valid=1")
    avgs = cursor.fetchone()
    print(f"   - Promedio 'score_product' en válidas: {avgs[0]:.4f}")
    print(f"   - Promedio 'confidence' en válidas:    {avgs[1]:.2f}")

    conn.close()

if __name__ == "__main__":
    auditar_scores()
