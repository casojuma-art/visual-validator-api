import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import os
import re
import html
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends, File, UploadFile, Form
from fastapi.security.api_key import APIKeyHeader

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SeeStocks Visual Validator & Suggester V2")

# --- RUTAS Y CONFIGURACIÓN ---
KEYS_PATH = "/app/api_keys.txt"
CSV_PATH = "gpc_id_to_path.csv" 
MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = "cpu"
# --- CONFIGURACIÓN DE UMBRAL (EL "FILTRO DE IGNORANCIA") ---
# 0.22 es un valor conservador para CLIP ViT-B/32. 
# Si la similitud es menor, es ruido o confusión geométrica.
TAXONOMY_THRESHOLD = 0.28
VALIDATION_MATCH_THRESHOLD = 0.22 # Subimos de 0.15 a 0.22 (exigimos más parecido al producto)
MAX_BAD_SCORE_ALLOWED = 0.60     # Bajamos de 0.70 a 0.50 (toleramos menos logos/basura)

# --- VARIABLES GLOBALES ---
model = None
processor = None
taxonomy_names = []
taxonomy_embeddings = None

# --- SEGURIDAD: API KEY ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if not os.path.exists(KEYS_PATH):
        # Modo desarrollo si no hay fichero de claves (opcional, cuidado en prod)
        valid_keys = {"seestocks_secret_key_wwRT"} 
    else:
        with open(KEYS_PATH, "r") as f:
            valid_keys = {line.strip() for line in f if line.strip()}

    if api_key_header in valid_keys:
        return api_key_header
    else:
        raise HTTPException(status_code=401, detail="Credenciales inválidas (X-API-Key required)")

def sanitize_input(text: str) -> str:
    if not text: return ""
    if len(text) > 500: text = text[:500]
    text = text.replace("\0", "").replace("\\", "")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    return html.escape(text).strip()

# --- CARGA DE MODELO Y TAXONOMÍA ---
def load_assets():
    global model, processor, taxonomy_names, taxonomy_embeddings
    try:
        logger.info(f">>> Cargando CLIP en {DEVICE}...")
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        if os.path.exists(CSV_PATH):
            logger.info(f">>> Cargando taxonomía desde {CSV_PATH}...")
            df = pd.read_csv(CSV_PATH)
            # Asumimos que la columna 1 tiene la ruta de categoría
            taxonomy_names = df.iloc[:, 1].astype(str).tolist()
            
            all_embeddings = []
            batch_size = 128
            with torch.no_grad():
                for i in range(0, len(taxonomy_names), batch_size):
                    batch = taxonomy_names[i:i+batch_size]
                    # Prompt engineering básico para mejorar la semántica
                    clean_batch = [f"a photo of {t.replace(' > ', ' ')}" for t in batch]
                    inputs = processor(text=clean_batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    text_features = model.get_text_features(**inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    all_embeddings.append(text_features)
            
            taxonomy_embeddings = torch.cat(all_embeddings)
            logger.info(f">>> {len(taxonomy_names)} categorías indexadas. Umbral activo: {TAXONOMY_THRESHOLD}")
        else:
            logger.error(f"!!! No se encontró {CSV_PATH}. La sugerencia no funcionará.")

    except Exception as e:
        logger.error(f"!!! Error en carga inicial: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_assets()

# --- ENDPOINT PRINCIPAL ---

@app.post("/verify")
async def verify(
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(...),
    token: str = Depends(get_api_key)
):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está listo.")

    try:
        safe_title = sanitize_input(title)
        safe_category = sanitize_input(category)

        # 1. Gestionar imagen y transparencias
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

        # 2. Inferencia para Validación (Quality Check)
        labels = [
            f"a photo of {safe_category}", 
            f"a photo of {safe_title}",
            "a photo with watermarks, logos or text on top",
            "a generic placeholder image or 'image not found' sign",
            "a blurry or low quality photo"
        ]

        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Extraer características visuales para la sugerencia posterior
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        # 3. Lógica de validación (¿Es la imagen válida?)
        score_match = max(probs[0], probs[1]) 
        score_bad = max(probs[2], probs[3], probs[4])
        #is_valid = score_match > 0.15 and score_bad < 0.7
        is_valid = score_match > VALIDATION_MATCH_THRESHOLD and score_bad < MAX_BAD_SCORE_ALLOWED

        # 4. Lógica de sugerencia (Taxonomía Google) CON UMBRAL V2
        suggested_cat = None
        suggested_confidence = 0.0

        if taxonomy_embeddings is not None:
            # Producto escalar (Similitud Coseno)
            similarities = (image_features @ taxonomy_embeddings.T).squeeze(0)
            
            # Obtenemos el mejor valor y su índice
            best_val, best_idx = similarities.topk(1)
            best_score = best_val.item()
            suggested_confidence = round(best_score, 4)

            # --- FILTRO DE ALUCINACIONES ---
            if best_score >= TAXONOMY_THRESHOLD:
                suggested_cat = taxonomy_names[best_idx.item()]
            else:
                # Si no supera el umbral, devolvemos None.
                # La IA "admite" que no sabe qué es.
                suggested_cat = None 

        return {
            "is_valid": is_valid,
            "confidence": round(score_match * 100, 2), # Confianza de la validación (¿Es buena foto?)
            "image_suggest_category": suggested_cat,   # Puede ser null
            "image_suggest_confidence": suggested_confidence, # Dato técnico para debugging
            "detections": {
                "category_match": round(probs[0], 4),
                "product_match": round(probs[1], 4),
                "watermark_text": round(probs[2], 4),
                "placeholder_or_error": round(probs[3], 4),
                "low_quality": round(probs[4], 4)
            },
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error procesando: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno analizando la imagen")

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "clip-visual-with-taxonomy-v2"}
