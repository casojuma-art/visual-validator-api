import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import os
import re
import html
import logging
from fastapi import FastAPI, HTTPException, Security, Depends, File, UploadFile, Form
from fastapi.security.api_key import APIKeyHeader

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SeeStocks Visual Validator API")

# --- RUTAS Y CONFIGURACIÓN ---
KEYS_PATH = "/app/api_keys.txt"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- SEGURIDAD: API KEY ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if not os.path.exists(KEYS_PATH):
        logger.warning("Fichero api_keys.txt no encontrado.")
        valid_keys = set()
    else:
        with open(KEYS_PATH, "r") as f:
            valid_keys = {line.strip() for line in f if line.strip()}

    if api_key_header in valid_keys:
        return api_key_header
    else:
        raise HTTPException(
            status_code=401, 
            detail="Credenciales inválidas (X-API-Key required)"
        )

# --- SEGURIDAD: SANITIZACIÓN ---
def sanitize_input(text: str) -> str:
    if not text: return ""
    if len(text) > 500: text = text[:500]
    text = text.replace("\0", "").replace("\\", "")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    return html.escape(text).strip()

# --- VARIABLES GLOBALES PARA LA IA ---
model = None
processor = None

def load_visual_model():
    global model, processor
    try:
        logger.info(f">>> Cargando CLIP en {DEVICE}...")
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        logger.info(">>> MODELO VISUAL LISTO.")
    except Exception as e:
        logger.error(f"!!! Error cargando el modelo: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_visual_model()

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

        # 1. Leer imagen y gestionar transparencias (FONDO BLANCO)
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGBA")
            # Creamos un fondo blanco sólido
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            # Combinamos la imagen sobre el fondo blanco
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

        # 2. Etiquetas de comparación
        labels = [
            f"a photo of {safe_category}", 
            f"a photo of {safe_title}",
            "a photo with watermarks, logos or text on top",
            "a generic placeholder image or 'image not found' sign",
            "a blurry or low quality photo"
        ]

        # 3. Inferencia
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        # 4. Lógica de decisión mejorada
        score_match = max(probs[0], probs[1]) 
        score_bad = max(probs[2], probs[3], probs[4])

        # Bajamos el umbral a 0.35 para ser más flexibles con joyería fina
        is_valid = score_match > 0.15 and score_bad < 0.7

        return {
            "is_valid": is_valid,
            "confidence": round(score_match * 100, 2),
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
    return {"status": "ok", "engine": "clip-visual"}
