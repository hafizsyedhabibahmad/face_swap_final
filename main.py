import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from retry import retry
from PIL import Image, ImageEnhance
import uuid
import hashlib
from cachetools import TTLCache
import tempfile

app = FastAPI()

# Ensure the static folder exists
os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/output", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cache with TTL of 1 hour
cache = TTLCache(maxsize=100, ttl=3600)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file_path: str) -> bool:
    return os.path.exists(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def compress_image(content: bytes, max_size: int = 1024) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            img = Image.open(tmp.name)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out_tmp:
                img.save(out_tmp.name, "PNG", optimize=True, quality=85)
                with open(out_tmp.name, "rb") as f:
                    return f.read()
    except Exception:
        return content

def enhance_image(image_path: str) -> None:
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Sharpness(img)
        img_enhanced = enhancer.enhance(2.0)
        img_enhanced.save(image_path, "PNG")
    except Exception:
        pass

def save_output_image(result_path: str, output_dir: str, output_name: str) -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        img = Image.open(result_path).convert("RGB")
        img.save(output_path, "PNG")
        enhance_image(output_path)
        return output_path
    except Exception:
        return ""

@retry(tries=3, delay=2, backoff=2)
async def face_swap(source_image: str, dest_image: str, source_face_idx: int = 1, dest_face_idx: int = 1) -> str:
    try:
        if not all([validate_file(source_image), validate_file(dest_image)]):
            return "Invalid input files"

        client = Client("Dentro/face-swap")
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=source_face_idx,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if result and os.path.exists(result):
            unique_filename = f"face_swap_{uuid.uuid4().hex}.png"
            final_path = save_output_image(result, OUTPUT_FOLDER, unique_filename)
            return final_path if final_path else "Failed to save output"
        return "Face swap failed"
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_image": None})

@app.post("/swap")
async def swap_faces(source_image: UploadFile = File(...), dest_image: UploadFile = File(...)):
    if not source_image.filename or not dest_image.filename:
        return JSONResponse(status_code=400, content={"error": "No file selected"})

    if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
        return JSONResponse(status_code=400, content={"error": "Invalid file format. Only PNG, JPG, JPEG allowed"})

    source_content = await source_image.read()
    dest_content = await dest_image.read()
    source_content = compress_image(source_content)
    dest_content = compress_image(dest_content)

    cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"
    if cache_key in cache:
        return {"result_image": f"/{cache[cache_key]}"}

    with tempfile.TemporaryDirectory() as temp_dir:
        source_ext = source_image.filename.rsplit('.', 1)[1]
        dest_ext = dest_image.filename.rsplit('.', 1)[1]
        source_path = os.path.join(temp_dir, f"source.{source_ext}")
        dest_path = os.path.join(temp_dir, f"dest.{dest_ext}")

        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(dest_path, "wb") as f:
            f.write(dest_content)

        result = await face_swap(source_path, dest_path)
        if result.startswith("Error") or result in ["Invalid input files", "Failed to save output"]:
            return JSONResponse(status_code=500, content={"error": result})

        # Save and cache the result image
        cache[cache_key] = result
        return {"result_image": f"/static/output/{os.path.basename(result)}"}
