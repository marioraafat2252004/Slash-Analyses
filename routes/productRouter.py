from fastapi.responses import JSONResponse
import json
from fastapi import APIRouter, File, UploadFile
from controllers.productController import analyze_image_controller

router = APIRouter()

@router.post("/analyse-image")
async def analyze_image(file: UploadFile = File(...)):  # Field name must match
    print(f"Received image file: {file.filename}")
    try:
        # Process the file
        analysis = await analyze_image_controller(file)
        return JSONResponse(content=json.loads(analysis))
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


