from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from app.models import AnalysisRequest, AnalysisResponse, ImageAnalysisRequest, ImageAnalysisResponse
from app.trained_model_analyzer import analyze_content
import uvicorn
import os
import json
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

app = FastAPI(
    title="Venture Content Guard",
    description="A high-precision content moderation API for text and image posts.",
    version="2.0.0"
)

@app.post("/analyze", response_model=AnalysisResponse, status_code=200)
async def analyze_post(request: AnalysisRequest):
    """
    Analyze incoming text posts for domain alignment and professionalism.
    """
    try:
        result = analyze_content(request.user_text, request.registered_domain, request.business_id)
        return result
        
    except ValueError as e:
        # Configuration error (e.g. missing API key)
        raise HTTPException(status_code=500, detail=str(e))
        
    except RuntimeError as e:
        # External service error (Gemini API failed, Quota exceeded, etc.)
        # 503 Service Unavailable is appropriate for upstream failures
        raise HTTPException(status_code=503, detail=str(e))
        
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/analyze_image", response_model=ImageAnalysisResponse, status_code=200)
async def analyze_image_post(
    image: UploadFile = File(..., description="Image file to analyze"),
    business_id: str = Form(..., description="Business ID"),
    confidence_threshold: float = Form(0.70, description="CLIP score threshold")
):
    """
    Analyze uploaded image post for domain alignment using:
    - CLIP for semantic similarity
    - YOLO for object detection
    - PaddleOCR for text extraction
    
    Zero-shot pretrained models only, no training/fine-tuning.
    """
    temp_path = None
    try:
        # Import business profiles helper
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from data.business_profiles_helper import get_business_profile
        
        # Get business profile
        business_profile = get_business_profile(business_id)
        if not business_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Business ID '{business_id}' not found"
            )
        
        allowed_domains = business_profile.get('domains', [])
        if not allowed_domains:
            raise HTTPException(
                status_code=400,
                detail=f"Business '{business_id}' has no allowed domains configured"
            )
        
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"upload_{business_id}_{image.filename}")
        
        with open(temp_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Analyze image
        from app.image_analyzer import verify_image_post
        
        result = verify_image_post(
            image_path=temp_path,
            allowed_domains=allowed_domains,
            confidence_threshold=confidence_threshold,
            business_id=business_id
        )
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise
        
    except Exception as e:
        # Clean up and handle errors
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Venture Content Guard",
        "version": "2.0.0",
        "endpoints": {
            "text_analysis": "/analyze",
            "image_analysis": "/analyze_image",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
