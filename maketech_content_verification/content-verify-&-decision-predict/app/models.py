from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Text Analysis Models
class AnalysisRequest(BaseModel):
    user_text: str = Field(..., alias="User_Text", description="The content the user wants to post.")
    registered_domain: str = Field(..., alias="Registered_Domain", description="The verified industry category of the startup.")
    business_id: Optional[str] = Field(None, alias="Business_ID", description="Optional business ID for domain-specific enforcement.")

    class Config:
        populate_by_name = True

class AnalysisResponse(BaseModel):
    status: str = Field(..., description="Approved / Rejected / Flagged for Manual Review")
    reason: str = Field(..., description="Brief explanation of why it was blocked or allowed")
    confidence_score: float = Field(..., description="Confidence score between 0.0 and 1.0")
    detected_category: str = Field(..., description="The industry sector detected in the text")


# Image Analysis Models
class ImageAnalysisRequest(BaseModel):
    business_id: str = Field(..., alias="business_id", description="Business ID to check allowed domains")
    confidence_threshold: Optional[float] = Field(0.70, alias="confidence_threshold", description="Minimum CLIP score threshold (0-1)")

    class Config:
        populate_by_name = True


class ImageAnalysisResponse(BaseModel):
    status: str = Field(..., description="ALLOW or BLOCK")
    reason: str = Field(..., description="Explanation of the decision")
    matched_domain: str = Field(..., description="Best matching domain from CLIP")
    clip_score: float = Field(..., description="CLIP similarity score (0-1)")
    clip_threshold: float = Field(..., description="Threshold used for decision")
    all_domain_scores: Dict[str, float] = Field(..., description="Scores for all allowed domains")
    detected_objects: List[str] = Field(..., description="Objects detected by YOLO")
    matched_objects: List[str] = Field(..., description="Objects that matched allowed domains")
    ocr_text: str = Field(..., description="Text extracted from image")
    ocr_match: bool = Field(..., description="Whether OCR text matches allowed domains")
    checks_passed: Dict[str, bool] = Field(..., description="Results of each verification check")
