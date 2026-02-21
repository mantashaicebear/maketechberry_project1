"""
Image Post Verification Module
Uses pretrained models in zero-shot mode:
- CLIP for semantic image-domain matching
- YOLO for object detection
- PaddleOCR for text extraction

NO custom datasets, NO training, NO fine-tuning.
"""

import os
import logging
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageVerifier:
    """
    Zero-shot image verification using pretrained models only.
    """
    
    def __init__(self, confidence_threshold: float = 0.70):
        """
        Initialize pretrained models.
        
        Args:
            confidence_threshold: Minimum similarity score to allow post (default: 0.70)
        """
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if self._check_cuda() else "cpu"
        
        logger.info(f"Initializing ImageVerifier on device: {self.device}")
        
        # Load models lazily on first use
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.yolo_model = None
        self.ocr = None
        
        logger.info("ImageVerifier initialization complete (lazy loading enabled)")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _ensure_clip_loaded(self):
        """Load CLIP model if not already loaded"""
        if self.clip_model is None:
            try:
                import torch
                import open_clip
                
                logger.info("Loading CLIP model...")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='laion2b_s34b_b79k'
                )
                self.clip_model = self.clip_model.to(self.device)
                self.clip_model.eval()
                self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
                logger.info("✓ CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                raise
    
    def _ensure_yolo_loaded(self):
        """Load YOLO model if not already loaded"""
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO
                
                logger.info("Loading YOLO model...")
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("✓ YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                raise
    
    def _ensure_ocr_loaded(self):
        """Load PaddleOCR model if not already loaded"""
        if self.ocr is None:
            try:
                from paddleocr import PaddleOCR
                
                logger.info("Loading PaddleOCR...")
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=(self.device == "cuda")
                )
                logger.info("✓ PaddleOCR loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                raise
    
    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract image embedding using CLIP.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Normalized image embedding vector
        """
        try:
            self._ensure_clip_loaded()
            import torch
            
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
        
        except Exception as e:
            logger.error(f"Error extracting image embedding: {e}")
            raise
    
    def compute_domain_similarity(
        self,
        image_path: str,
        allowed_domains: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Compute semantic similarity between image and allowed domains using CLIP.
        
        Args:
            image_path: Path to the image
            allowed_domains: List of allowed domain strings
        
        Returns:
            Tuple of (best_matching_domain, best_score, all_scores_dict)
        """
        try:
            self._ensure_clip_loaded()
            import torch
            
            # Get image embedding
            image_features = self.extract_image_embedding(image_path)
            image_features_tensor = torch.from_numpy(image_features).unsqueeze(0).to(self.device)
            
            # Create text prompts for each domain
            # Use descriptive prompts to improve matching
            domain_prompts = [
                f"a photo of {domain}" for domain in allowed_domains
            ]
            
            # Encode text prompts
            text_inputs = self.clip_tokenizer(domain_prompts).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features_tensor @ text_features.T).cpu().numpy()[0]
            
            # Create scores dictionary
            scores = {domain: float(score) for domain, score in zip(allowed_domains, similarity)}
            
            # Get best match
            best_idx = np.argmax(similarity)
            best_domain = allowed_domains[best_idx]
            best_score = float(similarity[best_idx])
            
            logger.info(f"CLIP scores: {scores}")
            logger.info(f"Best match: {best_domain} ({best_score:.4f})")
            
            return best_domain, best_score, scores
        
        except Exception as e:
            logger.error(f"Error computing domain similarity: {e}")
            raise
    
    def detect_objects(self, image_path: str) -> List[str]:
        """
        Detect objects in the image using pretrained YOLO.
        
        Args:
            image_path: Path to the image
        
        Returns:
            List of detected object labels
        """
        try:
            self._ensure_yolo_loaded()
            
            results = self.yolo_model(image_path, verbose=False)
            
            detected_objects = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = self.yolo_model.names[class_id]
                        
                        # Only include high-confidence detections
                        if confidence >= 0.25:
                            detected_objects.append(label)
            
            # Remove duplicates while preserving order
            detected_objects = list(dict.fromkeys(detected_objects))
            
            logger.info(f"Detected objects: {detected_objects}")
            return detected_objects
        
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def extract_ocr_text(self, image_path: str) -> str:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image_path: Path to the image
        
        Returns:
            Cleaned and normalized extracted text
        """
        try:
            self._ensure_ocr_loaded()
            
            result = self.ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                logger.info("No text detected in image")
                return ""
            
            # Extract and clean text
            texts = []
            for line in result[0]:
                if line[1][1] >= 0.5:  # Confidence threshold
                    texts.append(line[1][0])
            
            extracted_text = " ".join(texts).strip()
            logger.info(f"Extracted OCR text: '{extracted_text}'")
            
            return extracted_text
        
        except Exception as e:
            logger.error(f"Error extracting OCR text: {e}")
            return ""
    
    def check_object_domain_match(
        self,
        detected_objects: List[str],
        allowed_domains: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check if any detected objects match the allowed domains.
        Uses keyword matching and semantic similarity.
        
        Args:
            detected_objects: List of detected object labels
            allowed_domains: List of allowed domains
        
        Returns:
            Tuple of (match_found, matched_objects)
        """
        if not detected_objects:
            return False, []
        
        matched_objects = []
        
        # Domain-specific object keywords
        domain_keywords = {
            'food': ['pizza', 'sandwich', 'burger', 'cake', 'bread', 'food', 'fruit', 
                     'vegetable', 'drink', 'coffee', 'wine', 'bottle', 'cup', 'bowl', 
                     'fork', 'knife', 'spoon', 'dining', 'restaurant'],
            'tech': ['laptop', 'computer', 'phone', 'keyboard', 'mouse', 'monitor',
                     'tablet', 'tv', 'remote', 'cell phone', 'electronics'],
            'electronics': ['laptop', 'computer', 'phone', 'keyboard', 'mouse', 'monitor',
                           'tablet', 'tv', 'remote', 'cell phone', 'camera'],
            'education': ['book', 'laptop', 'computer', 'backpack', 'pencil', 'notebook',
                         'desk', 'chair', 'person', 'classroom'],
            'health': ['person', 'bicycle', 'sports', 'yoga', 'exercise', 'medical'],
            'finance': ['laptop', 'computer', 'desk', 'chair', 'person', 'office'],
            'fashion': ['person', 'handbag', 'tie', 'suitcase', 'umbrella', 'backpack',
                       'shoes', 'clothing', 'dress', 'shirt'],
            'beauty': ['person', 'bottle', 'cosmetics', 'mirror', 'comb', 'brush'],
            'automotive': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
                          'traffic light', 'stop sign'],
            'real_estate': ['house', 'building', 'door', 'window', 'furniture', 'couch',
                           'bed', 'dining table', 'chair'],
            'entertainment': ['tv', 'remote', 'book', 'sports', 'kite', 'skateboard',
                             'surfboard', 'tennis'],
            'books': ['book', 'laptop', 'desk', 'chair', 'person'],
            'travel': ['airplane', 'train', 'bus', 'car', 'suitcase', 'backpack',
                      'boat', 'bicycle'],
            'grocery': ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange',
                       'broccoli', 'carrot', 'pizza', 'cake'],
            'sports': ['sports ball', 'tennis racket', 'baseball bat', 'skateboard',
                      'surfboard', 'skis', 'snowboard', 'kite', 'bicycle', 'person']
        }
        
        # Check for keyword matches
        for domain in allowed_domains:
            domain_lower = domain.lower()
            if domain_lower in domain_keywords:
                keywords = domain_keywords[domain_lower]
                for obj in detected_objects:
                    obj_lower = obj.lower()
                    if any(keyword in obj_lower or obj_lower in keyword for keyword in keywords):
                        matched_objects.append(obj)
        
        # Remove duplicates
        matched_objects = list(set(matched_objects))
        
        match_found = len(matched_objects) > 0
        logger.info(f"Object-domain match: {match_found}, matched: {matched_objects}")
        
        return match_found, matched_objects
    
    def check_ocr_domain_match(
        self,
        ocr_text: str,
        allowed_domains: List[str]
    ) -> Tuple[bool, str]:
        """
        Check if OCR text matches any allowed domain using keyword matching.
        
        Args:
            ocr_text: Extracted text from image
            allowed_domains: List of allowed domains
        
        Returns:
            Tuple of (match_found, reason)
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            # If no meaningful text, consider it a pass (text is optional)
            return True, "No text detected (acceptable)"
        
        ocr_lower = ocr_text.lower()
        
        # Domain-specific keywords for text matching
        text_keywords = {
            'food': ['food', 'restaurant', 'menu', 'cuisine', 'dish', 'recipe',
                    'eat', 'delicious', 'tasty', 'meal', 'dinner', 'lunch'],
            'tech': ['technology', 'software', 'hardware', 'app', 'programming',
                    'code', 'developer', 'tech', 'digital', 'innovation'],
            'electronics': ['electronics', 'gadget', 'device', 'smartphone', 'laptop',
                           'computer', 'tablet', 'TV', 'camera'],
            'education': ['education', 'learn', 'study', 'course', 'class', 'training',
                         'tutorial', 'school', 'university', 'teaching', 'knowledge'],
            'health': ['health', 'fitness', 'wellness', 'medical', 'doctor', 'exercise',
                      'yoga', 'nutrition', 'healthy'],
            'finance': ['finance', 'money', 'investment', 'banking', 'loan', 'credit',
                       'savings', 'financial', 'economy'],
            'fashion': ['fashion', 'style', 'clothing', 'apparel', 'outfit', 'trend',
                       'designer', 'wear', 'collection'],
            'beauty': ['beauty', 'makeup', 'cosmetics', 'skincare', 'spa', 'salon',
                      'hair', 'nail', 'facial'],
            'automotive': ['car', 'auto', 'vehicle', 'automotive', 'drive', 'engine',
                          'motor', 'transport'],
            'real_estate': ['property', 'real estate', 'house', 'apartment', 'rent',
                           'sale', 'home', 'building'],
            'entertainment': ['entertainment', 'movie', 'music', 'game', 'show', 'fun',
                             'concert', 'theatre'],
            'books': ['book', 'read', 'novel', 'author', 'story', 'literature', 'publish'],
            'travel': ['travel', 'trip', 'vacation', 'tour', 'destination', 'journey',
                      'adventure', 'explore'],
            'grocery': ['grocery', 'supermarket', 'fresh', 'organic', 'produce', 'market'],
            'sports': ['sport', 'game', 'match', 'player', 'team', 'competition',
                      'athletic', 'championship']
        }
        
        # Check for matches
        for domain in allowed_domains:
            domain_lower = domain.lower()
            
            # Direct domain name match
            if domain_lower in ocr_lower:
                return True, f"Text contains domain keyword '{domain}'"
            
            # Check keyword matches
            if domain_lower in text_keywords:
                keywords = text_keywords[domain_lower]
                for keyword in keywords:
                    if keyword in ocr_lower:
                        return True, f"Text contains relevant keyword '{keyword}'"
        
        # No match found
        return False, f"Text does not match allowed domains: {', '.join(allowed_domains)}"
    
    def decision_engine(
        self,
        image_path: str,
        allowed_domains: List[str],
        business_id: str = None
    ) -> Dict:
        """
        Main decision engine with strict rules.
        
        ALLOW post ONLY IF:
        - CLIP similarity score ≥ confidence_threshold
        AND
        - At least one detected object matches the allowed domain
        AND
        - OCR text (if present) matches the allowed domain
        
        Args:
            image_path: Path to uploaded image
            allowed_domains: List of allowed domains for the business
            business_id: Optional business ID
        
        Returns:
            Decision dictionary with status, reason, and analysis details
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            logger.info(f"Allowed domains: {allowed_domains}")
            
            # Step 1: CLIP semantic similarity
            matched_domain, clip_score, all_scores = self.compute_domain_similarity(
                image_path, allowed_domains
            )
            
            # Step 2: Object detection
            detected_objects = self.detect_objects(image_path)
            
            # Step 3: OCR text extraction
            ocr_text = self.extract_ocr_text(image_path)
            
            # Step 4: Check object-domain match
            object_match, matched_objects = self.check_object_domain_match(
                detected_objects, allowed_domains
            )
            
            # Step 5: Check OCR-domain match
            ocr_match, ocr_reason = self.check_ocr_domain_match(
                ocr_text, allowed_domains
            )
            
            # Step 6: Apply strict decision rules
            clip_pass = clip_score >= self.confidence_threshold
            
            # Build decision
            if not clip_pass:
                status = "BLOCK"
                reason = (
                    f"CLIP similarity score ({clip_score:.4f}) is below threshold "
                    f"({self.confidence_threshold}). Image does not strongly match "
                    f"allowed domains: {', '.join(allowed_domains)}."
                )
            elif not object_match:
                status = "BLOCK"
                reason = (
                    f"No detected objects match the allowed domains. "
                    f"Detected: {', '.join(detected_objects) if detected_objects else 'nothing'}. "
                    f"Allowed domains: {', '.join(allowed_domains)}."
                )
            elif not ocr_match:
                status = "BLOCK"
                reason = f"OCR text mismatch: {ocr_reason}"
            else:
                status = "ALLOW"
                reason = (
                    f"Image approved: CLIP match with '{matched_domain}' "
                    f"(score: {clip_score:.4f}), objects detected: {', '.join(matched_objects)}, "
                    f"OCR validation passed."
                )
            
            result = {
                "status": status,
                "reason": reason,
                "matched_domain": matched_domain,
                "clip_score": round(clip_score, 4),
                "clip_threshold": self.confidence_threshold,
                "all_domain_scores": {k: round(v, 4) for k, v in all_scores.items()},
                "detected_objects": detected_objects,
                "matched_objects": matched_objects if object_match else [],
                "ocr_text": ocr_text,
                "ocr_match": ocr_match,
                "checks_passed": {
                    "clip_similarity": clip_pass,
                    "object_detection": object_match,
                    "ocr_validation": ocr_match
                }
            }
            
            logger.info(f"Decision: {status}")
            return result
        
        except Exception as e:
            logger.error(f"Error in decision engine: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "reason": f"Error analyzing image: {str(e)}",
                "matched_domain": "",
                "clip_score": 0.0,
                "detected_objects": [],
                "ocr_text": "",
                "error": str(e)
            }


# Singleton instance
_verifier_instance = None


def get_image_verifier(confidence_threshold: float = 0.70) -> ImageVerifier:
    """
    Get or create singleton ImageVerifier instance.
    
    Args:
        confidence_threshold: Minimum CLIP score to allow post
    
    Returns:
        ImageVerifier instance
    """
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = ImageVerifier(confidence_threshold)
    return _verifier_instance


def verify_image_post(
    image_path: str,
    allowed_domains: List[str],
    confidence_threshold: float = 0.70,
    business_id: str = None
) -> Dict:
    """
    Main function to verify image post.
    
    Args:
        image_path: Path to the uploaded image
        allowed_domains: List of allowed domains for the business
        confidence_threshold: Minimum CLIP similarity score (default: 0.70)
        business_id: Optional business ID
    
    Returns:
        Verification result dictionary
    """
    verifier = get_image_verifier(confidence_threshold)
    return verifier.decision_engine(image_path, allowed_domains, business_id)
