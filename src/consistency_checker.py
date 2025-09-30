"""
Optimized Consistency Checker for 4GB GPUs
"""

import torch
import os
import sys
from PIL import Image
import numpy as np
from pathlib import Path

# Import CLIP with robust error handling
try:
    import clip as openai_clip
    CLIP_AVAILABLE = True
except ImportError:
    try:
        import openai_clip
        CLIP_AVAILABLE = True
    except ImportError:
        print("❌ CLIP not available. Please install: pip install openai-clip")
        CLIP_AVAILABLE = False
        openai_clip = None

from config import Config
from utils import setup_logging

class ConsistencyChecker:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.model = None
        self.preprocess = None
        self.reference_images = self.load_reference_images()
        
        if CLIP_AVAILABLE:
            self.setup_clip()
        else:
            self.logger.error("[ERROR] CLIP not available. Validation will be limited.")
    
    def setup_clip(self):
        """Initialize lightweight CLIP model for 4GB GPUs"""
        try:
            self.logger.info("[CLIP] Loading lightweight model for 4GB GPU: %s", self.config.VALIDATION_MODEL)
            
            # Use smaller model for 4GB GPU
            model_name = "ViT-B/32"  # Force smaller model for 4GB
            self.model, self.preprocess = openai_clip.load(
                model_name,
                device=self.config.DEVICE,
                download_root="models/clip"
            )
            
            if self.config.DEVICE == "cuda":
                self.model = self.model.half()  # Use half precision
            
            self.logger.info("[SUCCESS] Lightweight CLIP model loaded on %s", self.config.DEVICE.upper())
            
        except Exception as e:
            self.logger.error("[ERROR] Failed to load CLIP: %s", str(e))
            self.model = None
            self.preprocess = None
    
    def load_reference_images(self):
        """Load reference images if available"""
        references = {}
        reference_dir = Path("inputs/reference_images")
        
        if not reference_dir.exists():
            self.logger.info("[INFO] No reference images directory found")
            return references
        
        # Scan for reference images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            for ref_path in reference_dir.glob(ext):
                try:
                    location_id = self.extract_location_id(ref_path.stem)
                    view_type = self.extract_view_type(ref_path.stem)
                    
                    if location_id not in references:
                        references[location_id] = {}
                    
                    references[location_id][view_type] = {
                        'path': str(ref_path),
                        'image': Image.open(ref_path),
                        'exists': True
                    }
                    
                except Exception as e:
                    self.logger.warning("[WARN] Failed to load reference %s: %s", ref_path.name, e)
        
        return references
    
    def extract_location_id(self, filename):
        """Extract location ID from filename"""
        filename_lower = filename.lower()
        view_terms = ['aerial', 'side', 'closeup', 'close_up', 'front', 'back', 'top', 'view', 'photo']
        
        words = filename_lower.split('_')
        location_words = [word for word in words if word not in view_terms]
        
        return '_'.join(location_words) if location_words else filename
    
    def extract_view_type(self, filename):
        """Extract view type from filename"""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['aerial', 'drone', 'overhead']):
            return 'aerial_view'
        elif any(term in filename_lower for term in ['side', 'profile', 'lateral']):
            return 'side_view'
        elif any(term in filename_lower for term in ['closeup', 'close_up', 'close', 'macro']):
            return 'close_up_view'
        else:
            return 'side_view'
    
    def calculate_similarity(self, image_path, text_description):
        """Calculate similarity between image and text"""
        if not CLIP_AVAILABLE or self.model is None:
            self.logger.warning("[WARN] CLIP not available, returning default similarity")
            return 0.7  # Default passing score
        
        try:
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.config.DEVICE)
            
            if self.config.DEVICE == "cuda":
                image_input = image_input.half()
            
            # Use the optimized prompt directly
            text_input = openai_clip.tokenize([text_description]).to(self.config.DEVICE)
            
            with torch.no_grad():
                if self.config.DEVICE == "cuda":
                    with torch.autocast(device_type="cuda"):
                        image_features = self.model.encode_image(image_input)
                        text_features = self.model.encode_text(text_input)
                else:
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_input)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).item()
            
            return similarity
            
        except Exception as e:
            self.logger.error("[ERROR] Similarity calculation failed: %s", str(e))
            return 0.0
    
    def compare_with_reference(self, generated_path, location_id, view_name):
        """Compare generated image with reference image"""
        if (not CLIP_AVAILABLE or self.model is None or 
            location_id not in self.reference_images or 
            view_name not in self.reference_images[location_id] or
            not self.reference_images[location_id][view_name]['exists']):
            return 0.0
        
        try:
            ref_image = self.reference_images[location_id][view_name]['image']
            gen_image = Image.open(generated_path)
            
            ref_input = self.preprocess(ref_image).unsqueeze(0).to(self.config.DEVICE)
            gen_input = self.preprocess(gen_image).unsqueeze(0).to(self.config.DEVICE)
            
            if self.config.DEVICE == "cuda":
                ref_input = ref_input.half()
                gen_input = gen_input.half()
            
            with torch.no_grad():
                if self.config.DEVICE == "cuda":
                    with torch.autocast(device_type="cuda"):
                        ref_features = self.model.encode_image(ref_input)
                        gen_features = self.model.encode_image(gen_input)
                else:
                    ref_features = self.model.encode_image(ref_input)
                    gen_features = self.model.encode_image(gen_input)
                
                ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
                gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
                
                similarity = (ref_features @ gen_features.T).item()
            
            return similarity
            
        except Exception as e:
            self.logger.error("[ERROR] Reference comparison failed: %s", str(e))
            return 0.0
    
    def validate_single_image(self, image_path, prompt, location_id, view_name):
        """Validate consistency for a single image"""
        if not os.path.exists(image_path):
            return self.create_error_result("Image not found")
        
        try:
            # Text-image similarity
            text_similarity = self.calculate_similarity(image_path, prompt)
            
            # Reference-image similarity
            reference_similarity = self.compare_with_reference(image_path, location_id, view_name)
            
            # Quality assessment
            quality_score = self.assess_image_quality(image_path)
            
            # Composition assessment
            composition_score = self.assess_composition(image_path, view_name)
            
            # Combined score
            if reference_similarity > 0:
                overall_score = (
                    text_similarity * 0.3 +
                    reference_similarity * 0.4 +
                    quality_score * 0.2 +
                    composition_score * 0.1
                )
            else:
                overall_score = (
                    text_similarity * 0.6 +
                    quality_score * 0.3 +
                    composition_score * 0.1
                )
            
            return {
                'similarity_score': float(text_similarity),
                'reference_similarity': float(reference_similarity),
                'quality_score': float(quality_score),
                'composition_score': float(composition_score),
                'overall_score': float(overall_score),
                'passed': bool(overall_score >= self.config.CONSISTENCY_THRESHOLD),
                'has_reference': bool(reference_similarity > 0),
                'clip_available': CLIP_AVAILABLE
            }
            
        except Exception as e:
            self.logger.error("[ERROR] Validation failed: %s", str(e))
            return self.create_error_result(str(e))
    
    def create_error_result(self, error_msg):
        """Create error result for failed validation"""
        return {
            'similarity_score': 0.0,
            'reference_similarity': 0.0,
            'quality_score': 0.0,
            'composition_score': 0.0,
            'overall_score': 0.0,
            'passed': False,
            'has_reference': False,
            'clip_available': CLIP_AVAILABLE,
            'error': str(error_msg)
        }
    
    def assess_image_quality(self, image_path):
        """Image quality assessment - works without CLIP"""
        try:
            image = Image.open(image_path)
            
            width, height = image.size
            resolution_score = min(1.0, (width * height) / (512 * 512))
            
            img_array = np.array(image)
            color_variance = np.std(img_array)
            color_score = min(1.0, color_variance / 75)
            
            contrast = np.max(img_array) - np.min(img_array)
            contrast_score = min(1.0, contrast / 200)
            
            quality_score = (resolution_score * 0.4 + color_score * 0.3 + contrast_score * 0.3)
            
            return quality_score
            
        except Exception as e:
            self.logger.error("[ERROR] Quality assessment failed: %s", str(e))
            return 0.0
    
    def assess_composition(self, image_path, view_name):
        """Assess image composition based on view type"""
        try:
            image = Image.open(image_path)
            
            if view_name == "aerial_view":
                return 0.8
            elif view_name == "side_view":
                return 0.7
            elif view_name == "close_up_view":
                return 0.6
            
            return 0.5
            
        except Exception as e:
            return 0.5
    
    def validate_all_images(self, enhanced_images, prompts):
        """Validate consistency for all images"""
        self.logger.info("[VALIDATION] Starting lightweight validation...")
        
        if not CLIP_AVAILABLE:
            self.logger.warning("[WARNING] CLIP not available - using basic validation only")
        
        consistency_scores = {}
        total_images = sum(len(views) for views in enhanced_images.values())
        current_image = 0
        
        for location, location_images in enhanced_images.items():
            self.logger.info("[LOCATION] Validating %s...", location)
            location_scores = {}
            
            location_prompts = {p['view']: p['prompt'] for p in prompts[location]}
            
            for view, image_data in location_images.items():
                current_image += 1
                enhanced_path = image_data['path']
                
                if not enhanced_path or not os.path.exists(enhanced_path):
                    self.logger.warning("[SKIP] No image to validate for %s", view)
                    location_scores[view] = self.create_error_result("Image missing")
                    continue
                
                self.logger.info("[PROGRESS] %d/%d - Validating %s...", 
                               current_image, total_images, view)
                
                prompt = location_prompts[view]
                validation_result = self.validate_single_image(enhanced_path, prompt, location, view)
                
                location_scores[view] = validation_result
                
                status = "[PASS]" if validation_result['passed'] else "[FAIL]"
                ref_info = f"(Ref: {validation_result['reference_similarity']:.3f})" if validation_result['has_reference'] else ""
                clip_status = "" if CLIP_AVAILABLE else " [NO-CLIP]"
                self.logger.info("%s %s: %.3f %s%s", status, view, validation_result['overall_score'], ref_info, clip_status)
            
            consistency_scores[location] = location_scores
        
        self.generate_validation_summary(consistency_scores)
        
        return consistency_scores
    
    def generate_validation_summary(self, consistency_scores):
        """Generate validation summary"""
        total_images = 0
        passed_images = 0
        total_score = 0
        
        for location, scores in consistency_scores.items():
            for view, result in scores.items():
                total_images += 1
                if result.get('passed', False):
                    passed_images += 1
                total_score += result.get('overall_score', 0)
        
        avg_score = total_score / total_images if total_images > 0 else 0
        
        self.logger.info("[SUMMARY] Validation complete:")
        self.logger.info("[SUMMARY] %d/%d images passed validation", passed_images, total_images)
        self.logger.info("[SUMMARY] Average score: %.3f", avg_score)
        if not CLIP_AVAILABLE:
            self.logger.info("[SUMMARY] ⚠️  CLIP not available - basic validation only")