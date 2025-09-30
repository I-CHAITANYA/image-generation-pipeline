"""
Optimized Quality Enhancer for 4GB GPUs
"""

import torch
from PIL import Image
import os
from pathlib import Path
from config import Config
from utils import setup_logging

class QualityEnhancer:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.upscaler = None
        self.enhancement_method = "pil"  # Default to PIL for 4GB GPUs
        self.setup_enhancer()
    
    def setup_enhancer(self):
        """Initialize lightweight enhancer for 4GB GPUs"""
        try:
            self.logger.info("[ENHANCER] Setting up lightweight enhancement for 4GB GPU...")
            
            # For 4GB GPUs, use PIL by default - no GPU memory usage
            self.enhancement_method = "pil"
            self.upscaler = None
            self.logger.info("[SUCCESS] Using PIL enhancement (GPU memory friendly)")
            
        except Exception as e:
            self.logger.error("[ERROR] Failed to setup enhancer: %s", str(e))
            self.enhancement_method = "pil"
            self.upscaler = None
    
    def enhance_image(self, image_path):
        """Enhance a single image using lightweight method"""
        if not os.path.exists(image_path):
            self.logger.error("[ERROR] Image not found: %s", image_path)
            return None
        
        try:
            if self.enhancement_method == "pil":
                return self.enhance_basic(image_path)
            else:
                # Fallback to basic enhancement
                return self.enhance_basic(image_path)
                
        except Exception as e:
            self.logger.error("[ERROR] Enhancement failed: %s", str(e))
            return self.enhance_basic(image_path)
    
    def enhance_basic(self, image_path):
        """Basic enhancement using PIL (memory efficient)"""
        try:
            img = Image.open(image_path)
            
            # Calculate new size
            new_size = (
                img.width * self.config.UPSCALE_FACTOR,
                img.height * self.config.UPSCALE_FACTOR
            )
            
            # Use LANCZOS resampling for good quality
            enhanced = img.resize(new_size, Image.LANCZOS)
            
            # Basic sharpening
            from PIL import ImageFilter
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
            
            return enhanced
            
        except Exception as e:
            self.logger.error("[ERROR] Basic enhancement failed: %s", str(e))
            return None
    
    def enhance_all_images(self, raw_images):
        """Enhance all raw images with memory efficiency"""
        self.logger.info("[ENHANCEMENT] Starting lightweight enhancement...")
        
        enhanced_images = {}
        total_images = sum(len(views) for views in raw_images.values())
        current_image = 0
        
        for location, location_images in raw_images.items():
            self.logger.info("[LOCATION] Enhancing images for %s...", location)
            location_enhanced = {}
            
            for view, image_data in location_images.items():
                current_image += 1
                raw_path = image_data['path']
                
                if raw_path is None or not os.path.exists(raw_path):
                    self.logger.warning("[SKIP] No image to enhance for %s", view)
                    continue
                
                self.logger.info("[PROGRESS] %d/%d - Enhancing %s...", 
                               current_image, total_images, view)
                
                enhanced_image = self.enhance_image(raw_path)
                
                if enhanced_image:
                    # Save enhanced image
                    enhanced_path = f"outputs/{location}/enhanced/{view}_enhanced.png"
                    os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
                    enhanced_image.save(enhanced_path, format='PNG', optimize=True)
                    
                    location_enhanced[view] = {
                        'image': enhanced_image,
                        'path': enhanced_path,
                        'original_path': raw_path,
                        'enhancement_method': self.enhancement_method
                    }
                    
                    self.logger.info("[SUCCESS] Enhanced: %s", enhanced_path)
                else:
                    self.logger.error("[FAILED] Enhancement failed for %s", view)
                    # Copy original as fallback
                    self.create_fallback_enhanced(raw_path, location, view)
                    location_enhanced[view] = {
                        'path': f"outputs/{location}/enhanced/{view}_enhanced.png",
                        'original_path': raw_path,
                        'enhancement_method': 'fallback'
                    }
            
            enhanced_images[location] = location_enhanced
        
        self.logger.info("[COMPLETE] Enhanced %d images", 
                        sum(len(imgs) for imgs in enhanced_images.values()))
        
        return enhanced_images
    
    def create_fallback_enhanced(self, raw_path, location, view):
        """Create fallback enhanced image by copying original"""
        try:
            enhanced_path = f"outputs/{location}/enhanced/{view}_enhanced.png"
            os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
            
            import shutil
            shutil.copy2(raw_path, enhanced_path)
            self.logger.info("[FALLBACK] Created fallback enhanced image")
            
        except Exception as e:
            self.logger.error("[ERROR] Fallback creation failed: %s", e)