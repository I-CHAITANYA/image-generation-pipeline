"""
Optimized Image Generator for 4GB GPUs - COMPLETELY FIXED
"""

import torch  # Import at TOP level
from PIL import Image
import os
from pathlib import Path
from config import Config
from utils import setup_logging, clear_gpu_memory

class ImageGenerator:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.pipeline = None
        self.device = self.config.DEVICE
        self.setup_model()
    
    def setup_model(self):
        """Initialize Stable Diffusion model with 4GB GPU optimizations"""
        try:
            self.logger.info("[MODEL] Loading optimized model for 4GB GPU: %s", self.config.GENERATION_MODEL)
            
            from diffusers import StableDiffusionPipeline
            
            # Force SD 1.5 for 4GB GPU (smaller than SD 2.1)
            model_name = "runwayml/stable-diffusion-v1-5"
            
            # Load with aggressive memory optimizations
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir="models/stable_diffusion",
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            self.pipeline = self.pipeline.to(self.device)
            self.logger.info("[SUCCESS] Optimized SD 1.5 model loaded for 4GB GPU")
            
            self.apply_low_vram_optimizations()
            
        except Exception as e:
            self.logger.error("[ERROR] Model setup failed: %s", str(e))
            self.fallback_to_minimal_model()
    
    def fallback_to_minimal_model(self):
        """Fallback to absolute minimal settings"""
        try:
            self.logger.info("[FALLBACK] Trying minimal configuration...")
            
            from diffusers import StableDiffusionPipeline
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                cache_dir="models/stable_diffusion"
            )
            
            # Move to device after loading
            self.pipeline = self.pipeline.to(self.device)
            
            # Apply minimal optimizations
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
            
            self.logger.info("[SUCCESS] Minimal fallback model loaded")
            
        except Exception as e:
            self.logger.error("[CRITICAL] All models failed: %s", str(e))
            raise
    
    def apply_low_vram_optimizations(self):
        """Apply aggressive memory optimizations for 4GB GPU"""
        if self.device == "cuda":
            # Essential optimizations for low VRAM
            self.pipeline.enable_attention_slicing()
            self.logger.info("[OPTIMIZATION] Attention slicing enabled")
            
            # Enable VAE slicing (significant memory savings)
            try:
                self.pipeline.enable_vae_slicing()
                self.logger.info("[OPTIMIZATION] VAE slicing enabled")
            except Exception as e:
                self.logger.warning("[OPTIMIZATION] VAE slicing failed: %s", e)
            
            # Try xformers but don't fail
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.logger.info("[OPTIMIZATION] XFormers enabled")
            except Exception:
                self.logger.info("[INFO] XFormers not available")
            
            # Disable model CPU offload (causes issues on low VRAM)
            try:
                self.pipeline.disable_model_cpu_offload()
            except:
                pass
    
    def generate_image(self, prompt, negative_prompt="", seed=None):
        """Generate image with memory-efficient approach - FIXED TORCH SCOPE"""
        try:
            # Clear memory before generation
            clear_gpu_memory()
            
            # Use the torch import from TOP level (not inside function)
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            self.logger.info(f"[GENERATION] Generating image: {prompt[:80]}...")
            
            # Use minimal settings optimized for 4GB GPU
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=512,  # Fixed 512x512 for 4GB
                width=512,
                num_inference_steps=15,  # Reduced steps for 4GB
                guidance_scale=7.0,  # Lower guidance scale
                generator=generator,
                output_type="pil"  # Direct PIL output to save memory
            )
            
            image = result.images[0]
            
            # Clear memory immediately after generation
            clear_gpu_memory()
            
            return image
            
        except Exception as e:
            self.logger.error("[ERROR] Image generation failed: %s", str(e))
            return None
    
    def generate_all_images(self, prompts):
        """Generate images for all locations and views with 4GB optimizations"""
        self.logger.info("[GENERATION] Starting optimized generation for 4GB GPU...")
        
        generated_images = {}
        total_images = sum(len(views) for views in prompts.values())
        current_image = 0
        
        for location_id, location_prompts in prompts.items():
            self.logger.info("[LOCATION] Generating images for %s...", location_id)
            location_images = {}
            
            for i, prompt_data in enumerate(location_prompts):
                current_image += 1
                view = prompt_data['view']
                prompt = prompt_data['prompt']
                negative_prompt = prompt_data.get('negative_prompt', '')
                
                self.logger.info("[PROGRESS] %d/%d - Generating %s...", 
                               current_image, total_images, view)
                
                # Generate image with unique seed for each view
                seed = self.config.SEED + i + hash(location_id) % 1000
                image = self.generate_image(prompt, negative_prompt, seed)
                
                if image:
                    # Save raw image
                    raw_path = f"outputs/{location_id}/raw/{view}_raw.png"
                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                    image.save(raw_path, format='PNG', optimize=True)
                    
                    location_images[view] = {
                        'image': image,
                        'path': raw_path,
                        'prompt': prompt,
                        'seed': seed
                    }
                    
                    self.logger.info("[SUCCESS] Saved: %s", raw_path)
                else:
                    self.logger.error("[FAILED] Generation failed for %s", view)
                    location_images[view] = {
                        'path': None,
                        'prompt': prompt,
                        'error': True
                    }
                
                # Clear memory between images
                clear_gpu_memory()
            
            generated_images[location_id] = location_images
        
        successful = sum(1 for loc in generated_images.values() for img in loc.values() if img.get('path'))
        self.logger.info("[COMPLETE] Generated %d/%d images successfully", successful, total_images)
        
        return generated_images