#!/usr/bin/env python3
"""
Main Pipeline Script - Enhanced with Best Open-Source Models and Flexible Locations
Coordinates the complete image generation workflow for any locations
"""

import os
import sys
import time
from pathlib import Path
from prompt_engineer import PromptEngineer
from image_generator import ImageGenerator
from quality_enhancer import QualityEnhancer
from consistency_checker import ConsistencyChecker
from utils import setup_logging, save_metadata, get_gpu_memory_info, optimize_memory_usage
from config import Config

class ThreeViewImagePipeline:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.manual_mode = False
        self.custom_locations = False
        
    def run(self):
        """Execute the enhanced three-view image generation pipeline"""
        self.logger.info("[START] Starting Enhanced Image Generation Pipeline")
        self.logger.info("[MODELS] Using best open-source models for optimal quality")
        
        try:
            # Display comprehensive system info
            self.log_system_info()
            
            # Optimize memory usage
            optimize_memory_usage()
            
            # Display generation plan
            self.display_generation_plan()
            
            # Step 1: Generate prompts (auto or manual)
            prompt_mode = "manual" if self.manual_mode else "auto"
            self.logger.info("[PROMPT] Step 1: Generating prompts (%s mode)...", prompt_mode)
            prompt_engineer = PromptEngineer()
            prompts = prompt_engineer.generate_all_prompts(mode=prompt_mode)
            
            if not prompts:
                self.logger.error("[ERROR] No prompts generated. Pipeline cannot continue.")
                return
            
            # Step 2: Generate images with best models
            self.logger.info("[GENERATION] Step 2: Generating images with %s...", self.config.GENERATION_MODEL)
            image_generator = ImageGenerator()
            raw_images = image_generator.generate_all_images(prompts)
            
            # Check if generation was successful
            successful_generations = self.count_successful_images(raw_images)
            if successful_generations == 0:
                self.logger.error("[ERROR] No images generated successfully. Stopping pipeline.")
                return
            
            # Step 3: Enhance image quality with best upscaler
            self.logger.info("[ENHANCEMENT] Step 3: Enhancing image quality with %s...", self.config.ENHANCEMENT_MODEL)
            quality_enhancer = QualityEnhancer()
            enhanced_images = quality_enhancer.enhance_all_images(raw_images)
            
            # Step 4: Validate consistency with better CLIP model
            self.logger.info("[VALIDATION] Step 4: Validating consistency with %s...", self.config.VALIDATION_MODEL)
            consistency_checker = ConsistencyChecker()
            consistency_scores = consistency_checker.validate_all_images(enhanced_images, prompts)
            
            # Step 5: Save comprehensive results and metadata
            self.logger.info("[RESULTS] Step 5: Saving results and comprehensive metadata...")
            self.save_final_results(enhanced_images, consistency_scores, prompts, raw_images)
            
            # Step 6: Generate final summary report
            self.logger.info("[SUMMARY] Step 6: Generating final evaluation summary...")
            self.generate_final_summary(enhanced_images, consistency_scores)
            
            self.logger.info("[SUCCESS] Enhanced pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error("[ERROR] Pipeline failed: %s", str(e))
            self.logger.info("[TROUBLESHOOTING] Check: GPU memory, internet connection, model downloads")
            raise
    
    def log_system_info(self):
        """Log comprehensive system and model information"""
        import torch
        
        self.logger.info("[SYSTEM] Python %s", sys.version.split()[0])
        self.logger.info("[SYSTEM] PyTorch %s", torch.__version__)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info("[HARDWARE] GPU: %s (%.1f GB VRAM)", gpu_name, vram)
            self.logger.info("[HARDWARE] CUDA: %s", torch.version.cuda)
        else:
            self.logger.info("[HARDWARE] Using CPU (GPU not available)")
        
        # Model information
        self.logger.info("[MODELS] Generation: %s", self.config.GENERATION_MODEL)
        self.logger.info("[MODELS] Enhancement: %s", self.config.ENHANCEMENT_MODEL)
        self.logger.info("[MODELS] Validation: %s", self.config.VALIDATION_MODEL)
        self.logger.info("[SETTINGS] Resolution: %dx%d", self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        self.logger.info("[SETTINGS] Inference steps: %d", self.config.NUM_INFERENCE_STEPS)
        
        # Memory information
        mem_info = get_gpu_memory_info()
        self.logger.info("[MEMORY] GPU: %s allocated, %s reserved", 
                        mem_info['allocated_gb'], mem_info['reserved_gb'])
    
    def display_generation_plan(self):
        """Display the generation plan with location and view details"""
        self.logger.info("[PLAN] Generation Plan:")
        self.logger.info("[PLAN] =========================================")
        
        # Note: Actual locations will be determined during prompt generation
        self.logger.info("[PLAN] Location Mode: %s", 
                        "Custom Locations" if self.custom_locations else "Default Locations")
        self.logger.info("[PLAN] Prompt Mode: %s", 
                        "Manual Input" if self.manual_mode else "Auto-Generated")
        self.logger.info("[PLAN] Views per Location: Aerial, Side, Close-up")
        self.logger.info("[PLAN] Total Images: Variable (based on location count × 3)")
        self.logger.info("[PLAN] =========================================")
    
    def count_successful_images(self, raw_images):
        """Count successfully generated images"""
        successful = 0
        total = 0
        
        for location, location_images in raw_images.items():
            for view, image_data in location_images.items():
                total += 1
                if image_data.get('path') and os.path.exists(image_data['path']):
                    successful += 1
        
        self.logger.info("[STATS] Successfully generated: %d/%d images", successful, total)
        return successful
    
    def save_final_results(self, enhanced_images, consistency_scores, prompts, raw_images):
        """Save comprehensive results and metadata for all locations"""
        for location_id, location_images in enhanced_images.items():
            # Get location info from prompts
            location_info = self.get_location_info(prompts, location_id)
            
            # Convert all values to JSON-serializable types
            metadata = {
                'location': location_info['display_name'],
                'location_type': location_info.get('type', 'unknown'),
                'location_id': location_id,
                'custom_location': bool(location_info.get('custom', False)),  # Convert to bool
                
                # Generation information
                'generation_model': str(self.config.GENERATION_MODEL),
                'enhancement_model': str(self.config.ENHANCEMENT_MODEL),
                'validation_model': str(self.config.VALIDATION_MODEL),
                'resolution': f"{self.config.IMAGE_WIDTH}x{self.config.IMAGE_HEIGHT}",
                'inference_steps': int(self.config.NUM_INFERENCE_STEPS),
                'guidance_scale': float(self.config.GUIDANCE_SCALE),
                
                # Views and prompts
                'views_generated': list(location_images.keys()),
                'prompts': {
                    view: {
                        'view': prompt_data.get('view', ''),
                        'prompt': prompt_data.get('prompt', ''),
                        'input_mode': prompt_data.get('input_mode', 'auto')
                    }
                    for view in location_images.keys()
                    for prompt_data in prompts.get(location_id, [])
                    if prompt_data.get('view') == view
                },
                
                # Convert consistency scores to serializable format
                'consistency_scores': {
                    view: {
                        'similarity_score': float(score.get('similarity_score', 0)),
                        'reference_similarity': float(score.get('reference_similarity', 0)),
                        'quality_score': float(score.get('quality_score', 0)),
                        'composition_score': float(score.get('composition_score', 0)),
                        'overall_score': float(score.get('overall_score', 0)),
                        'passed': bool(score.get('passed', False)),
                        'has_reference': bool(score.get('has_reference', False))
                    }
                    for view, score in consistency_scores.get(location_id, {}).items()
                },
                
                'validation_threshold': float(self.config.CONSISTENCY_THRESHOLD),
                
                # Image counts
                'total_images': int(len(location_images)),
                'successful_generations': int(self.count_successful_location_images(raw_images.get(location_id, {}))),
                
                # System information
                'device_used': "GPU" if self.config.DEVICE == "cuda" else "CPU",
                'pipeline_version': '3.0-enhanced',
                'input_mode': 'manual' if self.manual_mode else 'auto'
            }
            
            # Add enhancement method information
            enhancement_methods = {}
            for view, image_data in location_images.items():
                if 'enhancement_method' in image_data:
                    enhancement_methods[view] = str(image_data['enhancement_method'])
            
            if enhancement_methods:
                metadata['enhancement_methods'] = enhancement_methods
            
            save_metadata(location_id, metadata)
            self.logger.info("[SAVE] Comprehensive metadata saved for %s", location_id)
        
    def get_location_info(self, prompts, location_id):
        """Extract location information from prompts"""
        if 'metadata' in prompts and 'locations' in prompts['metadata']:
            return prompts['metadata']['locations'].get(location_id, {
                'display_name': location_id.replace('_', ' ').title(),
                'type': 'unknown'
            })
        
        # Fallback: extract from first prompt
        if location_id in prompts and prompts[location_id]:
            first_prompt = prompts[location_id][0]
            return {
                'display_name': location_id.replace('_', ' ').title(),
                'type': first_prompt.get('location_type', 'unknown'),
                'custom': first_prompt.get('input_mode') == 'manual'
            }
        
        return {'display_name': location_id.replace('_', ' ').title(), 'type': 'unknown'}
    
    def count_successful_location_images(self, location_images):
        """Count successful images for a specific location"""
        if not location_images:
            return 0
        return sum(1 for img_data in location_images.values() 
                  if img_data.get('path') and os.path.exists(img_data['path']))
    
    def generate_final_summary(self, enhanced_images, consistency_scores):
        """Generate comprehensive final summary report"""
        self.logger.info("[SUMMARY] =========================================")
        self.logger.info("[SUMMARY] FINAL PIPELINE SUMMARY")
        self.logger.info("[SUMMARY] =========================================")
        
        total_locations = len(enhanced_images)
        total_images = sum(len(images) for images in enhanced_images.values())
        
        # Calculate success rates
        successful_images = 0
        passed_validation = 0
        total_score = 0
        
        for location_id, location_images in enhanced_images.items():
            location_scores = consistency_scores.get(location_id, {})
            
            for view, image_data in location_images.items():
                if image_data.get('path') and os.path.exists(image_data['path']):
                    successful_images += 1
                    
                    view_score = location_scores.get(view, {})
                    if view_score.get('passed', False):
                        passed_validation += 1
                    
                    total_score += view_score.get('overall_score', 0)
        
        # Summary statistics
        self.logger.info("[SUMMARY] Locations processed: %d", total_locations)
        self.logger.info("[SUMMARY] Total images generated: %d/%d", successful_images, total_images)
        self.logger.info("[SUMMARY] Images passed validation: %d/%d", passed_validation, successful_images)
        
        if successful_images > 0:
            avg_score = total_score / successful_images
            success_rate = (successful_images / total_images) * 100
            validation_rate = (passed_validation / successful_images) * 100
            
            self.logger.info("[SUMMARY] Generation success rate: %.1f%%", success_rate)
            self.logger.info("[SUMMARY] Validation pass rate: %.1f%%", validation_rate)
            self.logger.info("[SUMMARY] Average quality score: %.3f", avg_score)
        
        # Location-wise breakdown
        self.logger.info("[SUMMARY] -----------------------------------------")
        self.logger.info("[SUMMARY] Location-wise Results:")
        
        for location_id, location_images in enhanced_images.items():
            location_scores = consistency_scores.get(location_id, {})
            location_success = self.count_successful_location_images(location_images)
            location_passed = sum(1 for score in location_scores.values() if score.get('passed', False))
            
            self.logger.info("[SUMMARY] 📍 %s: %d/%d generated, %d/%d passed", 
                           location_id, location_success, len(location_images), 
                           location_passed, len(location_scores))
        
        # Model performance
        self.logger.info("[SUMMARY] -----------------------------------------")
        self.logger.info("[SUMMARY] Model Performance:")
        self.logger.info("[SUMMARY] Generation: %s", self.config.GENERATION_MODEL)
        self.logger.info("[SUMMARY] Enhancement: %s", self.config.ENHANCEMENT_MODEL)
        self.logger.info("[SUMMARY] Validation: %s", self.config.VALIDATION_MODEL)
        
        # Output locations
        self.logger.info("[SUMMARY] -----------------------------------------")
        self.logger.info("[SUMMARY] Output Directory: outputs/")
        self.logger.info("[SUMMARY] Check individual location folders for:")
        self.logger.info("[SUMMARY]   - Raw generated images (outputs/[location]/raw/)")
        self.logger.info("[SUMMARY]   - Enhanced images (outputs/[location]/enhanced/)")
        self.logger.info("[SUMMARY]   - Metadata and scores (outputs/[location]/metadata.json)")
        self.logger.info("[SUMMARY] =========================================")
        
        # Save summary to file
        self.save_summary_to_file(enhanced_images, consistency_scores, 
                                successful_images, total_images, passed_validation)
    
    def save_summary_to_file(self, enhanced_images, consistency_scores, 
                           successful_images, total_images, passed_validation):
        """Save summary report to file"""
        summary_path = "results/final_evaluation_summary.md"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Image Generation Pipeline - Final Summary\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Overall Statistics\n")
            f.write(f"- **Total Locations:** {len(enhanced_images)}\n")
            f.write(f"- **Total Images Attempted:** {total_images}\n")
            f.write(f"- **Successfully Generated:** {successful_images}\n")
            f.write(f"- **Passed Validation:** {passed_validation}\n")
            
            if total_images > 0:
                success_rate = (successful_images / total_images) * 100
                validation_rate = (passed_validation / successful_images) * 100 if successful_images > 0 else 0
                f.write(f"- **Generation Success Rate:** {success_rate:.1f}%\n")
                f.write(f"- **Validation Pass Rate:** {validation_rate:.1f}%\n")
            
            f.write("\n## 🤖 Models Used\n")
            f.write(f"- **Generation Model:** {self.config.GENERATION_MODEL}\n")
            f.write(f"- **Enhancement Model:** {self.config.ENHANCEMENT_MODEL}\n")
            f.write(f"- **Validation Model:** {self.config.VALIDATION_MODEL}\n")
            
            f.write("\n## 📍 Location Details\n")
            for location_id, location_images in enhanced_images.items():
                location_scores = consistency_scores.get(location_id, {})
                location_success = self.count_successful_location_images(location_images)
                location_passed = sum(1 for score in location_scores.values() if score.get('passed', False))
                
                f.write(f"\n### {location_id.replace('_', ' ').title()}\n")
                f.write(f"- **Generated:** {location_success}/{len(location_images)} images\n")
                f.write(f"- **Validated:** {location_passed}/{len(location_scores)} passed\n")
                
                # View-specific scores
                for view, score_data in location_scores.items():
                    status = "✅ PASS" if score_data.get('passed', False) else "❌ FAIL"
                    f.write(f"  - {view}: {status} (Score: {score_data.get('overall_score', 0):.3f})\n")
        
        self.logger.info("[SUMMARY] Detailed summary saved to: %s", summary_path)


def main():
    """Main entry point for the enhanced pipeline"""
    try:
        pipeline = ThreeViewImagePipeline()
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)