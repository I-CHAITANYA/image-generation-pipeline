"""
Flexible Prompt Generation for Any Locations with Three Views (Aerial, Side, Close-up)
with Manual Prompt Input Support - OPTIMIZED FOR 4GB GPU
"""

import json
from pathlib import Path
from config import Config

class PromptEngineer:
    def __init__(self):
        self.config = Config()
        self.view_templates = self.get_view_templates()
        self.reference_images = self.check_reference_images()
    
    def get_view_templates(self):
        """Return the three specific views: aerial, side, and close-up"""
        return {
            "aerial_view": {
                "description": "Bird's eye view from above showing overall layout and surroundings",
                "keywords": "aerial view, bird's eye view, drone photography, overhead perspective"
            },
            "side_view": {
                "description": "Full profile view showing complete side轮廓",
                "keywords": "side view, profile, full side, complete side profile"
            },
            "close_up_view": {
                "description": "Detailed close-up view showing intricate features and textures",
                "keywords": "close up, macro view, detailed shot, intricate details"
            }
        }
    
    def check_reference_images(self):
        """Check if reference images exist for any locations"""
        reference_dir = Path("inputs/reference_images")
        references = {}
        
        if not reference_dir.exists():
            return references
        
        # Scan for any image files in reference_images directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        all_reference_files = []
        for ext in image_extensions:
            all_reference_files.extend(reference_dir.glob(ext))
        
        # Auto-detect location names from filenames
        for ref_path in all_reference_files:
            filename = ref_path.stem.lower()
            
            # Try to extract location name and view type from filename
            location_id = self.extract_location_id(filename)
            view_type = self.extract_view_type(filename)
            
            if location_id and view_type:
                if location_id not in references:
                    references[location_id] = {}
                
                references[location_id][view_type] = {
                    'path': str(ref_path),
                    'exists': True,
                    'size': ref_path.stat().st_size,
                    'filename': ref_path.name
                }
        
        return references
    
    def extract_location_id(self, filename):
        """Extract location ID from filename"""
        # Remove common view terms to get location name
        view_terms = ['aerial', 'side', 'closeup', 'close_up', 'front', 'back', 'top', 'view', 'photo']
        words = filename.split('_')
        
        # Filter out view terms and join remaining words as location ID
        location_words = [word for word in words if word not in view_terms]
        
        if location_words:
            return '_'.join(location_words)
        return filename
    
    def extract_view_type(self, filename):
        """Extract view type from filename"""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['aerial', 'drone', 'overhead', 'from above']):
            return 'aerial_view'
        elif any(term in filename_lower for term in ['side', 'profile', 'lateral']):
            return 'side_view'
        elif any(term in filename_lower for term in ['closeup', 'close_up', 'close', 'macro', 'detail']):
            return 'close_up_view'
        else:
            return 'side_view'
    
    def get_custom_locations(self):
        """Allow user to input custom locations"""
        print("\n📍 CUSTOM LOCATION SETUP")
        print("=" * 50)
        print("You can add any locations you want to generate images for.")
        print("For each location, you'll be able to provide descriptions and prompts.")
        
        locations = {}
        
        while True:
            print(f"\nCurrent locations: {list(locations.keys())}")
            print("\n1. Add new location")
            print("2. Use default locations (Mumbai, Rajasthan, Nagpur)")
            print("3. Continue with current locations")
            
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == "1":
                self.add_custom_location(locations)
            elif choice == "2":
                locations.update(self.get_default_locations())
                print("✅ Added default locations")
            elif choice == "3":
                if not locations:
                    print("⚠️  No locations selected. Using default locations.")
                    locations = self.get_default_locations()
                break
            else:
                print("❌ Invalid choice. Please try again.")
        
        return locations
    
    def add_custom_location(self, locations):
        """Add a custom location"""
        print("\n➕ ADDING NEW LOCATION")
        
        location_name = input("Enter location name (e.g., 'Eiffel Tower', 'Grand Canyon'): ").strip()
        if not location_name:
            print("❌ Location name cannot be empty.")
            return
        
        location_id = location_name.lower().replace(' ', '_')
        
        if location_id in locations:
            print(f"❌ Location '{location_name}' already exists.")
            return
        
        location_type = input("Enter location type (architecture/nature/landmark/other): ").strip().lower()
        if not location_type:
            location_type = "landmark"
        
        description = input(f"Enter description for {location_name} (or press Enter for auto): ").strip()
        if not description:
            description = self.generate_auto_description(location_name, location_type)
            print(f"🔧 Auto-generated description: {description}")
        
        locations[location_id] = {
            "display_name": location_name,
            "type": location_type,
            "description": description,
            "custom": True
        }
        
        print(f"✅ Added location: {location_name}")
    
    def generate_auto_description(self, location_name, location_type):
        """Generate automatic description based on location name and type"""
        base_descriptions = {
            "architecture": f"{location_name}, notable architectural structure, impressive design",
            "nature": f"{location_name}, natural landscape, beautiful scenery, environmental features",
            "landmark": f"{location_name}, famous landmark, iconic destination, popular attraction", 
            "other": f"{location_name}, interesting location, distinctive features"
        }
        
        return base_descriptions.get(location_type, f"{location_name}, notable location with unique features")
    
    def get_default_locations(self):
        """Get default locations"""
        return {
            "mumbai_bandra_worli_sea_link": {
                "display_name": "Mumbai Bandra-Worli Sea Link",
                "type": "architecture",
                "description": "Mumbai Bandra-Worli Sea Link cable-stayed bridge over Arabian Sea, modern architecture, urban landscape",
                "custom": False
            },
            "rajasthan_hawa_mahal": {
                "display_name": "Rajasthan Hawa Mahal", 
                "type": "architecture",
                "description": "Rajasthan Hawa Mahal in Jaipur, five-story pink sandstone palace with intricate latticework windows, traditional Rajasthani architecture",
                "custom": False
            },
            "nagpur_rainforest": {
                "display_name": "Nagpur Rainforest",
                "type": "nature", 
                "description": "Nagpur rainforest with dense tropical vegetation, lush greenery, tall trees, natural landscape",
                "custom": False
            }
        }
    
    def manual_prompt_input(self, locations):
        """Allow user to manually input prompts for each location and view"""
        print("\n🎯 MANUAL PROMPT INPUT MODE")
        print("=" * 50)
        
        views = list(self.view_templates.keys())
        manual_prompts = {}
        
        for location_id, location_info in locations.items():
            location_name = location_info["display_name"]
            print(f"\n📍 {location_name}:")
            location_prompts = []
            
            for view_name in views:
                view_display = self.view_templates[view_name]['description']
                print(f"\n   🎯 {view_name.replace('_', ' ').title()}:")
                print(f"   📝 {view_display}")
                
                # Show reference status if available
                has_ref = self.has_reference(location_id, view_name)
                ref_status = " ✅ (Reference available)" if has_ref else ""
                print(f"   📸 {ref_status}")
                
                # Get manual prompt input
                default_prompt = self.build_prompt(location_name, location_info["description"], view_name, self.view_templates[view_name], location_info["type"])
                print(f"   💡 Default: {default_prompt[:100]}...")
                
                prompt = input(f"   💬 Enter custom prompt (or press Enter for default): ").strip()
                if not prompt:
                    prompt = default_prompt
                    print("   🔧 Using default prompt")
                
                # Get negative prompt
                default_negative = self.get_negative_prompt(view_name, location_info["type"])
                negative_prompt = input(f"   🚫 Enter negative prompt (or press Enter for default): ").strip()
                if not negative_prompt:
                    negative_prompt = default_negative
                    print("   🔧 Using default negative prompt")
                
                location_prompts.append({
                    "view": view_name,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "description": f"{view_display} of {location_name}",
                    "has_reference": has_ref,
                    "input_mode": "manual",
                    "location_type": location_info["type"]
                })
            
            manual_prompts[location_id] = location_prompts
        
        return manual_prompts
    
    def has_reference(self, location_id, view_name):
        """Check if reference image exists for location and view"""
        return (location_id in self.reference_images and 
                view_name in self.reference_images[location_id] and 
                self.reference_images[location_id][view_name]['exists'])
    
    def auto_generate_prompts(self, locations):
        """Automatically generate prompts for all locations and views"""
        print("\n🤖 AUTO-GENERATING PROMPTS")
        print("=" * 50)
        
        auto_prompts = {}
        
        for location_id, location_info in locations.items():
            location_prompts = self.generate_location_prompts(location_info, location_id)
            auto_prompts[location_id] = location_prompts
        
        return auto_prompts
    
    def generate_location_prompts(self, location_info, location_id):
        """Generate prompts for all three views for a specific location"""
        prompts = []
        
        for view_name, view_template in self.view_templates.items():
            prompt = self.build_prompt(
                location_info["display_name"], 
                location_info["description"], 
                view_name, 
                view_template, 
                location_info["type"]
            )
            
            # Enhance with reference context if available
            enhanced_prompt = self.enhance_with_references(prompt, location_id, view_name)
            
            prompts.append({
                "view": view_name,
                "prompt": enhanced_prompt,
                "negative_prompt": self.get_negative_prompt(view_name, location_info["type"]),
                "description": f"{view_template['description']} of {location_info['display_name']}",
                "has_reference": self.has_reference(location_id, view_name),
                "input_mode": "auto",
                "location_type": location_info["type"]
            })
        
        return prompts
    
    def enhance_with_references(self, base_prompt, location_id, view_name):
        """Enhance prompts with reference image context"""
        if self.has_reference(location_id, view_name):
            return f"{base_prompt} Consistent with reference photography, maintaining accurate proportions"
        return base_prompt
    
    def build_prompt(self, location_name, location_description, view_name, view_template, location_type):
        """Build optimized prompt that fits within CLIP limits"""
        # Create concise base description
        base_description = f"{view_template['description']} of {location_name}"
        
        # Use shorter keywords
        view_keywords = view_template['keywords']
        
        # Quality boosters (shorter version)
        quality_boosters = "professional photography, high quality, detailed, sharp focus"
        
        # Location-type specific enhancements (shorter)
        location_enhancements = self.get_optimized_location_enhancements(location_type, view_name)
        
        prompt = f"{base_description}. {view_keywords}. {location_enhancements}. {quality_boosters}"
        
        # Ensure prompt is not too long
        if len(prompt.split()) > 60:  # Rough token count estimate
            prompt = f"{base_description}. {view_keywords}. {quality_boosters}"
        
        return prompt
    
    def get_optimized_location_enhancements(self, location_type, view_name):
        """Get shorter location-type enhancements"""
        enhancements = {
            "architecture": {
                "aerial_view": "architectural overview, structural layout",
                "side_view": "architectural profile, building facade", 
                "close_up_view": "architectural details, material textures"
            },
            "nature": {
                "aerial_view": "landscape overview, environmental context",
                "side_view": "natural profile, landscape layers",
                "close_up_view": "natural details, environmental textures"
            },
            "landmark": {
                "aerial_view": "landmark overview, iconic structure",
                "side_view": "landmark profile, iconic features",
                "close_up_view": "landmark details, characteristic features"
            }
        }
        
        return enhancements.get(location_type, {}).get(view_name, "professional photography")
    
    def get_negative_prompt(self, view_name, location_type):
        """Get negative prompts specific to each view type and location type"""
        base_negative = "blurry, low quality, distorted, unrealistic colors, watermark, text, logo, people"
        
        view_specific = {
            "aerial_view": "ground level, close up, side view, low angle, indoor",
            "side_view": "aerial view, close up, front view, top down, interior", 
            "close_up_view": "wide angle, aerial view, distant, landscape, far away"
        }
        
        location_specific = {
            "architecture": "natural landscape, forest, mountains, water, animals",
            "nature": "buildings, architecture, urban, city, man-made structures",
            "landmark": "generic, ordinary, common, unremarkable, plain"
        }
        
        location_negative = location_specific.get(location_type, "inappropriate, mismatched, inconsistent")
        
        return f"{base_negative}, {view_specific.get(view_name, '')}, {location_negative}"
    
    def generate_all_prompts(self, mode="auto"):
        """Generate prompts for any locations and three views"""
        # Step 1: Get locations (custom or default)
        locations = self.get_custom_locations()
        
        # Step 2: Generate prompts based on mode
        if mode == "manual":
            prompts = self.manual_prompt_input(locations)
        else:
            prompts = self.auto_generate_prompts(locations)
        
        # Save prompts for reference
        self.save_prompts(prompts, mode, locations)
        self.log_reference_status()
        self.log_location_summary(locations)
        
        return prompts
    
    def save_prompts(self, prompts, mode, locations):
        """Save generated prompts to JSON file"""
        output_path = Path("inputs/prompt_templates.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        enhanced_prompts = {
            **prompts,
            "metadata": {
                "version": "3.0",
                "input_mode": mode,
                "views": list(self.view_templates.keys()),
                "locations": locations,
                "has_references": self.has_any_references(),
                "reference_status": self.get_reference_summary(),
                "total_locations": len(prompts),
                "total_views": sum(len(views) for views in prompts.values()),
                "custom_locations": any(loc.get('custom', False) for loc in locations.values())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"💾 {mode.capitalize()} prompts saved to: inputs/prompt_templates.json")
    
    def has_any_references(self):
        """Check if any reference images exist"""
        for location_refs in self.reference_images.values():
            for view_ref in location_refs.values():
                if view_ref.get('exists', False):
                    return True
        return False
    
    def get_reference_summary(self):
        """Get summary of reference image status"""
        summary = {}
        for location_id, view_refs in self.reference_images.items():
            summary[location_id] = {
                view_name: ref_info.get('exists', False) 
                for view_name, ref_info in view_refs.items()
            }
        return summary
    
    def log_reference_status(self):
        """Log reference image status"""
        if not self.reference_images:
            print("📸 No reference images found in inputs/reference_images/")
            return
            
        print("📸 Reference Image Status:")
        for location_id, view_refs in self.reference_images.items():
            ref_count = sum(1 for ref in view_refs.values() if ref.get('exists', False))
            print(f"   📍 {location_id}: {ref_count}/3 views")
            for view_name, ref_info in view_refs.items():
                if ref_info.get('exists', False):
                    print(f"      ✅ {view_name}: {ref_info.get('filename', 'Reference')}")
    
    def log_location_summary(self, locations):
        """Log location summary"""
        print("\n📍 LOCATION SUMMARY:")
        custom_count = sum(1 for loc in locations.values() if loc.get('custom', False))
        default_count = len(locations) - custom_count
        
        print(f"   Total locations: {len(locations)}")
        print(f"   Custom locations: {custom_count}")
        print(f"   Default locations: {default_count}")
        
        for location_id, location_info in locations.items():
            custom_flag = " (Custom)" if location_info.get('custom', False) else ""
            print(f"   ✅ {location_info['display_name']}{custom_flag}")