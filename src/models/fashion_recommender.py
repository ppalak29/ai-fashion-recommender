import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os

from src.models.clip_encoder import CLIPEncoder
from src.models.blip2_analyzer import BLIP2Analyzer
from src.outfit.combiner import OutfitCombiner

class FashionRecommender:
    
    def __init__(self, use_blip2=True):
        self.clip_encoder = CLIPEncoder()
        self.device = self.clip_encoder.device
        
        # Storage
        self.clothing_database = {}
        self.categorized_items = {}
        self.item_metadata = {}
        
        if use_blip2:
            self.blip2_analyzer = BLIP2Analyzer()
        else:
            raise ValueError("BLIP-2 is required")
        
        self.outfit_combiner = None
    
    def add_clothing_item(self, image_path):
        print(f"\nProcessing {Path(image_path).name}...")
        
        try:
            image_features = self.clip_encoder.encode_image(image_path)
            self.clothing_database[image_path] = image_features
            
            analysis = self.blip2_analyzer.categorize_item(image_path)
            category = analysis['category']
            self.item_metadata[image_path] = analysis
            
            # Store categorized item
            if category != 'other':
                if category not in self.categorized_items:
                    self.categorized_items[category] = []
                
                self.categorized_items[category].append({
                    'path': image_path,
                    'category': category,
                    'analysis': analysis
                })
            
            print(f"Added as {category}")
            print(f"  Style: {analysis['style']}, Color: {analysis['color']}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    def search_outfits(self, text_query, top_k=5):
        if not self.clothing_database:
            print("No clothing items in database!")
            return []
        
        print(f"Searching for: '{text_query}'")
        
        # Use CLIPEncoder for text encoding
        text_features = self.clip_encoder.encode_text(text_query)
        
        similarities = []
        for image_path, image_features in self.clothing_database.items():
            similarity = torch.cosine_similarity(
                text_features,
                image_features,
                dim=-1
            ).item()
            similarities.append((image_path, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def initialize_outfit_combiner(self):
        print("Initializing outfit combination engine...")
        self.outfit_combiner = OutfitCombiner(self)
        print("Outfit combiner ready!")
    
    def display_outfit_combinations(self, text_query, top_k=3):
        """
        Search for and display complete outfit combinations
        """
        
        if not self.outfit_combiner:
            print("   Outfit combiner not initialized!")
            print("   Call initialize_outfit_combiner() first")
            return
        
        # Get outfit combinations
        combinations = self.outfit_combiner.create_outfit_combinations(text_query, top_k)
        
        if not combinations:
            print("No outfit combinations found!")
            return
        
        print(f"\n🎯 Top {len(combinations)} outfit combinations for: '{text_query}'")
        print("=" * 80)
        
        for combo_num, combo in enumerate(combinations, 1):
            print(f"\n{'='*80}")
            print(f"OUTFIT #{combo_num} - Overall Score: {combo['overall_score']:.3f}")
            print(f"Template: {combo['template']}")
            print(f"Compatibility: {combo['compatibility_score']:.3f}")
            print('='*80)
            
            # Get outfit items
            outfit_items = list(combo['outfit'].items())
            
            # Create figure to display this outfit
            fig, axes = plt.subplots(1, len(outfit_items), figsize=(5*len(outfit_items), 6))
            
            if len(outfit_items) == 1:
                axes = [axes]
            
            # Display each item in the outfit
            for idx, (category, item) in enumerate(outfit_items):
                try:
                    # Load image
                    image = Image.open(item['path'])
                    axes[idx].imshow(image)
                    
                    # Get individual score
                    individual_score = combo['individual_scores'].get(category, 'N/A')
                    
                    # Create title
                    filename = Path(item['path']).name
                    if individual_score != 'N/A':
                        title = f"{category.upper()}\n{filename}\nScore: {individual_score:.3f}"
                    else:
                        title = f"{category.upper()}\n{filename}"
                    
                    axes[idx].set_title(title, fontsize=11, fontweight='bold', pad=10)
                    axes[idx].axis('off')
                    
                    # Color-code borders by category
                    category_colors = {
                        'tops': '#3498db',      # Blue
                        'bottoms': '#2ecc71',   # Green  
                        'dresses': '#9b59b6',   # Purple
                        'shoes': '#e67e22',     # Orange
                        'outerwear': '#e74c3c'  # Red
                    }
                    
                    border_color = category_colors.get(category, 'gray')
                    for spine in axes[idx].spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(4)
                        spine.set_visible(True)
                        
                except Exception as e:
                    print(f"Error displaying {item['path']}: {e}")
                    axes[idx].text(0.5, 0.5, f'Error\n{Path(item["path"]).name}',
                                ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].axis('off')
            
            plt.suptitle(f"Outfit #{combo_num} - Overall: {combo['overall_score']:.3f}", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print text breakdown
            print(f"\nItems in this outfit:")
            for category, item in combo['outfit'].items():
                filename = Path(item['path']).name
                score = combo['individual_scores'].get(category, 'N/A')
                if score != 'N/A':
                    print(f"  {category.upper()}: {filename} (match: {score:.3f})")
                else:
                    print(f"  {category.upper()}: {filename}")
    
    def show_categorization_summary(self):
        """
        Show categorization results with BLIP-2 analysis
        """
        print(f"\nClothing Categorization Summary:")
        print("-" * 60)
        
        for category, items in self.categorized_items.items():
            print(f"\n{category.upper()}: {len(items)} items")
            
            for i, item in enumerate(items[:3]):
                filename = Path(item['path']).name
                
                if self.use_blip2 and item['path'] in self.item_metadata:
                    meta = self.item_metadata[item['path']]
                    print(f"  {filename}")
                    print(f"    Style: {meta['style']}")
                    print(f"    Color: {meta['color']}")
                    print(f"    Season: {meta['season']}")
                else:
                    print(f"  {filename}")
            
            if len(items) > 3:
                print(f"  ... and {len(items) - 3} more")
    
    def display_recommendations(self, text_query, top_k=5):
        
        results = self.search_outfits(text_query, top_k)
        
        if not results:
            print("No recommendations found!")
            return
            
        print(f"\n Top {len(results)} matches for: '{text_query}'")
        print("-" * 60)
        
        fig, axes = plt.subplots(1, len(results), figsize=(20, 6))
        
        if len(results) == 1:
            axes = [axes]
            
        for i, (image_path, similarity) in enumerate(results):
            try:
                image = Image.open(image_path)
                axes[i].imshow(image)
                
                filename = Path(image_path).name
                axes[i].set_title(f'{filename}\nMatch: {similarity:.3f}', 
                                fontsize=12, fontweight='bold', pad=10)
                axes[i].axis('off') 
                
                if similarity > 0.25:
                    border_color = 'green'  # Good match
                elif similarity > 0.15:
                    border_color = 'orange'  # Decent match
                else:
                    border_color = 'red'    # Poor match
                
                for spine in axes[i].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(4)
                    spine.set_visible(True)
                    
            except Exception as e:
                print(f"Error displaying {image_path}: {e}")
                axes[i].text(0.5, 0.5, f'Error loading\n{Path(image_path).name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nDetailed Results:")
        for i, (image_path, similarity) in enumerate(results, 1):
            match_quality = "Great" if similarity > 0.25 else "Good" if similarity > 0.15 else "Poor"
            filename = Path(image_path).name
            print(f"{i}. {match_quality} match ({similarity:.3f}) - {filename}")

    def inspect_item(self, image_path):
        """
        Ask custom questions about a specific clothing item
        """
        
        if not self.use_blip2:
            print("BLIP-2 not enabled. Enable with use_blip2=True")
            return
        
        print(f"\nInspecting: {Path(image_path).name}")
        print("Ask questions about this item (or 'done' to finish)")
        
        while True:
            question = input("\nYour question: ")
            
            if question.lower() in ['done', 'quit', 'exit']:
                break
            
            if question.strip():
                answer = self.blip2_analyzer.ask_about_image(image_path, question)
                print(f"Answer: {answer}")

def build_your_own_recommender():
    print("BUILDING SMART FASHION RECOMMENDER WITH BLIP-2")
    print("=" * 60)
    
    # Initialize with BLIP-2
    recommender = FashionRecommender(use_blip2=True)
    
    clothing_folder = "path/to/your/closet/photos"
    
    print(f"\nLoading images from: {clothing_folder}")
    
    for image_file in os.listdir(clothing_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(clothing_folder, image_file)
            recommender.add_clothing_item(image_path)
    
    recommender.show_categorization_summary()
    
    recommender.initialize_outfit_combiner()
    
    print(f"\nTry searching for outfit combinations!")
    
    while True:
        print("\nOptions:")
        print("1. Search for outfit")
        print("2. Inspect specific item")
        print("3. Quit")
        
        choice = input("\nChoice: ")
        
        if choice == "1":
            query = input("Enter search query: ")
            if query.strip():
                recommender.display_outfit_combinations(query, top_k=2)
        
        elif choice == "2":
            print("\nAvailable items:")
            all_items = []
            for items in recommender.categorized_items.values():
                all_items.extend(items)
            
            for i, item in enumerate(all_items[:10]):
                print(f"{i+1}. {Path(item['path']).name}")
            
            idx = int(input("Select item number: ")) - 1
            if 0 <= idx < len(all_items):
                recommender.inspect_item(all_items[idx]['path'])
        
        elif choice == "3":
            break

if __name__ == "__main__":
    build_your_own_recommender()