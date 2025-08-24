import torch
import clip
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

class FashionRecommender:
    
    def __init__(self):
        print("Loading CLIP model...")
        # Load pre-trained CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"CLIP loaded on {self.device}")
        
        # Storage for clothing items
        self.clothing_database = {}  # {image_path: image_vector}
        
    def add_clothing_item(self, image_path):
        
        print(f"Processing {image_path}...")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get image embedding using CLIP
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Store in database
            self.clothing_database[image_path] = image_features.cpu()
            print(f"Added {Path(image_path).name} to database")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        
    def search_outfits(self, text_query, top_k=5):
        """        
        Args:
            text_query: Description like "casual brunch outfit" 
            top_k: Number of recommendations to return
            
        Returns:
            List of (image_path, similarity_score) tuples
        """
        
        if not self.clothing_database:
            print("No clothing items in database! Add some first.")
            return []
            
        print(f"Searching for: '{text_query}'")
        
        # Encode text query using CLIP
        text_tokens = clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize for cosine similarity  
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarities = []
        
        for image_path, image_features in self.clothing_database.items():
            # Cosine similarity between text and image vectors
            similarity = torch.cosine_similarity(
                text_features.cpu(), 
                image_features, 
                dim=-1
            ).item()
            
            similarities.append((image_path, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def display_recommendations(self, text_query, top_k=5):
        
        # Get recommendations
        results = self.search_outfits(text_query, top_k)
        
        if not results:
            print("No recommendations found!")
            return
            
        print(f"\n Top {len(results)} matches for: '{text_query}'")
        print("-" * 60)
        
        # Create subplot for displaying images
        fig, axes = plt.subplots(1, len(results), figsize=(20, 6))
        
        # Handle single result case
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
            match_quality = "🟢 Great" if similarity > 0.25 else "🟡 Good" if similarity > 0.15 else "🔴 Poor"
            filename = Path(image_path).name
            print(f"{i}. {match_quality} match ({similarity:.3f}) - {filename}")


def build_your_own_recommender():
    
    print("BUILDING YOUR FASHION RECOMMENDER")
    print("=" * 50)
    
    recommender = FashionRecommender()

    clothing_folder = "path/to/your/closet/photos"
    
    print(f"\n Loading images from: {clothing_folder}")
    
    image_count = 0
    for image_file in os.listdir(clothing_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            image_path = os.path.join(clothing_folder, image_file)
            recommender.add_clothing_item(image_path)
            image_count += 1
    
    print(f"\n Successfully loaded {image_count} clothing items!")
    
    if image_count == 0:
        print("No images found! Make sure your images have extensions: .jpg, .png, .jpeg")
        return
    
    '''
    # testing different search queries
    queries = [
        "casual brunch outfit",
        "formal business meeting", 
        "cozy weekend wear",
        "summer beach outfit",
        "elegant date night"
    ]
        
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        recommender.display_recommendations(query, top_k=2)
        
        user_input = input(f"\nPress Enter to try next query, or type 'stop' to quit: ")
        if user_input.lower() == 'stop':
            break
    '''

    print(f"\n🎨 Try customer own queries (i.e. casual summer brunch outfit, formal business meeting, elegent date night, etc!")
    while True:
        custom_query = input(f"\nEnter your search (or 'quit' to exit): ")
        if custom_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if custom_query.strip():
            recommender.display_recommendations(custom_query, top_k=2)


if __name__ == "__main__":
    build_your_own_recommender()