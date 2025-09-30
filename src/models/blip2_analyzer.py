from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from pathlib import Path

class BLIP2Analyzer:
    """Handles BLIP-2 model operations"""
    
    def __init__(self):
        print("Loading BLIP-2 model...")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"BLIP-2 loaded on {self.device}")
    
    def ask_about_image(self, image_path, question):
        """Ask BLIP-2 a question about an image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.7
            )
        
        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return answer.strip()
    
    def categorize_item(self, image_path):
        """Categorize clothing item"""
        print(f"Analyzing {Path(image_path).name} with BLIP-2...")
        
        questions = {
            'category': "What type of clothing item is this? Answer with one word like: shirt, pants, dress, shoes, jacket, skirt, sweater, shorts, or coat.",
            'style': "Is this clothing item formal, casual, or sporty?",
            'color': "What is the primary color of this clothing item?",
            'season': "What season is this clothing appropriate for?"
        }
        
        analysis = {}
        for key, question in questions.items():
            answer = self.ask_about_image(image_path, question)
            analysis[key] = answer
            print(f"  {key}: {answer}")
        
        category = self._normalize_category(analysis['category'].lower())
        
        return {
            'category': category,
            'raw_category': analysis['category'],
            'style': analysis['style'],
            'color': analysis['color'],
            'season': analysis['season'],
            'confidence': 0.8
        }
    
    def _normalize_category(self, raw_category):
        """Map BLIP-2 responses to standard categories"""
        mapping = {
            'shirt': 'tops', 'blouse': 'tops', 't-shirt': 'tops', 'tshirt': 'tops',
            'top': 'tops', 'sweater': 'tops', 'tank': 'tops',
            'pants': 'bottoms', 'jeans': 'bottoms', 'trousers': 'bottoms',
            'shorts': 'bottoms', 'skirt': 'bottoms',
            'dress': 'dresses', 'gown': 'dresses',
            'shoes': 'shoes', 'sneakers': 'shoes', 'boots': 'shoes',
            'sandals': 'shoes', 'heels': 'shoes',
            'jacket': 'outerwear', 'coat': 'outerwear', 'cardigan': 'outerwear',
            'blazer': 'outerwear'
        }
        
        for keyword, standard in mapping.items():
            if keyword in raw_category:
                return standard
        return 'other'