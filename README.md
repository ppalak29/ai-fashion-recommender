# AI Fashion Recommender

An intelligent fashion recommendation system powered by CLIP and BLIP-2 that helps you find perfect outfit combinations from your wardrobe using natural language queries.

## Features

- **Natural Language Search**: Search with phrases like "casual brunch outfit" or "formal business meeting"
- **AI-Powered Analysis**: Uses OpenAI's CLIP and Salesforce's BLIP-2 for advanced image-text understanding
- **Complete Outfit Combinations**: Recommends coordinated outfits (top + bottom + shoes)
- **Smart Categorization**: Automatically categorizes clothing items using BLIP-2
- **Style Compatibility**: Analyzes how well items work together based on style, formality, and color
- **Rich Metadata**: Extracts style, color, season, and formality information for each item
- **Web Interface**: Easy-to-use Streamlit UI for browsing and searching

## How It Works
The system uses two complementary AI models:

1. **CLIP (Contrastive Language-Image Pre-training)**:
   - Fast similarity matching between text queries and images
   - Encodes both text and images into a shared vector space
   - Enables zero-shot outfit search

2. **BLIP-2 (Bootstrapping Language-Image Pre-training)**:
   - Dynamic clothing categorization through natural language
   - Extracts detailed metadata (style, color, season)
   - Can answer open-ended questions about clothing items

## Architecture
Text Query: "casual brunch outfit"
↓
CLIP Text Encoder → Style Vector
↓
Similarity Search → Find matching items
↓
BLIP-2 Analysis → Categorize & analyze items
↓
Outfit Combiner → Create coordinated outfits
↓
Compatibility Scoring → Rank by fit
↓
Display Results → Complete outfit recommendations

## Installation
### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster inference)

### Setup
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-fashion-recommender.git
cd outfit-recommender
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
# Note: First run will download model weights (~5GB for CLIP + BLIP-2)
```

## Usage
### Web Interface (Recommended)

1. **Start the Streamlit app**
```bash
streamlit run fashion_app.py
```

2. **Upload your wardrobe**
- Enter folder path containing clothing photos, OR
- Upload images directly through the interface

3. **Search for outfits**
- Enter occasions like "casual weekend", "business meeting", "date night"
- View complete outfit combinations with compatibility scores

### Python API
```bash
from src.models.fashion_recommender import FashionRecommender

# Initialize recommender
recommender = FashionRecommender(use_blip2=True)

# Add clothing items
recommender.add_clothing_item("path/to/shirt.jpg")
recommender.add_clothing_item("path/to/jeans.jpg")
recommender.add_clothing_item("path/to/shoes.jpg")

# Initialize outfit combiner
recommender.initialize_outfit_combiner()

# Search for outfit combinations
results = recommender.outfit_combiner.create_outfit_combinations(
    "casual brunch outfit",
    top_k=3
)

# Display results
for outfit in results:
    print(f"Score: {outfit['overall_score']:.2f}")
    for category, item in outfit['outfit'].items():
        print(f"  {category}: {item['path']}")
```

## Understanding the Scores
### Individual Match Score (0-1)
How well each item matches your search query
    > 0.25: Excellent match
    0.15 - 0.25: Good match
    < 0.15: Poor match

### Compatibility Score (0-1)
How well items work together in an outfit
    Analyzes style consistency
    Considers formality levels
    Evaluates visual harmony

### Overall Score (0-1)
Combined metric: (Individual Match × 0.6) + (Compatibility × 0.4)

## Limitations
Speed: BLIP-2 analysis takes ~2-5 seconds per item on first load
GPU: Recommended for real-time performance
Model Size: ~5GB of model weights need to be downloaded
Color Matching: Basic compatibility; doesn't use advanced color theory
Fashion Rules: General style matching; doesn't know specific fashion rules

## Technical Details
### Models Used
- CLIP ViT-B/32: 151M parameters, 512-dim embeddings
- BLIP-2 OPT-2.7B: 2.7B parameters, vision-language generation

### Performance
- CLIP encoding: ~50ms per image (GPU)
- BLIP-2 analysis: ~2s per image (GPU)
- Outfit combination: ~100ms for 50 items (CPU)

### Hardware Requirements
Minimum: 8GB RAM, CPU only (slow)
Recommended: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
Optimal: 32GB RAM, NVIDIA GPU with 12GB+ VRAM

## Acknowledgments
OpenAI for CLIP
Salesforce for BLIP-2
Hugging Face for model hosting