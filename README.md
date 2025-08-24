# AI Fashion Recommender

An intelligent fashion recommendation system that uses OpenAI's CLIP model to match clothing items with natural language descriptions. Simply describe what you want to wear, and the AI will find the best matching outfits from your wardrobe!

## Features

- **Natural Language Search**: Search with phrases like "casual brunch outfit" or "formal business meeting"
- **Visual Results**: See your top matching clothing items with similarity scores
- **Smart Matching**: Uses CLIP's advanced image-text understanding trained on 400M+ examples
- **Easy Setup**: Just add photos of your clothes and start searching!

## ML Design

The system combines two powerful AI components:

1. **Text Encoder**: Converts your search query into a "style vector" that captures meaning
2. **Image Encoder**: Converts clothing photos into "visual vectors" that represent their appearance
3. **Similarity Matching**: Finds images whose vectors are most similar to your query vector

Built on OpenAI's CLIP (Contrastive Language-Image Pre-training) model.

## Prerequisites
- Python 3.7+
- PyTorch
- CLIP
- PIL (Pillow)
- matplotlib
- numpy

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-fashion-recommender.git
cd ai-fashion-recommender
```

2. **Install Dependencies**
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install pillow matplotlib numpy pathlib
```

2. **Prepare Clothing Photos**
```bash
1. Create a folder and set to clothing_folder (e.g., closet/) 
2. Add photos of your clothing items
3. Supported formats: .jpg, .png, .jpeg
For best results: good lighting, clean background, single item per photo
```

## Understanding Results
### Similarity Scores
> 0.25: 🟢 Great match
0.15 - 0.25: 🟡 Good match
< 0.15: 🔴 Poor match

### Visual Display
Results shown as images with similarity scores
Color-coded borders indicate match quality
Ranked from highest to lowest similarity