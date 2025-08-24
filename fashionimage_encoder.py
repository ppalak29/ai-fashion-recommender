import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class FashionImageEncoder(nn.Module):
    """
    CNN that converts clothing images to visual vectors
    
    Architecture comparison:
    Text Encoder: Words → Embeddings → Linear layers → Style vector
    Image Encoder: Pixels → Conv layers → Pooling → Linear layers → Visual vector
    
    """
    
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Layer 1: Basic feature detection (edges, colors)
        # Input: 3 channels (RGB), Output: 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
       
        # Layer 2: More complex features (shapes, textures)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Layer 3: Fashion-specific features (collars, patterns)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layers: Reduce spatial size, keep important features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling: Convert feature maps to single vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear layers
        self.fc1 = nn.Linear(128, 128) 
        self.fc2 = nn.Linear(128, output_dim) 
        
        # Activation functions and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Args:
            x: Image tensor, shape [batch_size, 3, 224, 224]
               - batch_size: how many images at once
               - 3: RGB channels
               - 224×224: image dimensions
               
        Returns:
            visual_vector: Dense vector representing image meaning [batch_size, 256]
        """
        
        batch_size = x.size(0)
            
        # Layer 1: Basic features
        x = self.conv1(x)              
        x = self.relu(x)              
        x = self.pool(x)              
        
        # Layer 2: More complex features  
        x = self.conv2(x)              
        x = self.relu(x)              
        x = self.pool(x)              
        
        # Layer 3: High-level features
        x = self.conv3(x)             
        x = self.relu(x)              
        x = self.pool(x)              
        
        # Global pooling: Feature maps → Single vector
        x = self.global_pool(x)       
        
        # Flatten: Remove spatial dimensions
        x = x.view(batch_size, -1)    
        
        # Linear layers: Same pattern as your text encoder!
        x = self.fc1(x)                
        x = self.relu(x)               
        x = self.dropout(x)            
        
        # Final visual vector
        visual_vector = self.fc2(x)   
        print(f"Final visual vector: {visual_vector.shape}")
        
        return visual_vector


def create_sample_fashion_data():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),        
        transforms.ToTensor(),                
        transforms.Normalize(                
            mean=[0.485, 0.456, 0.406],      
            std=[0.229, 0.224, 0.225]        
        )
    ])
    
    # Load Fashion-MNIST
    dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3), 
            transform
        ])
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader


def train_image_encoder_demo():
    
    print("\n=== IMAGE ENCODER TRAINING DEMO ===")
    
    model = FashionImageEncoder()
    
    try:
        dataloader = create_sample_fashion_data()
        
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("Training setup complete!")
        print("- Model: FashionImageEncoder")
        print("- Optimizer: Adam (same as text encoder)")
        print("- Data: Fashion images")
        
        model.train()
        for i, (images, labels) in enumerate(dataloader):
            if i >= 2: 
                break
                
            print(f"\nBatch {i+1}:")
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            
            visual_vectors = model(images)
            print(f"Visual vectors shape: {visual_vectors.shape}")
            
            # loss = criterion(outputs, labels)
            # optimizer.zero_grad()
            # loss.backward() 
            # optimizer.step()
            
            break 
            
    except Exception as e:
        print(f"Data loading error (expected in demo): {e}")


def demonstrate_visual_similarity():
    
    model = FashionImageEncoder()
    model.eval()
    
    print("Creating visual vectors for different clothing items...")
    
    with torch.no_grad():
        # Simulate casual dress
        casual_dress = torch.randn(1, 3, 224, 224)
        casual_vector = model(casual_dress)
        print(f"Casual dress vector shape: {casual_vector.shape}")
        
        # Simulate formal suit  
        formal_suit = torch.randn(1, 3, 224, 224)
        formal_vector = model(formal_suit)
        print(f"Formal suit vector shape: {formal_vector.shape}")
        
        # Simulate another casual item
        casual_tshirt = torch.randn(1, 3, 224, 224) 
        tshirt_vector = model(casual_tshirt)
        print(f"Casual t-shirt vector shape: {tshirt_vector.shape}")
    
    # Calculate similarities
    cos_sim = nn.CosineSimilarity(dim=1)
    
    sim_dress_tshirt = cos_sim(casual_vector, tshirt_vector)
    sim_dress_suit = cos_sim(casual_vector, formal_vector)
    sim_tshirt_suit = cos_sim(tshirt_vector, formal_vector)
    
    print(f"\nSimilarity Results:")
    print(f"Casual dress ↔ Casual t-shirt: {sim_dress_tshirt.item():.3f}")
    print(f"Casual dress ↔ Formal suit: {sim_dress_suit.item():.3f}")
    print(f"Casual t-shirt ↔ Formal suit: {sim_tshirt_suit.item():.3f}")


def main():
    
    print("🖼️ FASHION IMAGE ENCODER")
    print("=" * 60)
    
    # 4. Show training process
    train_image_encoder_demo()
    
    # 5. Demonstrate similarity matching
    demonstrate_visual_similarity()


if __name__ == "__main__":
    main()