import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

class FashionTextEncoder(nn.Module):
    """
    Text encoder that creates style embeddings
    
    Architecture:
    Input: Token IDs [2, 12] → Embeddings → Neural Network → Style Vector
    
    Output: creates a dense vector that captures the MEANING of text,
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, output_dim=256):
        super().__init__()
        
        # Word embedding layer: token IDs → dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Neural network layers (with proper activations!)
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Final embedding size
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Initialize embeddings randomly
        nn.init.normal_(self.embedding.weight, std=0.1)
    
    def forward(self, token_ids):
        """        
        Args:
            token_ids: [batch_size, sequence_length] - tensor of token IDs
            
        Returns:
            style_embedding: [batch_size, output_dim] - dense style vector
        """
        
        embedded = self.embedding(token_ids) 
        
        # combines variable-length sequences
        pooled = embedded.mean(dim=1) 
        
        x = self.layer1(pooled)
        x = self.relu(x)                
        x = self.dropout(x)            
        
        x = self.layer2(x) 
        x = self.relu(x)                
        x = self.dropout(x)             
        
        style_embedding = self.output_layer(x)
        
        return style_embedding


class SimpleTokenizer:
    
    def __init__(self):
        # Fashion vocabulary
        self.word_to_id = {
            '<PAD>': 0,   # Padding
            '<UNK>': 1,   # Unknown words
            
            # Style descriptors
            'casual': 2, 'formal': 3, 'business': 4, 'elegant': 5,
            'relaxed': 6, 'professional': 7, 'chic': 8, 'trendy': 9,
            'smart': 10, 'comfortable': 11, 'stylish': 12, 'modern': 13,
            
            # Occasions
            'work': 14, 'meeting': 15, 'brunch': 16, 'dinner': 17,
            'party': 18, 'wedding': 19, 'date': 20, 'vacation': 21,
            'beach': 22, 'office': 23, 'weekend': 24, 'evening': 25,
            
            # Clothing
            'dress': 26, 'shirt': 27, 'pants': 28, 'skirt': 29,
            'jacket': 30, 'outfit': 31, 'top': 32, 'blouse': 33,
            
            # Seasons
            'summer': 34, 'winter': 35, 'spring': 36, 'fall': 37,
            
            # Common words
            'for': 38, 'and': 39, 'with': 40, 'look': 41, 'wear': 42
        }
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def tokenize(self, text):
        
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        
        # Convert words to IDs, use <UNK> for unknown words
        token_ids = []
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id['<UNK>'])
            token_ids.append(token_id)
            
            if token_id == self.word_to_id['<UNK>']:
                print(f"Unknown word: '{word}' → <UNK> (ID: 1)")
        
        return token_ids
    
    def decode(self, token_ids):
        words = [self.id_to_word.get(id, '<UNK>') for id in token_ids]
        return ' '.join(words)


def create_style_similarity_data():
    
    # Positive pairs (similar styles)
    positive_pairs = [
        ("casual brunch", "relaxed weekend"),
        ("formal meeting", "business dinner"), 
        ("beach vacation", "summer casual"),
        ("elegant dinner", "formal evening"),
        ("office wear", "professional outfit"),
        ("weekend look", "comfortable casual"),
    ]
    
    # Negative pairs (different styles)
    negative_pairs = [
        ("casual brunch", "formal meeting"),
        ("beach vacation", "business dinner"),
        ("relaxed weekend", "elegant dinner"),
        ("summer casual", "professional outfit"),
    ]
    
    return positive_pairs, negative_pairs


def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
    
    # Loss function:
    # - If similar (label=1): minimize distance
    # - If different (label=0): maximize distance (up to margin)
    loss = label * torch.pow(distance, 2) + \
           (1 - label) * torch.pow(torch.clamp(margin - distance, min=0), 2)
    
    return loss.mean()


def train_style_encoder():
    
    print("=== TRAINING STYLE ENCODER ===\n")
    
    tokenizer = SimpleTokenizer()
    model = FashionTextEncoder(vocab_size=tokenizer.vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Get training data
    positive_pairs, negative_pairs = create_style_similarity_data()
    
    epochs = 300
    for epoch in range(epochs):
        total_loss = 0
        
        for text1, text2 in positive_pairs:
            # Tokenize texts
            tokens1 = tokenizer.tokenize(text1)
            tokens2 = tokenizer.tokenize(text2)
            
            # Pad to same length
            max_len = max(len(tokens1), len(tokens2))
            tokens1 += [0] * (max_len - len(tokens1)) 
            tokens2 += [0] * (max_len - len(tokens2))
            
            # Convert to tensors
            input1 = torch.tensor([tokens1], dtype=torch.long)
            input2 = torch.tensor([tokens2], dtype=torch.long)
            
            # Get embeddings
            embedding1 = model(input1)
            embedding2 = model(input2)
            
            # Calculate loss (label=1 for similar pairs)
            loss = contrastive_loss(embedding1, embedding2, 
                                  torch.tensor([1.0]), margin=1.0)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Train on negative pairs
        for text1, text2 in negative_pairs:
            tokens1 = tokenizer.tokenize(text1)
            tokens2 = tokenizer.tokenize(text2)
            
            max_len = max(len(tokens1), len(tokens2))
            tokens1 += [0] * (max_len - len(tokens1))
            tokens2 += [0] * (max_len - len(tokens2))
            
            input1 = torch.tensor([tokens1], dtype=torch.long)
            input2 = torch.tensor([tokens2], dtype=torch.long)
            
            embedding1 = model(input1)
            embedding2 = model(input2)
            
            # Calculate loss (label=0 for different pairs)
            loss = contrastive_loss(embedding1, embedding2,
                                  torch.tensor([0.0]), margin=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            avg_loss = total_loss / (len(positive_pairs) + len(negative_pairs))
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    print("Training complete!")
    return model, tokenizer


def test_embeddings(model, tokenizer):
    
    print("\n=== TESTING STYLE EMBEDDINGS ===")
    
    test_descriptions = [
        "casual brunch",
        "relaxed weekend", 
        "formal meeting",
        "business dinner",
        "beach vacation"
    ]
    
    model.eval()
    embeddings = {}
    
    with torch.no_grad():
        for desc in test_descriptions:
            tokens = tokenizer.tokenize(desc)
            padded = tokens + [0] * (10 - len(tokens)) 
            input_tensor = torch.tensor([padded], dtype=torch.long)
            
            embedding = model(input_tensor)
            embeddings[desc] = embedding.squeeze() 
    
    print("\nSimilarity Matrix (cosine similarity):")
    print("-" * 60)
    
    for desc1 in test_descriptions:
        similarities = []
        for desc2 in test_descriptions:
            # Calculate cosine similarity
            emb1 = embeddings[desc1]
            emb2 = embeddings[desc2]
            similarity = torch.cosine_similarity(emb1, emb2, dim=0)
            similarities.append(f"{similarity.item():.2f}")
        
        print(f"{desc1:15} | {' | '.join(similarities)}")


def main():
    print("\n--- Testing Tokenizer ---")
    tokenizer = SimpleTokenizer()
    
    test_texts = ["casual brunch outfit", "formal business meeting"]
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"'{text}' → {tokens} → '{tokenizer.decode(tokens)}'")
    
    # Train the model
    print("\n--- Training Text Encoder ---")
    model, tokenizer = train_style_encoder()
    
    # Test embeddings
    test_embeddings(model, tokenizer)
    
    return model, tokenizer

if __name__ == "__main__":
    main()