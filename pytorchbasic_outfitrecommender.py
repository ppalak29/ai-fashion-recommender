import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class OutfitRecommenderPyTorch(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Layer 1: Input (2) -> Hidden (3)
        self.hidden_layer = nn.Linear(2, 3) 
        
        # Layer 2: Hidden (3) -> Output (1) 
        self.output_layer = nn.Linear(3, 1)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        hidden = self.hidden_layer(x)  # Matrix multiplication + bias
        hidden_activation = self.sigmoid(hidden)
        
        output = self.output_layer(hidden_activation)
        final_output = self.sigmoid(output)
        
        return final_output


def create_pytorch_training_data():
    raw_data = [
        (20, 9, 0.9), (30, 8, 0.8), (25, 7, 0.75),
        (85, 2, 0.1), (90, 1, 0.05), (80, 3, 0.2),
        (60, 5, 0.5), (70, 6, 0.6), (40, 4, 0.4),
        (90, 9, 0.8), (20, 2, 0.3)
    ]
    
    inputs = []
    targets = []
    
    for temp, formality, target in raw_data:
        # Normalize inputs
        normalized_input = [temp / 100.0, formality / 10.0]
        inputs.append(normalized_input)
        targets.append([target])  # PyTorch expects lists for targets
    
    # Convert to tensors
    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    
    return X, y

def train_pytorch_network():
    
    model = OutfitRecommenderPyTorch()
    
    X, y = create_pytorch_training_data()
    
    criterion = nn.MSELoss()  # Mean Squared Error
    
    # Optimizer (replaces your manual weight updates)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    print("=== PYTORCH NEURAL NETWORK ===\n")
    
    # Test before training
    print("--- BEFORE TRAINING ---")
    model.eval()
    test_cases = torch.tensor([[0.3, 0.8], [0.75, 0.3], [0.5, 0.6]], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(test_cases)
        for i, pred in enumerate(predictions):
            print(f"Test case {i+1}: {pred.item():.3f}")
    
    print("\n--- TRAINING ---")
    model.train()
    
    epochs = 500
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        # 3. Backpropagation 
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update weights
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("\n--- AFTER TRAINING ---")
    model.eval()
    with torch.no_grad():
        predictions = model(test_cases)
        for i, pred in enumerate(predictions):
            print(f"Test case {i+1}: {pred.item():.3f}")
    
    return model


def visualize_pytorch_vs_handcoded():
    pytorch_model = train_pytorch_network()
    
    temps = np.linspace(0, 100, 50)
    formalities = np.linspace(1, 10, 50)
    
    temp_grid, formal_grid = np.meshgrid(temps, formalities)
    
    # Prepare data for PyTorch
    test_inputs = []
    for i in range(len(temps)):
        for j in range(len(formalities)):
            test_inputs.append([temps[i]/100.0, formalities[j]/10.0])
    
    test_tensor = torch.tensor(test_inputs, dtype=torch.float32)
    
    # Get predictions
    pytorch_model.eval()
    with torch.no_grad():
        predictions = pytorch_model(test_tensor).numpy()
    
    # Reshape for plotting
    pred_grid = predictions.reshape(len(formalities), len(temps))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(temp_grid, formal_grid, pred_grid, levels=20, cmap='RdYlBu_r')
    plt.colorbar(label='Formality Score')
    plt.xlabel('Temperature (°F)')
    plt.ylabel('Event Formality (1-10)')
    plt.title('PyTorch Neural Network: Outfit Formality Recommendations')
    plt.show()

if __name__ == "__main__":
    train_pytorch_network()
    visualize_pytorch_vs_handcoded()