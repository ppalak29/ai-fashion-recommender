import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class FashionClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        # Input: 28×28 = 784 pixels
        self.layer1 = nn.Linear(784, 128)
        
        # Hidden layer 2: 64 neurons  
        self.layer2 = nn.Linear(128, 64)
        
        # Output: 10 clothing categories
        self.layer3 = nn.Linear(64, 10)
        
        self.relu = nn.ReLU() 
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        
        # Flatten image from 28×28 to 784×1 vector
        x = x.view(x.size(0), -1)  # Reshape: (batch_size, 28, 28) → (batch_size, 784)
        
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2 + activation  
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation - we'll apply softmax later)
        x = self.layer3(x)
        
        return x


def load_fashion_data():
    """
    Load the Fashion-MNIST dataset 
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform  
    )
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader


def visualize_fashion_data(dataloader):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i//4, i%4]
        # Convert from tensor to numpy and reshape
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{classes[labels[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Image shape: {images[0].shape}")  # Should be [1, 28, 28]
    print(f"Pixel value range: {images[0].min():.2f} to {images[0].max():.2f}")


def train_fashion_classifier():
        
    trainloader, testloader = load_fashion_data()
    
    model = FashionClassifier()
    
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam is like SGD but smarter
    
    epochs = 5  # Fewer epochs since this is a bigger dataset
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):   
            outputs = model(images)
            loss = criterion(outputs, labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/100:.4f}')
                running_loss = 0.0
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} Accuracy: {accuracy:.2f}%')
    
    print("Training complete!")
    return model


def test_classifier(model, testloader):

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model.eval()
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class stats
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'\nOverall Test Accuracy: {100 * correct / total:.2f}%\n')
    
    for i in range(10):
        if class_total[i] > 0:
            print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')


def demonstrate_prediction(model, testloader):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Show first 4 predictions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(4):
        ax = axes[i//2, i%2]
        
        # Show image
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        
        # Show prediction
        actual = classes[labels[i]]
        predicted_class = classes[predicted[i]]
        confidence = probabilities[i][predicted[i]] * 100
        
        title = f'Actual: {actual}\nPredicted: {predicted_class}\nConfidence: {confidence:.1f}%'
        ax.set_title(title)
        ax.axis('off')
        
        # Color code: green if correct, red if wrong
        if labels[i] == predicted[i]:
            ax.title.set_color('green')
        else:
            ax.title.set_color('red')
    
    plt.tight_layout()
    plt.show()

def main():
    
    # Load and visualize data
    print("\n--- Loading Fashion-MNIST Dataset ---")
    trainloader, testloader = load_fashion_data()
    visualize_fashion_data(trainloader)
    
    # Train the model
    print("\n--- Training the Classifier ---")
    model = train_fashion_classifier()
    
    # Test the model
    print("\n--- Testing the Classifier ---")
    test_classifier(model, testloader)
    
    # Show individual predictions
    print("\n--- Individual Predictions ---")
    demonstrate_prediction(model, testloader)
   
if __name__ == "__main__":
    main()