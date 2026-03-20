#!/usr/bin/env python3
"""
MLflow-instrumented FashionMNIST Classifier Training Script
Assignment 3: Observable ML with MLflow

This script trains a neural network on FashionMNIST and logs all metrics, 
parameters, and artifacts to MLflow for experiment tracking and comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import os
import sys


# ============ Configuration ============
FASHION_MNIST_DIR = './data'
STUDENT_ID = "202202184"

# Default hyperparameters (can be overridden via command line)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5
MOMENTUM = 0.9


# ============ Device Setup ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ Data Loading ============
def load_data(batch_size):
    """Load FashionMNIST training and test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # PreComputed FashionMNIST stats
    ])
    
    train_dataset = datasets.FashionMNIST(
        root=FASHION_MNIST_DIR, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=FASHION_MNIST_DIR, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============ Model Definition ============
class FashionMNISTNet(nn.Module):
    """Neural network classifier for FashionMNIST."""
    
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# ============ Training Loop ============
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


# ============ Main Training Function ============
def main(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, 
         epochs=EPOCHS, momentum=MOMENTUM):
    """Main training and logging function."""
    
    # Set MLflow experiment
    mlflow.set_experiment("Assignment3_OmarMohamed")
    
    with mlflow.start_run():
        # ===== Log Parameters =====
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("momentum", momentum)
        
        # ===== Log Tags =====
        mlflow.set_tag("student_id", STUDENT_ID)
        mlflow.set_tag("model_type", "FeedForward_NN")
        mlflow.set_tag("dataset", "FashionMNIST")
        
        print("=" * 60)
        print(f"Starting training with LR={learning_rate}, BS={batch_size}, Epochs={epochs}")
        print("=" * 60)
        
        # Load data
        train_loader, test_loader = load_data(batch_size)
        
        # Initialize model, loss, optimizer
        model = FashionMNISTNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
        # Training loop with live logging
        best_test_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Live logging to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
            
            # Track best accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        
        # ===== Log Final Metrics =====
        mlflow.log_metric("best_test_accuracy", best_test_acc)
        
        # ===== Save Model with MLflow =====
        model_dir = f"./mlruns/{mlflow.get_experiment_by_name('Assignment3_Observable_ML').experiment_id}/models"
        os.makedirs(model_dir, exist_ok=True)
        
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="FashionMNIST_Classifier"
        )
        
        # Save model weights as artifact
        model_path = os.path.join(model_dir, f"model_lr{learning_rate}_bs{batch_size}.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        print("\n" + "=" * 60)
        print(f"✅ Training complete! Best Test Accuracy: {best_test_acc:.2f}%")
        print(f"📊 MLflow Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)
        
        return best_test_acc


if __name__ == "__main__":
    # Allow command-line overrides
    if len(sys.argv) > 1:
        learning_rate = float(sys.argv[1])
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else BATCH_SIZE
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS
        momentum = float(sys.argv[4]) if len(sys.argv) > 4 else MOMENTUM
        main(learning_rate, batch_size, epochs, momentum)
    else:
        main()
