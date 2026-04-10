"""Evaluation script for fruit freshness classification."""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from . import config
from .data_loader import get_dataloaders
from .model import get_model

def calculate_accuracy(outputs, targets):
    # Lấy class có xác suất cao nhất
    _, preds = torch.max(outputs, 1)
    return (preds == targets).sum().item() / targets.size(0)

def evaluate_model(checkpoint_path=None):
    device = torch.device(config.DEVICE)

    # Nếu không có checkpoint thì dùng model tốt nhất
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        
    # Load dữ liệu test
    _, _, test_loader, _ = get_dataloaders()
    
    # Khởi tạo model
    model = get_model().to(device)
    
    # Load trọng số đã train
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # chế độ đánh giá
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Không tính gradient khi evaluate
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Cộng dồn loss theo batch
            test_loss += loss.item() * images.size(0)

            # Lấy dự đoán
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    print(f"\nTest Loss: {test_loss / total:.4f}")
    print(f"Test Accuracy: {correct / total:.4f}")
    return correct / total

if __name__ == "__main__":
    import os
    evaluate_model()