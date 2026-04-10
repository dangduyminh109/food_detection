"""Inference script for single image prediction."""
import torch
from PIL import Image
from torchvision import transforms
from . import config
from .model import get_model
import os

def predict(image_path, checkpoint_path=None):
    # Xác định thiết bị chạy (CPU hoặc GPU) từ config
    device = torch.device(config.DEVICE)

    # Nếu không truyền checkpoint thì dùng model tốt nhất đã lưu
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    # ===== Load model =====
    model = get_model().to(device)

    # Load checkpoint (trạng thái đã train)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Nạp trọng số vào model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Chuyển model sang chế độ inference (tắt dropout, batchnorm dùng thống kê cố định)
    model.eval()
    
    # Lấy tên class (nếu không có thì mặc định 2 lớp)
    class_names = checkpoint.get('class_names', ['fresh', 'rotten'])
    
    # ===== Tiền xử lý ảnh =====
    transform = transforms.Compose([
        # Resize ảnh về kích thước model yêu cầu
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),

        # Chuyển ảnh sang tensor (HWC → CHW, giá trị [0,255] → [0,1])
        transforms.ToTensor(),

        # Chuẩn hóa theo mean/std của ImageNet (giúp model học tốt hơn)
        transforms.Normalize(
            [0.485, 0.456, 0.406],  # mean
            [0.229, 0.224, 0.225]   # std
        )
    ])
    
    # Mở ảnh và đảm bảo là RGB (tránh lỗi ảnh grayscale)
    image = Image.open(image_path).convert("RGB")

    # Áp dụng transform + thêm batch dimension (1, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # ===== Inference =====
    with torch.no_grad():  # Tắt tính gradient để giảm bộ nhớ + tăng tốc
        outputs = model(image_tensor)

        probs = torch.softmax(outputs, dim=1)

        conf, preds = torch.max(probs, 1)
        # conf: xác suất cao nhất
        # preds: index của class
    
    # Chuyển index → tên class
    result = class_names[preds.item()]

    # Lấy độ tin cậy dạng float
    confidence = conf.item()
    
    print(f"Prediction: {result} ({confidence:.2f})")
    return result, confidence


if __name__ == "__main__":
    # Ví dụ sử dụng:
    # predict("path/to/image.jpg")
    pass