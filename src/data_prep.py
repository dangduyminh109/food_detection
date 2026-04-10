import os
import shutil
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
from tqdm import tqdm
from src import config

def process_data(raw_dir=config.RAW_DIR, processed_dir=config.PROCESSED_DIR, 
                 train_split=0.7, val_split=0.15, test_split=0.15):
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-9
    
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    for split in ['train', 'val', 'test']:
        for label in ['fresh', 'rotten']:
            (processed_path / split / label).mkdir(parents=True, exist_ok=True)
            
    all_data = []
    
    print(f"Scanning raw data in {raw_path}...")
    for class_folder in raw_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        name = class_folder.name.lower()
        if name.startswith('fresh'):
            label = 'fresh'
        elif name.startswith('rotten'):
            label = 'rotten'
        else:
            print(f"Skipping unknown folder: {name}")
            continue
            
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in extensions]
        
        for img_path in images:
            all_data.append((img_path, label))
            
    print(f"Found {len(all_data)} total images.")
    
    random.seed(config.SEED)
    random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    splits = {
        'train': all_data[:n_train],
        'val': all_data[n_train:n_train + n_val],
        'test': all_data[n_train + n_val:]
    }
    
    image_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    for split_name, data_list in splits.items():
        print(f"\nProcessing {split_name} split ({len(data_list)} images)...")
        for img_path, label in tqdm(data_list):
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    
                    target_path = processed_path / split_name / label / img_path.name
                    img.save(target_path, "JPEG", quality=90)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    process_data()
