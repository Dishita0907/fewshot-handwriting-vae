import os
from PIL import Image
from pathlib import Path

def preprocess_image(input_path, output_path):
    try:
        image = Image.open(input_path).convert("L")  # Convert to grayscale
        image = image.resize((64, 64))            # Resize
        image.save(output_path)                     # Save to output
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_dataset(language):
    input_dir = Path(f"data/raw/{language}")
    output_dir = Path(f"data/processed/{language}")
    valid_extensions = [".jpg", ".png", ".bmp", ".tiff", ".jpeg"]

    output_dir.mkdir(parents=True, exist_ok=True)
    
    for label_dir in input_dir.iterdir():
        if label_dir.is_dir():
            output_label_dir = output_dir / label_dir.name
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in label_dir.glob("*"):
                if img_file.suffix.lower() in valid_extensions:
                    out_path = output_label_dir / (img_file.stem + ".png")
                    preprocess_image(img_file, out_path)

def run():
    # Process both Hindi and English datasets
    process_dataset("hindi")
    process_dataset("english")

if __name__ == "__main__":
    run()