import os
import cv2
import numpy as np

input_root = 'chest-dataset'
output_root = 'chest-dataset-enhanced'
target_size = (224, 224)
valid_extensions = ('.png', '.jpg', '.jpeg')

def enhance_image(image_path):
    print(f"🔍 Reading: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to read: {image_path}")
        return None
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_resized)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_clahe, -1, kernel)
    img_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_GRAY2RGB)
    return img_rgb

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files_processed = 0
    for item in os.listdir(input_folder):
        input_path = os.path.join(input_folder, item)
        output_path = os.path.join(output_folder, item)
        if os.path.isdir(input_path):
            # Recursive call if subfolder found
            files_processed += process_folder(input_path, output_path)
        elif item.lower().endswith(valid_extensions):
            enhanced = enhance_image(input_path)
            if enhanced is not None:
                # Save as PNG (you can keep original extension if you want)
                save_path = os.path.splitext(output_path)[0] + '.png'
                cv2.imwrite(save_path, enhanced)
                files_processed += 1
        else:
            print(f"⚠️ Skipping non-image file: {input_path}")
    return files_processed

total = process_folder(input_root, output_root)
print(f"\n🎉 Image enhancement complete! Total images processed: {total}")
