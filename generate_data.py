import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random

def create_synthetic_data(num_samples=700):
    os.makedirs('data/images', exist_ok=True)

    classes = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }

    class_list = list(classes.keys())
    data = []

    for i in range(num_samples):
        img_id = f'ISIC_{i:07d}'
        label = random.choice(class_list)

        # Create a synthetic image
        img = Image.new('RGB', (128, 128), color=(255, 224, 189)) # Skin tone
        draw = ImageDraw.Draw(img)

        # Draw something based on label to make it learnable
        # Using much more distinct shapes and colors
        if label == 'akiec':
            draw.ellipse([10, 10, 110, 30], fill=(255, 0, 0)) # Red top bar
        elif label == 'bcc':
            draw.rectangle([10, 10, 30, 110], fill=(0, 255, 0)) # Green left bar
        elif label == 'bkl':
            draw.rectangle([100, 10, 120, 110], fill=(0, 0, 255)) # Blue right bar
        elif label == 'df':
            draw.ellipse([10, 100, 110, 120], fill=(255, 255, 0)) # Yellow bottom bar
        elif label == 'mel':
            draw.rectangle([40, 40, 80, 80], fill=(255, 0, 255)) # Magenta center square
        elif label == 'nv':
            draw.ellipse([40, 40, 80, 80], fill=(0, 255, 255)) # Cyan center circle
        elif label == 'vasc':
            draw.line([0, 0, 128, 128], fill=(0, 0, 0), width=15) # Black diagonal

        img.save(f'data/images/{img_id}.jpg')

        data.append({
            'lesion_id': f'HAM_{i:07d}',
            'image_id': img_id,
            'dx': label,
            'dx_type': 'consensus',
            'age': random.randint(10, 80),
            'sex': random.choice(['male', 'female']),
            'localization': random.choice(['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 'face', 'chest'])
        })

    df = pd.DataFrame(data)
    df.to_csv('data/HAM10000_metadata.csv', index=False)
    print(f"Generated {num_samples} synthetic images and metadata.")

if __name__ == "__main__":
    create_synthetic_data()
