import os
import requests
import pandas as pd
from tqdm import tqdm
import time

def download_isic_data(limit=200):
    print(f"Downloading images from ISIC Archive...")
    os.makedirs('data/images', exist_ok=True)

    # API endpoint for images with metadata
    base_url = "https://api.isic-archive.com/api/v2/images"

    # We want images with 'diagnosis' in metadata
    params = {
        'limit': limit,
        'has_full_metadata': 'true'
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        images_metadata = response.json()['results']
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return False

    data = []
    downloaded_count = 0

    for img_meta in tqdm(images_metadata):
        isic_id = img_meta['isic_id']
        meta = img_meta.get('metadata', {}).get('clinical', {})
        diag = meta.get('diagnosis')

        if not diag:
            continue

        diag = diag.lower()
        # Map to HAM10000 classes
        label = None
        if 'melanoma' in diag: label = 'mel'
        elif 'nevus' in diag or 'nevi' in diag: label = 'nv'
        elif 'basal cell' in diag: label = 'bcc'
        elif 'actinic' in diag: label = 'akiec'
        elif 'dermatofibroma' in diag: label = 'df'
        elif 'vascular' in diag: label = 'vasc'
        elif 'keratosis' in diag: label = 'bkl'

        if not label:
            continue

        # Download thumbnail (faster and sufficient for 128x128)
        img_url = f"https://api.isic-archive.com/api/v2/images/{isic_id}/thumbnail"
        try:
            img_data = requests.get(img_url, timeout=20).content
            with open(f'data/images/{isic_id}.jpg', 'wb') as handler:
                handler.write(img_data)

            data.append({
                'image_id': isic_id,
                'dx': label,
                'age': meta.get('age'),
                'sex': meta.get('sex'),
                'localization': meta.get('anatom_site_general')
            })
            downloaded_count += 1
        except Exception as e:
            print(f"Failed to download {isic_id}: {e}")
            continue

    if downloaded_count < 20:
        print(f"Too few images found with valid diagnosis ({downloaded_count}).")
        return False

    df = pd.DataFrame(data)
    df.to_csv('data/HAM10000_metadata.csv', index=False)
    print(f"Successfully downloaded {downloaded_count} real images.")
    return True

if __name__ == "__main__":
    success = download_isic_data(300)
    if not success:
        print("Falling back to high-quality synthetic data...")
        import generate_data
        generate_data.create_synthetic_data(700)
