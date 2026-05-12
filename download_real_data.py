import os
import requests
import pandas as pd
from tqdm import tqdm

def download_isic_data(limit=100):
    print(f"Downloading {limit} images from ISIC Archive...")
    os.makedirs('data/images', exist_ok=True)

    # API endpoint for metadata
    base_url = "https://api.isic-archive.com/api/v2"

    # Map ISIC diagnoses to HAM10000-like labels
    # ISIC has many diagnoses, we'll try to group them

    params = {
        'limit': limit,
        'offset': 0,
        'has_full_metadata': 'true'
    }

    response = requests.get(f"{base_url}/images/", params=params)
    if response.status_code != 200:
        print("Error fetching metadata from ISIC")
        return

    images_metadata = response.json()['results']

    data = []
    for img_meta in tqdm(images_metadata):
        isic_id = img_meta['isic_id']
        diag = img_meta['metadata']['clinical'].get('diagnosis')

        if not diag:
            continue

        # Grouping for simplicity (HAM10000 classes: akiec, bcc, bkl, df, mel, nv, vasc)
        diag = diag.lower()
        label = 'other'
        if 'melanoma' in diag: label = 'mel'
        elif 'nevus' in diag or 'nevi' in diag: label = 'nv'
        elif 'basal cell carcinoma' in diag: label = 'bcc'
        elif 'actinic keratosis' in diag: label = 'akiec'
        elif 'dermatofibroma' in diag: label = 'df'
        elif 'vascular' in diag: label = 'vasc'
        elif 'keratosis' in diag: label = 'bkl'

        if label == 'other':
            continue

        # Download image
        img_url = f"{base_url}/images/{isic_id}/thumbnail" # Thumbnails are smaller and faster
        img_data = requests.get(img_url).content
        with open(f'data/images/{isic_id}.jpg', 'wb') as handler:
            handler.write(img_data)

        data.append({
            'image_id': isic_id,
            'dx': label
        })

    df = pd.DataFrame(data)
    df.to_csv('data/HAM10000_metadata.csv', index=False)
    print(f"Successfully downloaded {len(df)} images and created metadata.")

if __name__ == "__main__":
    try:
        download_isic_data(150)
    except Exception as e:
        print(f"Failed to download from ISIC: {e}")
        print("Falling back to synthetic data...")
        import generate_data
        generate_data.create_synthetic_data()
