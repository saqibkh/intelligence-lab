import os
import requests
from datasets import load_dataset
from PIL import Image
from io import BytesIO

# Base this path relative to where the script is located
SCRIPT_DIR = os.path.abspath(__file__) 
BASE_DIR = os.path.dirname(SCRIPT_DIR)

OUT_DIR_IMAGES = os.path.join(BASE_DIR, "data/images")
OUT_DIR_CAPTIONS = os.path.join(BASE_DIR, "data/captions")

DATASET_NAME = "laion/laion400m"
NUM_SAMPLES = 1

# Create output directories
os.makedirs(OUT_DIR_IMAGES, exist_ok=True)
os.makedirs(OUT_DIR_CAPTIONS, exist_ok=True)

# Load dataset metadata
print(f"🔄 Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split=f"train[:{NUM_SAMPLES}]")

# Determine already downloaded files
existing_images = set(os.listdir(OUT_DIR_IMAGES))
existing_indices = {
    int(fname.split("_")[1].split(".")[0])
    for fname in existing_images
    if fname.endswith(".jpg") and fname.startswith("image_")
}

success_count = 0
for idx, sample in enumerate(dataset):
    if success_count in existing_indices:
        print(f"⏭️ Skipping image_{success_count:04d}.jpg (already exists)")
        success_count += 1
        continue

    image_url = sample.get("url")
    caption = sample.get("caption", "").strip()

    if not image_url or not caption:
        continue

    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            continue

        image = Image.open(BytesIO(response.content)).convert("RGB")

        image_filename = f"image_{success_count:04d}.jpg"
        image_path = os.path.join(OUT_DIR_IMAGES, image_filename)
        image.save(image_path)

        caption_filename = f"image_{success_count:04d}.txt"
        caption_path = os.path.join(OUT_DIR_CAPTIONS, caption_filename)
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

        print(f"✅ Saved {image_filename} and caption")
        success_count += 1

    except Exception as e:
        print(f"❌ Skipped {image_url}: {e}")

print(f"\n🎉 Done. Successfully saved or skipped up to {success_count} image-caption pairs.")

