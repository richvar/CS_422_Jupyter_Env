import zipfile
import os

# Edit to path of dataset path within a zip file
zip_path = "./../richard/512_basic_training/512images.zip"
extract_to = "./images"

# Make sure output folder exists
os.makedirs(extract_to, exist_ok=True)

# Unpack zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Extracting images...")
    zip_ref.extractall(extract_to)
    print(f"✅ Done! Extracted to: {extract_to}")