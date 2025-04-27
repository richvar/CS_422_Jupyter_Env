#!/usr/bin/env python
# coding: utf-8

# ## ❗ Run this before going through steps, allows Jupyter to actually see installed packages ❗

# In[1]:






# ### 1. Refresh the page
# ### 2. Go to Kernel > Change Kernel
# ### 3. Select Python (StyleGAN2)
# ### 4. Start running through steps

# ## Step 1: Extract dataset images from zip file 

# In[ ]:


import zipfile
import os

# Edit to path of dataset path within a zip file
zip_path = "./post_impressionist_1024_images.zip"
extract_to = "./images"

# Make sure output folder exists
os.makedirs(extract_to, exist_ok=True)

# Unpack zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Extracting images...")
    zip_ref.extractall(extract_to)
    print(f"✅ Done! Extracted to: {extract_to}")


# ## Step 2: Clone the StyleGAN2-ADA repo from Github

# In[3]:



# Print current directory to confirm

# List contents to verify structure


# ## Step 3: Install needed requirements

# In[27]:


import os

# Check current working directory
cwd = os.getcwd()
expected = "stylegan2-ada-pytorch"

if expected not in os.path.basename(cwd):
    print(f"You're in '{cwd}', not inside the '{expected}' folder.")
    print("Use `%cd path/to/stylegan2-ada-pytorch` to navigate there before running this cell.")
else:
    print(f"In correct directory: {cwd}")

    # Manually install dependencies

    # Confirm install
    import torch
    import torchvision
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


# ## Step 4: Convert image folder to TF records

# In[42]:


import os
import shutil
import subprocess
import sys

def convert_images_for_stylegan2ada_pytorch(
    source_dir="../images/full",
    dest_dir="./datasets/post-impressionist",
    force_overwrite=False
):
    # Make sure image source folder exists
    if not os.path.exists(source_dir):
        print(f"Source folder does not exist: {source_dir}")
        return
    if not os.listdir(source_dir):
        print(f"Source folder is empty: {source_dir}")
        return

    # Make sure destination folder exists and is empty
    if os.path.exists(dest_dir):
        if force_overwrite:
            print(f"Deleting existing folder: {dest_dir}")
            shutil.rmtree(dest_dir)
        else:
            print(f"Destination folder must be empty or removed: {dest_dir}")
            return

    os.makedirs(dest_dir, exist_ok=True)
    
    command = [
        "python3", "dataset_tool.py",
        "--source=" + source_dir,
        "--dest=" + dest_dir
    ]

    print(f"Converting image dataset with: {' '.join(command)}")

    env = os.environ.copy()
    env["LC_ALL"] = "C.UTF-8"
    env["LANG"] = "C.UTF-8"
    

    # Verify output format
    if os.path.exists(dest_dir):
        subdirs = [d for d in os.listdir(dest_dir) if d.startswith("000")]
        json_file = os.path.join(dest_dir, "dataset.json")
        if subdirs and os.path.isfile(json_file):
            print(f"Success! Image dataset prepared in '{dest_dir}' with {len(subdirs)} shards.")
        else:
            print(f"Folder structure or metadata file missing in '{dest_dir}'.")
    else:
        print(f"Destination folder not found: {dest_dir}")

        
convert_images_for_stylegan2ada_pytorch(force_overwrite=True)


# ## Step 5: Start training loop for the model

# In[47]:


import os
from datetime import datetime


# Define results and log output directory
log_dir = "./results"
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Create results directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Start training

print(f"Training started in background. Logs will be saved to:\n {log_file}")


# In[ ]:



