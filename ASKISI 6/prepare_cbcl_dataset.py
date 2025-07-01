#!/usr/bin/env python
# Download MIT-CBCL Face DB #1, download individual files, and create two folders:
#     faces/      (1 000 random faces)
#     non_faces/  (1 000 random non-faces)
#
# Afterwards you can run face_vs_nonface_full.py without touching anything.

import urllib.request, random, shutil, os, pathlib, tempfile, json

RNG_SEED          = 42
N_SAMPLES_PER_CLS = 1_000

# 1) Where to fetch the data --------------------------------------------------
GITHUB_ROOT = "https://raw.githubusercontent.com/galeone/mlcnn/master/mitcbcl/train/"
GITHUB_API_ROOT = "https://api.github.com/repos/galeone/mlcnn/contents/mitcbcl/train/"

# File patterns - we'll discover available files via GitHub API
FACE_PATTERNS = ["face{:05d}.pgm".format(i) for i in range(1, 3000)]  # More than enough
NONFACE_PATTERNS = ["nonface{:05d}.pgm".format(i) for i in range(1, 5000)]  # More than enough

# 2) Destination structure ----------------------------------------------------
DST_FACE_DIR      = pathlib.Path("faces")
DST_NONFACE_DIR   = pathlib.Path("non_faces")
DST_FACE_DIR.mkdir(exist_ok=True)
DST_NONFACE_DIR.mkdir(exist_ok=True)

def download(url, target):
    """Download a single file from URL to target path"""
    if not target.exists():
        try:
            print(f"→ downloading {target.name}")
            urllib.request.urlretrieve(url, target)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False  # File doesn't exist
            else:
                raise
    else:
        print(f"✔ already have {target.name}")
        return True

def get_available_files(subfolder):
    """Get list of available files from GitHub API"""
    api_url = GITHUB_API_ROOT + subfolder + "/"
    try:
        print(f"→ discovering available {subfolder} files from GitHub API...")
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
        
        # Extract .pgm filenames
        pgm_files = [item['name'] for item in data if item['name'].endswith('.pgm')]
        print(f"   found {len(pgm_files)} {subfolder} images")
        return pgm_files
    except Exception as e:
        print(f"   Error accessing GitHub API: {e}")
        # Fallback: try common naming patterns
        print(f"   Using fallback file discovery for {subfolder}...")
        return None

def download_samples(subfolder, dst_dir):
    """
    Download N_SAMPLES_PER_CLS random *.pgm files from GitHub
    """
    # First, get available files
    available_files = get_available_files(subfolder)
    
    if available_files is None:
        # Fallback: try to download files based on common patterns
        print(f"   Trying fallback method for {subfolder}...")
        if subfolder == "face":
            patterns = [f"face{i:05d}.pgm" for i in range(1, 3000)]
        else:  # non-face
            patterns = [f"nonface{i:05d}.pgm" for i in range(1, 5000)]
        
        # Test which files actually exist
        available_files = []
        for i, pattern in enumerate(patterns[:100]):  # Test first 100
            test_url = GITHUB_ROOT + subfolder + "/" + pattern
            try:
                urllib.request.urlopen(test_url)
                available_files.append(pattern)
            except:
                continue
            if len(available_files) >= N_SAMPLES_PER_CLS:
                break
    
    if len(available_files) < N_SAMPLES_PER_CLS:
        print(f"   Warning: Only found {len(available_files)} {subfolder} files, but need {N_SAMPLES_PER_CLS}")
        sample_size = min(len(available_files), N_SAMPLES_PER_CLS)
    else:
        sample_size = N_SAMPLES_PER_CLS
    
    # Select random sample
    random.seed(RNG_SEED)
    selected_files = random.sample(available_files, sample_size)
    
    # Download selected files
    successful_downloads = 0
    for filename in selected_files:
        file_url = GITHUB_ROOT + subfolder + "/" + filename
        dst_path = dst_dir / filename
        
        if download(file_url, dst_path):
            successful_downloads += 1
        
        if successful_downloads >= N_SAMPLES_PER_CLS:
            break
    
    print(f"   Successfully downloaded {successful_downloads} {subfolder} images")

# ---------------------------------------------------------------------------
def main():
    print("Downloading CBCL Face Dataset from GitHub...")
    print("Getting face images...")
    download_samples("face", DST_FACE_DIR)
    
    print("\nGetting non-face images...")
    download_samples("non-face", DST_NONFACE_DIR)
    
    print(f"\nDone! ✓ Both folders now contain up to {N_SAMPLES_PER_CLS} images each.")

if __name__ == "__main__":
    main()
