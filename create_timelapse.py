import cv2
import os
import glob
import re

# --- Configuration ---
INPUT_DIR = 'aligned'
OUTPUT_FILE = 'timelapse.mp4'
FPS = 30
EXTENSIONS = ['*.jpg', '*.png', '*.jpeg']

def numerical_sort(value):
    """
    Helper to sort filenames numerically if they contain numbers, 
    otherwise falls back to string sort.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    # 1. Get Images
    images = []
    for ext in EXTENSIONS:
        images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    # Sort carefully to ensure time consistency
    # Assuming filenames are like "01 02 2023.jpg" etc. 
    # Standard string sort works for "MM DD YYYY" if YYYY is at end, wait...
    # "01 02 2023" vs "01 03 2023" works.
    # "01 02 2023" vs "02 01 2023" works.
    # But "01 01 2023" vs "01 01 2024"? 
    # If the user's format is MM DD YYYY, then standard sort fails across years (01 01 2024 comes before 02 01 2023).
    # Since I see all 2023 in the file list, simple sort might work, but let's be safe(r).
    # Actually, the user's files seem to be "Day Month Year" or "Month Day Year"?
    # "01 02 2023.jpg" -> Jan 2nd? or Feb 1st?
    # Given "01 02", "01 03", ..., "01 09", "02 01"... it looks like Month-Day-Year or Day-Month-Year.
    # Let's rely on standard sort for now as the user didn't specify format, 
    # and standard sort is usually what file explorers do.
    images.sort()
    
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(images)} images.")
    
    # 2. Setup Video Writer
    first_frame = cv2.imread(images[0])
    height, width, layers = first_frame.shape
    size = (width, height)
    
    print(f"Video Resolution: {width}x{height}, FPS: {FPS}")
    
    # mp4v is a safe generic option for MP4 container
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, size)

    # 3. Write Frames
    count = 0
    for img_path in images:
        img = cv2.imread(img_path)
        
        # Ensure size matches (just in case)
        if (img.shape[1] != width) or (img.shape[0] != height):
            print(f"Warning: Resizing {os.path.basename(img_path)} to fit video.")
            img = cv2.resize(img, size)
            
        out.write(img)
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(images)} frames...", end='\r')

    out.release()
    print(f"\nDone! Video saved to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
