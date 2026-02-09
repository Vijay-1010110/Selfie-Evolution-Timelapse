import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import time

# --- Configuration ---
REFERENCE_IMAGE_PATH = 'reference.png'
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'aligned'
LOG_FILE = 'alignment.log'
EXTENSIONS = ['*.jpg', '*.png', '*.jpeg']

# --- Global objects ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_landmarks(image_path):
    """
    Detects landmarks using MediaPipe Face Mesh.
    Returns a list of (x, y) tuples for the 468 landmarks.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return None, None

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"Error: No face detected in {image_path}")
        return None, image

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = []
    for lm in face_landmarks.landmark:
        pt = (int(lm.x * w), int(lm.y * h))
        landmarks.append(pt)
    
    return landmarks, image

def get_boundary_points(w, h, div=2):
    """
    Returns points along the boundary of the image (corners + midpoints).
    div=2 adds midpoints.
    """
    points = []
    # Corners
    points.append((0, 0))
    points.append((w - 1, 0))
    points.append((w - 1, h - 1))
    points.append((0, h - 1))
    
    # Midpoints
    points.append((w // 2, 0))
    points.append((w - 1, h // 2))
    points.append((w // 2, h - 1))
    points.append((0, h // 2))
    
    return points

def get_delaunay_triangles(rect, landmarks):
    """
    Compute Delaunay triangles based on reference landmarks.
    Returns list of triangle indices (pt1_idx, pt2_idx, pt3_idx).
    """
    subdiv = cv2.Subdiv2D(rect)
    
    # Map (x, y) to index for easy lookup later
    # We expect duplicates might exist if boundary points overlap with face (unlikely but safe to handle)
    # Using a list and index mapping is safer than dict if points are very close.
    # But for strict indexing, we just insert sequentially.
    
    for pt in landmarks:
        subdiv.insert(pt)
        
    triangle_list = subdiv.getTriangleList()
    delaunay_tri_indices = []
    
    # We need to map coordinates back to indices. 
    # Since Subdiv2D can slightly alter float coords, we use a proximity check.
    
    landmark_np = np.array(landmarks)
    
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        
        for pt in pts:
            # Find closest landmark
            dists = np.linalg.norm(landmark_np - pt, axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < 1.0: # Threshold for float errors
                indices.append(min_idx)
                
        if len(indices) == 3:
             delaunay_tri_indices.append(tuple(indices))
            
    return delaunay_tri_indices

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Warp src_tri to dst_tri.
    Uses BORDER_CONSTANT with white color (255,255,255) to pad gaps.
    """
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    
    # BORDER_CONSTANT = 0. borderValue = (255, 255, 255)
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, 
                         flags=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, 
                         borderValue=(255, 255, 255))
    return dst

def warp_triangle(img, dst_img, src_tri, dst_tri):
    """
    Warps a triangular region.
    """
    # Find bounding box for each triangle
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))

    # r1 is the cropping rect for Source. 
    # Critical: Use intersection with image bounds to avoid errors?
    # cv2.warpAffine handles out-of-bounds source reading with borderMode.
    # But slicing `img[y:h, x:w]` will fail if indices are negative.
    # So we must handle the source cropping carefully or rely on warpAffine to handle the offset.
    
    # Better approach: 
    # Just configure the warp matrix relative to r2 (dest rect) and r1 (source rect).
    
    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t2_rect_int.append(((dst_tri[i][0] - r2[0]), (dst_tri[i][1] - r2[1])))
        t2_rect.append(((dst_tri[i][0] - r2[0]), (dst_tri[i][1] - r2[1])))
        t1_rect.append(((src_tri[i][0] - r1[0]), (src_tri[i][1] - r1[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    # We must grab the source patch `img_rect`. 
    # If r1 is outside img bounds, we need to handle it.
    
    # Helper for safe cropping
    x, y, w, h = r1
    
    # Because landmarks (boundary points) can be outside image, r1 can be outside.
    # If using warpAffine, we can pass the whole image? No, slow.
    # We must pad the source image logic strictly?
    
    # To keep it simple and correct:
    # 1. Calculate warped patch size (r2).
    # 2. Compute Affine T from t1_rect (src offsets) -> t2_rect (dst offsets).
    # 3. But t1_rect assumes we cropped `img` at `r1`.
    # 4. If we don't crop, we use `src_tri` directly and `dst_tri` translated by `-r2`.
    
    # Use full Source Image + Translation in Affine Matrix:
    # Mapping: src_tri -> (dst_tri - r2_offset).
    # Result image size: (r2.w, r2.h)
    
    size = (r2[2], r2[3])
    
    # Correct Matrix: Map Source Global Coords -> Dest Local Coords
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(t2_rect))
    
    warped_img = cv2.warpAffine(img, warp_mat, size, None, 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
    
    warped_img = warped_img * mask

    # Copy triangular region of the rectangular patch to the output image
    dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + warped_img


def main():
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"Error: Reference image not found at {REFERENCE_IMAGE_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Processing Reference Image...")
    # 1. Processing Reference
    ref_landmarks, ref_img = get_landmarks(REFERENCE_IMAGE_PATH)
    if ref_landmarks is None:
        return 

    h, w, _ = ref_img.shape
    rect = (0, 0, w, h)
    
    # Extend Reference Landmarks with Boundary Points
    ref_boundary = get_boundary_points(w, h)
    ref_full_landmarks = ref_landmarks + ref_boundary
    
    # 2. Triangulation (on FULL set)
    # Note: Delaunay on full set (Face + Frame) covers the whole canvas.
    dt_indices = get_delaunay_triangles((0, 0, w, h), ref_full_landmarks)
    print(f"Reference processed. Triangles: {len(dt_indices)}")

    # 3. Processing Inputs
    image_files = []
    for ext in EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    image_files.sort()
    
    # Clear logs
    with open(LOG_FILE, 'w') as f:
        f.write("Filename, Scale, ShiftX, ShiftY, ProcessingTime(s)\n")

    for img_path in image_files:
        start_time = time.time()
        filename = os.path.basename(img_path)
        print(f"Aligning {filename}...", end="")
        
        src_landmarks, src_img = get_landmarks(img_path)
        
        if src_landmarks is None:
            print(" Skipping (No face)")
            continue

        # --- A. Estimate Global Affine Transform (Input -> Ref) ---
        # We use this to establish "Input Boundary Points" that match the Ref Boundary Points
        src_pts_np = np.array(src_landmarks, dtype=np.float32)
        ref_pts_np = np.array(ref_landmarks, dtype=np.float32)
        
        # Estimate Similarity Transform (Scale, Rotation, Translation) - closer to "Camera Movement"
        # estimateAffinePartial2D checks for (scale, rotation, translation)
        # We need the transform that maps SRC -> REF
        M, _ = cv2.estimateAffinePartial2D(src_pts_np, ref_pts_np)
        
        if M is None:
             print(" Skipping (Cannot verify matrix)")
             continue
             
        # Log Metrics
        # M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        tx = M[0, 2]
        ty = M[1, 2]
        
        # --- B. Back-Project Reference Boundary to Input Space ---
        # We need landmarks for the Input that correspond to Ref Boundary.
        # Ideally, P_src = M_inv * P_ref
        
        # Ref Boundary
        ref_b_np = np.array([ref_boundary], dtype=np.float32) # (1, N, 2)
        
        # Invert M to map Ref -> Src
        M_inv = cv2.invertAffineTransform(M)
        
        # Apply M_inv to Ref Boundary Points
        src_boundary_np = cv2.transform(ref_b_np, M_inv)
        src_boundary = []
        for pt in src_boundary_np[0]:
            src_boundary.append((int(pt[0]), int(pt[1])))
            
        # Combine
        src_full_landmarks = src_landmarks + src_boundary
        
        # --- C. Warp ---
        # Create output canvas (White background init)
        aligned_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Warp each triangle
        for idx1, idx2, idx3 in dt_indices:
            t_ref = [ref_full_landmarks[idx1], ref_full_landmarks[idx2], ref_full_landmarks[idx3]]
            t_src = [src_full_landmarks[idx1], src_full_landmarks[idx2], src_full_landmarks[idx3]]
            
            warp_triangle(src_img, aligned_img, t_src, t_ref)
            
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, aligned_img)
        
        # Log Stats
        elapsed = time.time() - start_time
        
        log_entry = f"{filename}, {scale:.2f}x, {tx:.1f}px, {ty:.1f}px, {elapsed:.2f}s"
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry + "\n")
            
        print(f" Done ({elapsed:.2f}s)")

if __name__ == "__main__":
    main()
