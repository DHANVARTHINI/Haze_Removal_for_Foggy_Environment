import cv2
import numpy as np
import time
import os

# Apply White Balance using LAB color space
def apply_white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    white_balanced = cv2.merge((l, a, b))
    return cv2.cvtColor(white_balanced, cv2.COLOR_LAB2BGR)

# Apply CLAHE for Contrast Enhancement
def apply_contrast_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    contrast_enhanced = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(contrast_enhanced, cv2.COLOR_LAB2BGR)

# Compute Weight Maps
def compute_weight_maps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    chroma = np.sqrt(np.sum((img / 255.0) ** 2, axis=2))
    saliency = cv2.Laplacian(gray, cv2.CV_64F).var() * np.ones_like(gray)

    total_weight = gray + chroma + saliency
    w_luminance = gray / total_weight
    w_chromaticity = chroma / total_weight
    w_saliency = saliency / total_weight

    return w_luminance, w_chromaticity, w_saliency

# Fuse images using weight maps
def fuse_images(img1, img2, weights1, weights2):
    fused = np.zeros_like(img1, dtype=np.float32)
    for i in range(3):
        fused[..., i] = (img1[..., i] * weights1[i]) + (img2[..., i] * weights2[i])
    return np.clip(fused, 0, 255).astype(np.uint8)

# PSNR calculation
def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)

# SNR calculation
def compute_snr(image):
    image = image.astype(np.float32)
    smooth = cv2.GaussianBlur(image, (5, 5), 1)
    noise = image - smooth

    signal_power = np.mean(image ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)

# Process images from folder
def process_image_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Process only image files
            input_path = os.path.join(input_folder, filename)
            frame = cv2.imread(input_path)

            if frame is None:
                print(f"Warning: Could not read {filename}. Skipping...")
                continue

            start_time = time.time()

            # Apply white balance and contrast enhancement
            white_balanced = apply_white_balance(frame)
            contrast_enhanced = apply_contrast_enhancement(frame)

            # Compute weight maps
            w1 = compute_weight_maps(white_balanced)
            w2 = compute_weight_maps(contrast_enhanced)

            # Fuse images
            defogged_image = fuse_images(white_balanced, contrast_enhanced, w1, w2)

            # Compute quality metrics
            psnr_output = compute_psnr(frame, defogged_image)
            snr_input = compute_snr(frame)
            snr_output = compute_snr(defogged_image)

            print(f"{filename}")
            print(f"PSNR (Defogged): {psnr_output:.2f} dB")
            print(f"SNR  (Input):    {snr_input:.2f} dB")
            print(f"SNR  (Output):   {snr_output:.2f} dB")
            print(f"Time Taken:      {(time.time() - start_time)*1000:.2f} ms\n")

            # Save output image with a "defogged_" prefix
            output_path = os.path.join(output_folder, f"fusedefogged_{filename}")
            cv2.imwrite(output_path, defogged_image)

# Run it
input_folder = r"C:\Users\dhane\Downloads\main project\fog image"      # Change this to your input folder path
output_folder = r"C:\Users\dhane\Downloads\main project\output"  # Change this to your output folder path

process_image_folder(input_folder, output_folder)
