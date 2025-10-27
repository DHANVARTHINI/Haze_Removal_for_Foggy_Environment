import cv2
import numpy as np
import os
import time

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)

def compute_single_image_snr(image):
    image = image.astype(np.float32)
    smooth = cv2.GaussianBlur(image, (5, 5), 1)
    noise = image - smooth
    signal_power = np.mean(image ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def defog_image(image):
    # Convert to LAB and apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for light enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Apply denoising for a clean look
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h=10, hColor=10,
                                               templateWindowSize=7, searchWindowSize=21)
    return enhanced

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"clahedefogged_{filename}")

            input_image = cv2.imread(input_path)
            if input_image is None:
                print(f"Could not read: {filename}")
                continue

            start_time = time.time()
            defogged_image = defog_image(input_image)
            elapsed = (time.time() - start_time) * 1000

            psnr_val = compute_psnr(input_image, defogged_image)
            snr_input = compute_single_image_snr(input_image)
            snr_output = compute_single_image_snr(defogged_image)

            print(f"Processed: {filename}")
            print(f"PSNR: {psnr_val:.2f} dB")
            print(f"SNR (Input):  {snr_input:.2f} dB")
            print(f"SNR (Output): {snr_output:.2f} dB")
            print(f"Time: {elapsed:.2f} ms\n")

            cv2.imwrite(output_path, defogged_image)

#Set your input/output folder paths here:
input_folder = r"C:\Users\dhane\Downloads\main project\fog image"
output_folder = r"C:\Users\dhane\Downloads\main project\output"

process_images(input_folder, output_folder)
