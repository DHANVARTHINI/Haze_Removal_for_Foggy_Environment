import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Load the fog classification model
model = tf.keras.models.load_model('fog_classifier_model.h5')

# DCP and enhancement pipeline
def get_dark_channel(image, size=15):
    min_channel = cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def get_atmosphere(image, dark):
    h, w = dark.shape
    num_pixels = h * w
    num_bright = int(max(num_pixels * 0.001, 1))
    flat_dark = dark.reshape(-1)
    indices = np.argsort(flat_dark)[-num_bright:]
    brightest = image.reshape(-1, 3)[indices]
    A = np.mean(brightest, axis=0)
    return A

def get_transmission(image, A, omega=0.99, size=15):
    norm_img = image / A
    dark = get_dark_channel(norm_img, size)
    transmission = 1 - omega * dark
    return transmission

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * I + mean_b
    return q

def recover(image, transmission, A, t0=0.1):
    transmission = np.clip(transmission, t0, 1)
    J = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        J[:, :, i] = (image[:, :, i] - A[i]) / transmission + A[i]
    return np.clip(J, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma=1.1):
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])).astype("uint8")
    return cv2.LUT(image, table)

def defog_dcp(image):
    image = image.astype(np.float32)
    dark = get_dark_channel(image)
    A = get_atmosphere(image, dark)
    transmission = get_transmission(image, A)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    transmission_refined = guided_filter(gray, transmission, r=60, eps=1e-3)
    recovered = recover(image, transmission_refined, A)
    gamma_corrected = gamma_correction(recovered, gamma=1.1)
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 3)
    sharpened = cv2.addWeighted(gamma_corrected, 1.5, blurred, -0.5, 0)
    return sharpened

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_snr(image):
    image = image.astype(np.float64)
    signal = np.mean(image ** 2)
    noise = np.mean((image - np.mean(image)) ** 2)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(signal / noise)

# Paths
input_folder = r"C:\Users\dhane\Downloads\main project\fog image"
output_folder = r"C:\Users\dhane\Downloads\main project\output"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"[WARNING] Could not read {filename}. Skipping.")
            continue

        resized = cv2.resize(image, (128, 128))
        img_array = resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction <= 0.5:
            start = time.time()
            snr_input = calculate_snr(image)
            defogged = defog_dcp(image)
            snr_output = calculate_snr(defogged)
            psnr_value = calculate_psnr(image, defogged)
            end = time.time()

            output_path = os.path.join(output_folder, f"cnndefogged_{filename}")
            cv2.imwrite(output_path, defogged)

            print(f"Processed: {filename}")
            print(f"Fog Detected")
            print(f"PSNR: {psnr_value:.2f} dB")
            print(f"SNR (Input) : {snr_input:.2f} dB")
            print(f"SNR (Output): {snr_output:.2f} dB")
            print(f"Time Taken : {(end - start) * 1000:.2f} ms\n")
        else:
            print(f"Skipped: {filename} (Non-Fog)")
