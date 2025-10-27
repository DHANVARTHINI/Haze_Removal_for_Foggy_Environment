import os
import PIL.Image as Image
import skimage.io as io
import numpy as np
import time
from gf import guided_filter
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_dark_channel(src, radius):
    print("Computing Dark Channel...")
    start = time.time()
    tmp = src.min(axis=2)
    rows, cols = tmp.shape
    dark = np.zeros((rows, cols), dtype=np.double)
    for i in range(rows):
        for j in range(cols):
            rmin = max(0, i - radius)
            rmax = min(i + radius, rows - 1)
            cmin = max(0, j - radius)
            cmax = min(j + radius, cols - 1)
            dark[i, j] = tmp[rmin:rmax + 1, cmin:cmax + 1].min()
    print("Dark Channel Computation Time:", time.time() - start)
    return dark

def compute_snr(image):
    signal = np.mean(image ** 2)
    noise = np.mean((image - np.mean(image)) ** 2)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(signal / noise)

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double) / 255.

        if self.src is None:
            print(f"Failed to load image: {img_path}")
            return

        if len(self.src.shape) == 2:
            print(f"Converting grayscale image to RGB: {img_path}")
            self.src = np.stack([self.src, self.src, self.src], axis=-1)

        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    def get_dark_channel(self):
        self.dark = compute_dark_channel(self.src, self.radius)

    def get_air_light(self):
        print("Computing Air Light...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)
        print("Air Light Computation Time:", time.time() - start)

    def get_transmission(self):
        print("Computing Transmission...")
        start = time.time()
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                pixel = (self.src[rmin:rmax + 1, cmin:cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1. - self.omega * pixel
        print("Transmission Computation Time:", time.time() - start)

    def guided_filter(self):
        print("Applying Guided Filter...")
        start = time.time()
        self.gtran = guided_filter(self.src, self.tran, self.r, self.eps)
        print("Guided Filter Time:", time.time() - start)

    def recover(self):
        print("Recovering Image...")
        start = time.time()
        self.gtran[self.gtran < self.t0] = self.t0
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        self.dst = (self.src - self.Alight) / t + self.Alight
        self.dst *= 255
        self.dst[self.dst > 255] = 255
        self.dst[self.dst < 0] = 0
        self.dst = self.dst.astype(np.uint8)
        print("Recovery Time:", time.time() - start)

    def show(self, output_path, filename):
        io.imsave(os.path.join(output_path, filename), self.dst)
        print(f"Saved: {filename}")

if __name__ == '__main__':
    folder_path = r"C:\Users\dhane\Downloads\main project\fog image"  
    output_folder = r"C:\Users\dhane\Downloads\main project\claheoutput"  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hr = HazeRemoval()

    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing: {file_name}")

            start_time = time.time()

            hr.open_image(image_path)
            if hr.src is not None:
                snr_input = compute_snr(hr.src)
                hr.get_dark_channel()
                hr.get_air_light()
                hr.get_transmission()
                hr.guided_filter()
                hr.recover()
                snr_output = compute_snr(hr.dst.astype(np.double) / 255.0)
                end_time = time.time()
                total_time = end_time - start_time

                hr.show(output_folder, file_name)

                print(f"SNR (Input) : {snr_input:.2f} dB")
                print(f"SNR (Output): {snr_output:.2f} dB")
                print(f"Processing Time: {total_time:.2f} seconds")
            else:
                print(f"Skipping file due to loading error: {file_name}")
