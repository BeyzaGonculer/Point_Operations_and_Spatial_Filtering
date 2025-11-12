import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data

def load_sample_rgb():
    # skimage.data.astronaut() RGB döndürür (uint8)
    rgb = data.astronaut()
    return rgb
def imshow_rgb(img, title=None):
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis("off")
    plt.show()

def imshow_gray(img, title=None):
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    if title: plt.title(title)
    plt.axis("off")
    plt.show()
def main():
    print("Color Space Conversion and Analysis.")

if __name__ == "__main__":
    main()





plt.rcParams["figure.figsize"] = (6, 4)


print("Part 1.1")
rgb = load_sample_rgb()
print("Image shape:", rgb.shape, "dtype:", rgb.dtype)

print("ECE530 A1 - Part 1.1")
rgb = load_sample_rgb()
imshow_rgb(rgb, "Input RGB (astronaut)")



