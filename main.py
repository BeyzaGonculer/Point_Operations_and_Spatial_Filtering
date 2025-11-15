import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.figsize"] = (6, 4)

def load_bgr_image():
    img_path = Path("images/astronaut.png")
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Görsel bulunamadı!")
    return bgr


bgr = load_bgr_image()
print("BGR shape:", bgr.shape)

# BGR -> RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# BGR -> HSV
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# BGR -> CIE L*a*b*
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

def imshow_gray(img, title=None):
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    if title: plt.title(title)
    plt.axis("off")
    plt.show()

# RGB channels
R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
imshow_gray(R, "R channel")
imshow_gray(G, "G channel")
imshow_gray(B, "B channel")

# HSV channels
H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
imshow_gray(H, "H (Hue) channel")
imshow_gray(S, "S (Saturation) channel")
imshow_gray(V, "V (Value) channel")

# LAB channels
L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
imshow_gray(L, "L* (Luminance) channel")
imshow_gray(a, "a* channel")
imshow_gray(b, "b* channel")

def save_gray(img, path):
    # güvenli: uint8 değilse dönüştür
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)

save_gray(R, "outputs/ch_R.png")
save_gray(G, "outputs/ch_G.png")
save_gray(B, "outputs/ch_B.png")

save_gray(H, "outputs/ch_H.png")   # H: 0–179 (uint8) — doğrudan kaydedilebilir
save_gray(S, "outputs/ch_S.png")   # 0–255
save_gray(V, "outputs/ch_V.png")   # 0–255

save_gray(L, "outputs/ch_Lstar.png")  # 0–255 (OpenCV 0..100’ün ölçekli sürümü)
save_gray(a, "outputs/ch_astar.png")  # 0–255 (offsetli)
save_gray(b, "outputs/ch_bstar.png")  # 0–255 (offsetli)

print("All channels saved under outputs/")


# 1.2

# 1.2.1

gray = L.copy()

def compute_histogram(gray_u8: np.ndarray):
    """0–255 histogram (256 bin)."""
    assert gray_u8.dtype == np.uint8
    hist = np.bincount(gray_u8.flatten(), minlength=256)
    return hist

def plot_hist(hist, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Intensity (0–255)")
    plt.ylabel("Pixel count")
    plt.bar(np.arange(256), hist, width=1.0)
    plt.xlim(0, 255)
    plt.show()

hist_orig = compute_histogram(gray)
plot_hist(hist_orig, "Original L* histogram")
imshow_gray(gray, "Original L* image")
save_gray(gray, "outputs/L_original.png")

def hist_equalize_from_scratch(gray_u8: np.ndarray):

    assert gray_u8.dtype == np.uint8

    # 1) Histogram
    hist = np.bincount(gray_u8.flatten(), minlength=256).astype(np.float64)

    # 2) PDF (Probability Density Function)
    num_pixels = hist.sum()
    pdf = hist / num_pixels

    # 3) CDF (Cumulative Distribution Function)
    cdf = np.cumsum(pdf)          # 0–1 aralığında monoton artan

    # 4) Mapping: s_k = T(r_k) = (L-1) * CDF
    mapping = np.floor(255 * cdf + 0.5).astype(np.uint8)   # 0–255 arası yeni değerler

    # 5) Mapped orginal pixel values
    eq_img = mapping[gray_u8]

    # 6) Equilized histogram
    hist_eq = np.bincount(eq_img.flatten(), minlength=256).astype(np.float64)

    return eq_img, hist, hist_eq, pdf, cdf

eq, hist_orig, hist_eq, pdf, cdf = hist_equalize_from_scratch(gray)


# Equilized image draw and save
imshow_gray(eq, "Equalized L* image")
save_gray(eq, "outputs/L_equalized.png")

# Original vs Equilized histogram draw
plot_hist(hist_orig, "Original L* histogram")
plot_hist(hist_eq, "Equalized L* histogram")

clean = eq.copy()   # histogram-equalized L* görüntüsü
imshow_gray(clean, "Clean equalized L* image")
save_gray(clean, "outputs/L_equalized_clean.png")
