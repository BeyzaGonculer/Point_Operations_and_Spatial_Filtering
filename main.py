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

# 2.1.1 (Add Gaussian Noise)

def add_gaussian_noise(img_u8: np.ndarray, sigma: float = 20.0, mu: float = 0.0) -> np.ndarray:

    assert img_u8.dtype == np.uint8

    # convert to float and add noise
    img_f = img_u8.astype(np.float64)

    # create Gaussian noise
    noise = np.random.normal(loc=mu, scale=sigma, size=img_f.shape)

    # add noise
    noisy_f = img_f + noise

    # 0–255 crop
    noisy_u8 = np.clip(np.rint(noisy_f), 0, 255).astype(np.uint8)
    return noisy_u8

gauss_noisy = add_gaussian_noise(clean, sigma=20.0)
imshow_gray(gauss_noisy, "Gaussian noisy image (sigma=20)")
save_gray(gauss_noisy, "outputs/L_gaussian_noise.png")

# 2.1.2 (Salt and Pepper)

def add_salt_pepper_noise(img_u8: np.ndarray, amount: float = 0.05) -> np.ndarray:

    assert img_u8.dtype == np.uint8
    out = img_u8.copy()

    # total pixel
    num_pixels = out.size
    num_noisy = int(num_pixels * amount)

    # 1D index
    flat = out.flatten()

    # choose indexes for noise
    idx = np.random.choice(num_pixels, size=num_noisy, replace=False)

    # half of them are 0, half of them are 255
    half = num_noisy // 2
    flat[idx[:half]] = 0
    flat[idx[half:]] = 255

    # reshape to reach previos image
    out = flat.reshape(out.shape)
    return out

sp_noisy = add_salt_pepper_noise(clean, amount=0.05)  # %5 pixel
imshow_gray(sp_noisy, "Salt & Pepper noisy image (5%)")
save_gray(sp_noisy, "outputs/L_saltpepper_noise.png")

# 2.2. Linear Filtering (From Scratch)

def cross_correlation_2d(img_u8: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    # convert to float
    img = img_u8.astype(np.float64)
    kH, kW = kernel.shape
    H, W = img.shape

    pad_h = kH // 2
    pad_w = kW // 2

    # Padding with 0
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)

    out = np.zeros_like(img, dtype=np.float64)

    # 2D shift
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]  # kH x kW window
            out[i, j] = np.sum(region * kernel)

    return out


def to_uint8(img_f: np.ndarray) -> np.ndarray:

    return np.clip(np.rint(img_f), 0, 255).astype(np.uint8)


def make_box_kernel(size: int) -> np.ndarray:

    k = np.ones((size, size), dtype=np.float64)
    k /= k.size  # yani size*size
    return k

box3 = make_box_kernel(3)   # 3x3
box7 = make_box_kernel(7)   # 7x7

print("box3 kernel:\n", box3)
print("box7 kernel shape:", box7.shape)


def make_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:

    assert size % 2 == 1, "Kernel size should be odd."

    k = size // 2
    xs = np.arange(-k, k + 1)
    ys = np.arange(-k, k + 1)
    X, Y = np.meshgrid(xs, ys)

    # 2D Gaussian
    g = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # normalize
    g /= g.sum()
    return g

gauss5 = make_gaussian_kernel(size=5, sigma=1.0)
print("gaussian 5x5 kernel sum:", gauss5.sum())

def apply_and_show_filter(img_u8: np.ndarray, kernel: np.ndarray, title: str, save_name: str):
    filtered_f = cross_correlation_2d(img_u8, kernel)
    filtered_u8 = to_uint8(filtered_f)

    imshow_gray(filtered_u8, title)
    save_gray(filtered_u8, f"outputs/{save_name}")
    return filtered_u8


filtered_box3 = apply_and_show_filter(
    gauss_noisy, box3,
    "3x3 Box filter on Gaussian noise",
    "gauss_noisy_box3.png"
)

filtered_box7 = apply_and_show_filter(
    gauss_noisy, box7,
    "7x7 Box filter on Gaussian noise",
    "gauss_noisy_box7.png"
)

filtered_gauss5 = apply_and_show_filter(
    gauss_noisy, gauss5,
    "5x5 Gaussian filter on Gaussian noise",
    "gauss_noisy_gauss5.png"
)

# 2.3. Non-Linear Filtering (From Scratch)

def median_filter_2d(img_u8: np.ndarray, ksize: int = 3) -> np.ndarray:

    assert img_u8.dtype == np.uint8
    assert ksize % 2 == 1, "Kernel size should be odd.(3,5,7) "

    img = img_u8.astype(np.float64)
    H, W = img.shape
    pad = ksize // 2

    # edge-preserving
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")

    out = np.zeros_like(img, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            region = padded[i:i + ksize, j:j + ksize]   # ksize x ksize window
            out[i, j] = np.median(region)

    return to_uint8(out)


# 3x3 median filter on Salt & Pepper noisy image
sp_median3 = median_filter_2d(sp_noisy, ksize=3)
imshow_gray(sp_median3, "3x3 Median filter on Salt & Pepper noise")
save_gray(sp_median3, "outputs/sp_noisy_median3.png")


sp_box3 = apply_and_show_filter(
    sp_noisy,
    box3,
    "3x3 Box filter on Salt & Pepper noise",
    "sp_noisy_box3.png"
)
