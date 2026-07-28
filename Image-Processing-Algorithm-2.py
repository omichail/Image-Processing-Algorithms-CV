import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def draw(array_1, array_2, string_1, string_2):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(array_1, cmap='gray')
    plt.title(string_1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(array_2, cmap='gray')
    plt.title(string_2)
    plt.axis('off')

    plt.show()

def histogram(array, string):
    plt.hist(array.ravel(), bins=256, range=[0, 256], density=True, color='gray')
    plt.title(string)
    plt.xlabel('Image intensity')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, 256, 30))


def apply_sobel(image, kernel_size):
    img_float = image.astype(np.float32)

    if kernel_size == 3:
        Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gx = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
    elif kernel_size == 5:
        Gx = np.array([[1/4, 1/2, 1, 1/2, 1/4],
                        [1/2, 1, 2, 1, 1/2],
                        [0, 0, 0, 0, 0],
                        [-1/2, -1, -2, -1, -1/2],
                        [-1/4, -1/2, -1, -1/2, -1/4]])
        Gy = np.array([[-1/4, -1/2, 0, 1/2, 1/4],
                        [-1/2, -1, 0, 1, 1/2],
                        [-1, -2, 0, 2, 1],
                        [-1/2, -1, 0, 1, 1/2],
                        [-1/4, -1/2, 0, 1/2, 1/4]])

    gradient_x = convolve2d(img_float, Gx, mode='same', boundary='symm')
    gradient_y = convolve2d(img_float, Gy, mode='same', boundary='symm')

    sobel = np.sqrt(gradient_x**2 + gradient_y**2)
    return sobel

def linear_contrast(img):
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
       return np.zeros_like(img, dtype=np.uint8)

    contrasted = (img - min_val) * (255.0 / (max_val - min_val))
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def invert_and_threshold(img, threshold=200):
    inverted = 255 - img

    binary = np.where(inverted > threshold, 255, 0).astype(np.uint8)
    return binary

def piecewise_linear_transform(img, x1, x2):
    img_float = img.astype(np.float32)
    result = np.piecewise(img_float,
        [img_float <= x1, (img_float > x1) & (img_float < x2), img_float >= x2],
        [0, lambda x: (x - x1) * 255 / (x2 - x1), 255]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    image_path = "C:\\Users\\user\\Desktop\\cameraman.tif"
    image = Image.open(image_path).convert('L')
    orig_img = np.array(image)

    raw_sobel3 = apply_sobel(orig_img, kernel_size=3)
    contrasted_sobel3 = linear_contrast(raw_sobel3)
    negative_sobel3 = 255 - contrasted_sobel3
    final_sobel3 = invert_and_threshold(contrasted_sobel3, threshold=200)

    raw_sobel5 = apply_sobel(orig_img, kernel_size=5)
    contrasted_sobel5 = linear_contrast(raw_sobel5)
    negative_sobel5 = 255 - contrasted_sobel5
    final_sobel5 = invert_and_threshold(contrasted_sobel5, threshold=200)

    x1, x2 = 0, 80
    prepped_img = piecewise_linear_transform(orig_img, x1, x2)

    raw_sobel3_prep = apply_sobel(prepped_img, kernel_size=3)
    contrasted_sobel3_prep = linear_contrast(raw_sobel3_prep)
    negative_sobel3_prep = 255 - contrasted_sobel3_prep
    final_sobel3_prep = invert_and_threshold(contrasted_sobel3_prep, threshold=200)

    plt.imshow(orig_img, cmap='gray')
    plt.title('Original image')
    plt.axis('off')
    plt.show()


    draw(contrasted_sobel3, contrasted_sobel5, "Sobel [3x3]", "Sobel [5x5]")
    draw(negative_sobel3, negative_sobel5, "Sobel [3x3] (Inversion)", "Sobel [5x5] (Inversion)")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    histogram(negative_sobel3, "Histogram Sobel [3x3]")
    plt.subplot(1, 2, 2)
    histogram(negative_sobel5, "Histogram Sobel [5x5]")
    plt.show()

    draw(final_sobel3, final_sobel5, "Sobel [3x3] (Threshold function)", "Sobel [5x5] (Threshold function)")

    histogram(orig_img, "Histogram of the original image")
    plt.show()

    plt.imshow(prepped_img, cmap='gray')
    plt.title(f'Prepared image\n(x1={x1}, x2={x2})')
    plt.axis('off')
    plt.show()

    plt.imshow(contrasted_sobel3_prep, cmap='gray')
    plt.title("Sobel 3x3 from the prepared")
    plt.axis('off')
    plt.show()

    plt.imshow(negative_sobel3_prep, cmap='gray')
    plt.title("Sobel 3x3 from the prepared (Inversion)")
    plt.axis('off')
    plt.show()

    histogram(negative_sobel3_prep, "Histogram")
    plt.show()

    plt.imshow(final_sobel3_prep, cmap='gray')
    plt.title("Sobel 3x3 from the prepared (Inversion + Threshold)")
    plt.axis('off')
    plt.show()


    plt.figure(figsize=(15, 8))
    plt.suptitle('Results', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(final_sobel3, cmap='gray')
    plt.title('Sobel 3x3')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(final_sobel5, cmap='gray')
    plt.title('Sobel 5x5')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(final_sobel3_prep, cmap='gray')
    plt.title('Sobel 3x3 prep')
    plt.axis('off')

    plt.show()

main()
