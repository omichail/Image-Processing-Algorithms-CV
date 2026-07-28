import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = "C:\\Users\\user\\Desktop\\cameraman.tif"
image = Image.open(image_path).convert('L')
original_array = np.array(image)

plt.imshow(original_array, cmap='gray')
plt.title('Original image')
plt.show()

c = 1

log_transform = c * np.log(1 + original_array)

x_min = np.min(log_transform)
x_max = np.max(log_transform)

lin_contrast = (255 / (x_max - x_min)) * (log_transform - x_min)

lin_contrast = np.array(lin_contrast, dtype=np.uint8)

plt.imshow(lin_contrast, cmap='gray')
plt.title('After logarithmization and contrast')
plt.show()

plt.hist(lin_contrast.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram')
plt.xlabel('Image intensity')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, 256, 20))
plt.show()


x1, x2 = 60, 190


lut = np.zeros(256, dtype=np.float32)

for r in range(256):
    if r < x1:
        lut[r] = (0 / x1) * r if x1 > 0 else 0
    elif x1 <= r <= x2:

        lut[r] = ((255 - 0) / (x2 - x1)) * (r - x1) + 0
    else:

        lut[r] = ((255 - 255) / (255 - x2)) * (r - x2) + 255 if x2 < 255 else 255


lut = np.clip(lut, 0, 255).astype(np.uint8)

result_img = lut[lin_contrast]

plt.imshow(result_img, cmap='gray')
plt.title('After preparation')
plt.show()
