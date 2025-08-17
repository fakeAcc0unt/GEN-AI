import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
image = x_train[0]  

# Normalize to [0,1]
image = image / 255.0

# Add random noise
noisy_image = image + 0.5 * np.random.rand(*image.shape)
noisy_image = np.clip(noisy_image, 0, 1)

plt.imshow(noisy_image, cmap="gray")
plt.title("Step 0: Noisy Image")
plt.show()

# Denoising steps
for step in range(1, 3):
    denoised = gaussian_filter(noisy_image, sigma=step) 
    plt.imshow(denoised, cmap="gray")
    plt.title(f"Step {step}: Denoised")
    plt.show()
