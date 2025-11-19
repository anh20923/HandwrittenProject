import numpy as np
import cv2
from PIL import Image

def preprocess_image_for_mnist(pil_img):
    """
    Converts arbitrary uploaded image into MNIST format:
    - grayscale
    - 28x28
    - normalized to 0..1
    - shape (1, 28, 28, 1)
    """
    # Convert to grayscale
    img = pil_img.convert("L")

    # Convert to numpy
    img_np = np.array(img)

    # Resize using OpenCV (keeps better quality)
    img_resized = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize 0..1
    img_norm = img_resized.astype("float32") / 255.0

    # Reshape for CNN input
    img_norm = img_norm.reshape(1, 28, 28, 1)

    return img_norm
