# segmentation_evaluation.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model

# --- Configuration ---
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BEST_MODEL_PATH = "FireSegmentation_best.keras"
EVALUATION_IMAGE_PATH = "examples/fire_image.png" # custom image for evaluation

# --- Main Evaluation Function ---
def evaluate_on_custom_image():
    """
    Loads the best trained model and predicts a mask for a custom image.
    """
    print(f"\n--- Evaluating model on custom image: {EVALUATION_IMAGE_PATH} ---")
    if not os.path.exists(EVALUATION_IMAGE_PATH):
        print(f"Error: Evaluation image not found at {EVALUATION_IMAGE_PATH}")
        return

    # Load the best model
    model = load_model(BEST_MODEL_PATH, compile=False)

    # Load and preprocess the custom image
    img = cv2.imread(EVALUATION_IMAGE_PATH)
    if img is None:
        raise ValueError('Could not decode the image')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    # The model expects a batch of images, so we add an extra dimension
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict the mask
    predicted_mask = model.predict(img_input)[0]

    # Threshold the mask to get a binary result (black and white)
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_resized)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Fire Mask")
    # Use cmap='gray' to show the binary mask in black and white
    plt.imshow(predicted_mask_binary.squeeze(-1), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_on_custom_image()
