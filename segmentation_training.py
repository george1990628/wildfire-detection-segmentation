# Public maintenance of the uploaded Pioneer notebook; see docs/maintenance.md.
# segmentation_training.py

import os
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, MaxPooling2D, Dropout, Lambda, Conv2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import load_img

# --- Configuration ---
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
EPOCHS = 30
BATCH_SIZE = 16
TRAIN_SET_RATIO = 0.85

DIR_IMAGES = "frames/Segmentation/Data/Images"
DIR_MASKS = "frames/Segmentation/Data/Masks"
BEST_MODEL_PATH = "FireSegmentation_best.keras"

# --- U-Net Model Architecture ---
def model_unet_kaggle(img_height, img_width, img_channel):
    """
    Defines the U-Net model architecture.
    """
    inputs = Input((img_height, img_width, img_channel))
    s = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Main Training Function ---
def train_model():
    """
    Loads data, builds the U-Net model, and trains it.
    """
    print("--- Starting Segmentation Model Training ---")

    from dataset_pairs import split_pairs
    training, validation = split_pairs(DIR_IMAGES, DIR_MASKS, TRAIN_SET_RATIO, 1337)
    train_img_paths, train_mask_paths = map(list, zip(*training))
    val_img_paths, val_mask_paths = map(list, zip(*validation))

    # Load training data
    print(f"\nLoading {len(train_img_paths)} training images and masks...")
    x_train = np.zeros((len(train_img_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    y_train = np.zeros((len(train_mask_paths), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    for i, path in tqdm(enumerate(train_img_paths)):
        x_train[i] = load_img(path, target_size=IMAGE_SIZE)
    for i, path in tqdm(enumerate(train_mask_paths)):
        mask = load_img(path, target_size=IMAGE_SIZE, color_mode="grayscale", interpolation="nearest")
        y_train[i] = (np.asarray(mask) > 0)[..., np.newaxis]

    # Load validation data
    print(f"\nLoading {len(val_img_paths)} validation images and masks...")
    x_val = np.zeros((len(val_img_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    y_val = np.zeros((len(val_mask_paths), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    for i, path in tqdm(enumerate(val_img_paths)):
        x_val[i] = load_img(path, target_size=IMAGE_SIZE)
    for i, path in tqdm(enumerate(val_mask_paths)):
        mask = load_img(path, target_size=IMAGE_SIZE, color_mode="grayscale", interpolation="nearest")
        y_val[i] = (np.asarray(mask) > 0)[..., np.newaxis]

    # Build and compile the model
    model = model_unet_kaggle(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name='binary_iou')])

    # Callback to save the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_binary_iou', mode='max')

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])
    print(f"--- Training complete. Best model saved to {BEST_MODEL_PATH} ---")

if __name__ == '__main__':
    train_model()
