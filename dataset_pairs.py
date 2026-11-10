"""Match image/mask stems before splitting the historical segmentation dataset."""
from pathlib import Path
import random


def split_pairs(image_directory, mask_directory, train_ratio=0.85, seed=1337):
    images = {p.stem: str(p) for p in Path(image_directory).glob('*.jpg')}
    masks = {p.stem: str(p) for p in Path(mask_directory).glob('*.png')}
    if images.keys() != masks.keys():
        raise ValueError('Image/mask filename stems must match exactly')
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be strictly between zero and one')
    pairs = sorted((images[key], masks[key]) for key in images)
    if len(pairs) < 2:
        raise ValueError('At least two image/mask pairs are required')
    random.Random(seed).shuffle(pairs)
    n_validation = max(1, int(len(pairs) * (1 - train_ratio)))
    return pairs[:-n_validation], pairs[-n_validation:]
