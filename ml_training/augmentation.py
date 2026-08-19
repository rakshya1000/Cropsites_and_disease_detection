"""
Image augmentation, implemented from scratch with NumPy + Pillow only.

Pillow is used here only for the geometric resampling that rotation and
zoom require (the same role it plays in preprocessing.py for resizing —
there's no way to do sub-pixel interpolation in plain NumPy without
reimplementing what Pillow already does). Flipping, brightness, and noise
are pure NumPy with no Pillow involved at all.

Every function here takes a single image as a NumPy array of shape
(H, W, 3), dtype float32, values already normalized to [0.0, 1.0] — i.e.
the same format produced by preprocessing.load_and_preprocess_image().
Every function RETURNS A NEW ARRAY. None of them modify the input in place,
so the original dataset arrays are never touched by calling these.
"""

import numpy as np
from PIL import Image


def horizontal_flip(image):
    """Flips the image left-right. Pure NumPy — no Pillow needed."""
    return image[:, ::-1, :].copy()


def vertical_flip(image):
    """Flips the image top-bottom. Pure NumPy — no Pillow needed."""
    return image[::-1, :, :].copy()


def rotate(image, angle_degrees):
    """
    Rotates the image by angle_degrees (positive = counter-clockwise) and
    keeps the output the same size as the input. Uses Pillow only for the
    interpolation math; the array in and the array out are both fresh
    NumPy arrays, and the input is never modified.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    rotated = pil_img.rotate(angle_degrees, resample=Image.BILINEAR, expand=False)
    return np.asarray(rotated, dtype=np.float32) / 255.0


def zoom(image, zoom_factor):
    """
    Zooms in (zoom_factor > 1.0) or out (zoom_factor < 1.0), then
    crops/pads back to the original (H, W) so every image in a batch
    keeps the same shape. Uses Pillow only for the resize interpolation.
    """
    h, w = image.shape[:2]
    img_uint8 = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    new_h, new_w = max(1, int(h * zoom_factor)), max(1, int(w * zoom_factor))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    if zoom_factor >= 1.0:
        # Zoomed in: crop back to original size from the center
        left = max(0, (new_w - w) // 2)
        top = max(0, (new_h - h) // 2)
        cropped = resized.crop((left, top, left + w, top + h))
        return np.asarray(cropped, dtype=np.float32) / 255.0
    else:
        # Zoomed out: paste the smaller image onto a black canvas, centered
        canvas = Image.new("RGB", (w, h))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        canvas.paste(resized, (left, top))
        return np.asarray(canvas, dtype=np.float32) / 255.0


def adjust_brightness(image, factor):
    """
    Multiplies pixel values by `factor` (>1.0 brightens, <1.0 darkens),
    then clips back into [0.0, 1.0]. Pure NumPy.
    """
    return np.clip(image * factor, 0.0, 1.0)


def add_random_noise(image, rng, std=0.02):
    """
    Adds Gaussian noise (mean 0, given std) to every pixel, then clips
    back into [0.0, 1.0]. Pure NumPy — rng is a np.random.Generator.
    """
    noise = rng.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def random_augment(
    image,
    rng,
    flip_h_prob=0.5,
    flip_v_prob=0.2,
    rotation_range=15.0,
    zoom_range=(0.9, 1.1),
    brightness_range=(0.8, 1.2),
    noise_prob=0.3,
    noise_std=0.02,
):
    """
    Applies a random combination of the above transforms to ONE image and
    returns a new array. Called once per image, per batch, per epoch during
    training — so the same source image gets different random augmentations
    every time it's seen. `rng` should be a shared np.random.Generator so
    randomness is reproducible when seeded.
    """
    augmented = image

    if rng.random() < flip_h_prob:
        augmented = horizontal_flip(augmented)

    if rng.random() < flip_v_prob:
        augmented = vertical_flip(augmented)

    angle = rng.uniform(-rotation_range, rotation_range)
    augmented = rotate(augmented, angle)

    zoom_factor = rng.uniform(zoom_range[0], zoom_range[1])
    augmented = zoom(augmented, zoom_factor)

    brightness_factor = rng.uniform(brightness_range[0], brightness_range[1])
    augmented = adjust_brightness(augmented, brightness_factor)

    if rng.random() < noise_prob:
        augmented = add_random_noise(augmented, rng, std=noise_std)

    return augmented
