import numpy as np
from scipy.ndimage import convolve, binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation, disk, remove_small_objects
from skimage.filters import threshold_otsu, unsharp_mask
from skimage.morphology import reconstruction
from skimage.segmentation import clear_border
import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import pickle
import sys


np.set_printoptions(threshold=sys.maxsize)

# --- Image Analysis ---
def image_analysis(I_Cy5, I_mCherry, I_dsGreen):
    h = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]]) / 4.0
    I_sharp = convolve(I_Cy5, h)

    co = 100
    seed = np.ones_like(I_sharp) * 65535
    seed[:, 0] = seed[:, -1] = seed[0, :] = seed[-1, :] = 0
    mask = I_sharp.copy()
    mask[:, 0] = mask[:, -1] = mask[0, :] = mask[-1, :] = 0

    I_filled = reconstruction(seed, mask, method='erosion')
    I_inside = I_filled.astype(int) - I_sharp

    fgm = np.zeros_like(I_Cy5)
    fgm[I_inside > co] = 1
    fgm = binary_fill_holes(fgm).astype(np.uint8)
    fgm = erosion(fgm, disk(2))

    labeled = label(fgm)
    props = regionprops(labeled)

    for i, prop in enumerate(props):
        A = prop.area
        P = prop.perimeter
        C = (P ** 2) / (4 * np.pi * A) if A > 0 else np.inf
        if not (0.5 <= C <= 2):
            fgm[labeled == (i + 1)] = 0

    labeled = label(fgm)
    props = regionprops(labeled)

    Area, dsGreen, dsGreen_var = [], [], []
    Centroids, mCherry, mCherry_var, Cy5 = [], [], [], []
    LumenPixels, BorderPixels = [], []

    for prop in props:
        # Core/lumen mask
        coords = np.array(prop.coords)
        lumen_coords = [tuple(p) for p in coords]
        lumen_mask = np.zeros_like(fgm, dtype=bool)
        lumen_mask[tuple(coords.T)] = True

        # Border mask
        expanded_mask = lumen_mask.copy()
        for _ in range(6):
            for axis in [0, 1]:
                expanded_mask |= np.roll(expanded_mask, 1, axis=axis)
                expanded_mask |= np.roll(expanded_mask, -1, axis=axis)

        border_mask = expanded_mask & (~(labeled == prop.label))
        border_coords = list(zip(*np.where(border_mask)))

        # Save metrics
        Area.append(len(coords))
        ds_vals = I_dsGreen[tuple(coords.T)]
        dsGreen.append(np.mean(ds_vals))
        dsGreen_var.append(np.var(ds_vals.astype(np.float32)))
        Centroids.append(prop.centroid[::-1])  # (x, y)

        m_vals = I_mCherry[tuple(zip(*border_coords))]
        mCherry.append(np.mean(m_vals))
        mCherry_var.append(np.var(m_vals.astype(np.float32)))
        Cy5_vals = I_Cy5[tuple(zip(*border_coords))]
        Cy5.append(np.mean(Cy5_vals))

        # Save masks as lists of coordinates
        LumenPixels.append(lumen_coords)
        BorderPixels.append(border_coords)

    ResultID = np.arange(len(Area))

    return {
        "ResultID": ResultID,
        "Area": np.array(Area),
        "dsGreen": np.array(dsGreen),
        "dsGreen_var": np.array(dsGreen_var),
        "Centroids": np.array(Centroids).T,
        "mCherry": np.array(mCherry),
        "mCherry_var": np.array(mCherry_var),
        "Cy5": np.array(Cy5),
        "LumenPixels": LumenPixels,
        "BorderPixels": BorderPixels,
    }

# --- Crop and Save Liposomes ---
def crop_and_save_liposomes(I_Cy5, I_mCherry, I_dsGreen, centroids, CropSize, folder, sample_name, result_ids, LumenPixels, BorderPixels):
    def normalize_image(im):
        im = im.astype(np.float32)
        im -= np.min(im)
        if np.max(im) > 0:
            im /= np.max(im)
        return im

    norm_Cy5 = normalize_image(I_Cy5)
    norm_mCherry = normalize_image(I_mCherry)
    norm_dsGreen = normalize_image(I_dsGreen)

    pad = CropSize
    padded_Cy5 = np.pad(norm_Cy5, pad, mode='constant')
    padded_mCherry = np.pad(norm_mCherry, pad, mode='constant')
    padded_dsGreen = np.pad(norm_dsGreen, pad, mode='constant')

    # Adjusted lumen and border coordinates for each cropped image
    adjusted_LumenPixels = []
    adjusted_BorderPixels = []

    for idx, (x, y) in enumerate(centroids.T):
        y_pad, x_pad = int(y) + pad, int(x) + pad
        crop = (slice(y_pad - pad, y_pad + pad), slice(x_pad - pad, x_pad + pad))

        # For each centroid, crop the image and adjust lumen and border coordinates
        cropped_image = padded_mCherry[crop] + padded_Cy5[crop]

        # Lumen and border coordinate adjustments
        lumen_coords = np.array(LumenPixels[idx])
        border_coords = np.array(BorderPixels[idx])

        # Adjust the lumen and border coordinates to the cropped frame
        lumen_coords_adjusted = lumen_coords - np.array([y_pad - pad, x_pad - pad])
        border_coords_adjusted = border_coords - np.array([y_pad - pad, x_pad - pad])

        adjusted_LumenPixels.append(lumen_coords_adjusted)
        adjusted_BorderPixels.append(border_coords_adjusted)

        # Combine R, G, B channels to form RGB image
        R = padded_mCherry[crop] + padded_Cy5[crop]
        G = padded_dsGreen[crop] + padded_Cy5[crop]
        B = padded_mCherry[crop] + padded_Cy5[crop]

        rgb_image = np.stack([np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)], axis=-1)

        obj_id = result_ids[idx]
        filename = f"{sample_name}_obj{obj_id:05d}.tif"
        full_path = os.path.join(folder, filename)
        imageio.imwrite(full_path, (rgb_image * 255).astype(np.uint8))

    print(f"Saved {len(centroids.T)} cropped liposome images to {folder}")

    return adjusted_LumenPixels, adjusted_BorderPixels
