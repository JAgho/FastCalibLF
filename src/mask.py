import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label

def make_mask(input_file, percentile, closing):
    """
    Parameters:
    input_file : str Path to .npy file of shape (num_echoes, X, Y, Z)
    percentile : int Keep only voxels brighter than {percentile} % of the distribution.
    closing : int Close shapes that are {closing} voxels from being closed

    Example
    make_mask("image.npy", 70, 3)
    """

    img = np.load(input_file) # Shape: (E, X, Y, Z)

    if np.iscomplexobj(img):
        img = np.abs(img)


    masks = np.zeros_like(img, dtype=np.uint8)

    for e in range(img.shape[0]):

        volume = img[e]
        nonzero = volume[volume > 0]

        threshold = np.percentile(nonzero, percentile)
        print(f"Echo {e}: threshold = {threshold:.3f}")

        # Initial mask
        mask = volume > threshold

        # Keep largest connected component
        labels, num = label(mask)

        if num > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest = np.argmax(sizes)
            mask = labels == largest

        # closing
        mask = binary_closing(mask, iterations=closing)

        # Fill holes
        mask = binary_fill_holes(mask)
        masks[e] = mask.astype(np.uint8)

    return masks
