import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.ndimage import label

def make_mask(input_file, output_file):
    """
    Parameter: 
    input_file : input location
    output_file : output location

    Returns:
    output_file : in the specified location
    The file's location

    Run make_mask("/cubric/data/c21122708/HCPData/James_data/JamesSNRAware/mag_denoised2.nii", "/cubric/data/c21122708/HCPData/James_data/JamesSNRAware/mag_denoised1_mask.nii")
    """
    nii = nib.load(input_file)
    img = nii.get_fdata()
    
    percentile = 70 #percentile thresh
    closing = 3 #controls closing holes
    
    
    # Compute percentile threshold
    # Ignore background zeros
    nonzero = img[img > 0]
    threshold = np.percentile(nonzero, percentile)
    print(f"Threshold = {threshold:.3f}")
    
    
    # Initial mask
    mask = img > threshold
    
    # Keep only largest component
    labels, num = label(mask)
    
    if num > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0  # Ignore background
        largest = np.argmax(sizes)
        mask = labels == largest
    
    
    # Close gaps up to {closing} voxels
    mask = binary_closing(mask, closing)
    
    # Now fill holes
    mask = binary_fill_holes(mask)
    mask = mask.astype(np.uint8)
    
    # Save mask
    mask_nii = nib.Nifti1Image(mask, nii.affine, nii.header)
    nib.save(mask_nii, output_file)
    
    print(output_file)