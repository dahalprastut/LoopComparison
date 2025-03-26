import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Step 1: Load the 3D byte file
volume_size = (512, 512, 512)  # Assuming 512x512x512 volume
byte_file_path = "data/vol_12_5_3_Ryx2T.byte"  # Replace with your byte file path

# Read the byte file as a 3D numpy array
volume = np.fromfile(byte_file_path, dtype=np.uint8).reshape(volume_size)

# Step 2: Select the appropriate 2D slice
# In the 2D plot, X = img_mf[0], Y = img_mf[1]
# In the 3D volume: X = img_mf[0], Y = img_mf[2], Z = img_mf[1]
# We want the XZ plane (Volume X vs. Volume Z) at a specific Volume Y (original Z)
# Choose a Y-slice that best represents the coronal loops (e.g., middle of volume)
# y_slice_idx = 0  # Example: middle of the volume (adjust as needed)
# slice_2d = volume[:, y_slice_idx, :]  # Shape: (512, 512), XZ plane

# Alternative: If a single slice doesn’t work, project along Y-axis (original Z)
slice_2d = np.max(volume, axis=1)  # Max projection along Y-axis
slice_2d_rot90 = np.rot90(slice_2d, 1)

# Step 3: Load and preprocess the AIA FITS file
aia_file_path = "data/aia.2023.01.10.fits"  # Replace with your AIA FITS file path
aia_file = fits.open(aia_file_path)
aia_data = aia_file[0].data  # e.g., 512*512


# Step 4: Normalize the data (optional, for fair comparison)
# AIA and 3D slice have different scales, normalize to [0, 1]
aia_normalized = (aia_data - aia_data.min()) / (aia_data.max() - aia_data.min())
slice_normalized = (slice_2d_rot90 - slice_2d_rot90.min()) / (slice_2d_rot90.max() - slice_2d_rot90.min())


# Step 5: Calculate pixel-wise absolute differences
absolute_diff = np.abs(aia_normalized - slice_normalized)  # Pixel-wise absolute difference
total_absolute_diff = np.sum(absolute_diff)  # Sum of all differences
total_pixels = aia_data.size  # Total number of pixels (512 * 512)
average_absolute_diff = total_absolute_diff / total_pixels  # Average difference per pixel



# Step 6: Plot for visual confirmation
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(aia_data,vmin=-500.0, vmax=1500.0, cmap='gray', origin='lower')
plt.title("AIA Image")
plt.axis('off')

plt.subplot(132)
plt.imshow(slice_2d_rot90, cmap='gray')
plt.title("3D Volume Projection (Rotated 90°)")
plt.axis('off')

plt.subplot(133)
plt.imshow(absolute_diff, cmap='hot')
plt.title("Absolute Difference")
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 7: Print results
print(f"Total Absolute Difference: {total_absolute_diff}")
print(f"Average Absolute Difference: {average_absolute_diff}")
print(f"Total Pixels: {total_pixels}")