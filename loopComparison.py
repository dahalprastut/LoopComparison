import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def process_byte_file(byte_file_path, aia_file_path, volume_size=(512, 512, 512)):
    """
    Process a single byte file and compare with AIA FITS image
    
    Parameters:
    -----------
    byte_file_path : str
        Path to the byte file
    aia_file_path : str
        Path to the AIA FITS file
    volume_size : tuple, optional
        Size of the 3D volume (default: 512x512x512)
    
    Returns:
    --------
    dict : Dictionary containing analysis results
    """
    # Read the byte file
    volume = np.fromfile(byte_file_path, dtype=np.uint8).reshape(volume_size)
    
    # Project along Y-axis
    slice_2d = np.max(volume, axis=1)
    slice_2d_rot90 = np.rot90(slice_2d, 1)
    
    # Load AIA FITS file
    aia_file = fits.open(aia_file_path)
    aia_data = aia_file[0].data
    aia_file.close()
    
    # Normalize the data
    aia_normalized = (aia_data - aia_data.min()) / (aia_data.max() - aia_data.min())
    slice_normalized = (slice_2d_rot90 - slice_2d_rot90.min()) / (slice_2d_rot90.max() - slice_2d_rot90.min())
    
    # Calculate differences
    absolute_diff = np.abs(aia_normalized - slice_normalized)
    total_absolute_diff = np.sum(absolute_diff)
    total_pixels = aia_data.size
    average_absolute_diff = total_absolute_diff / total_pixels
    
    # Extract alpha value from file path
    alpha_match = [x for x in byte_file_path.split('_') if x.startswith('[') and x.endswith(']')]
    alpha = float(alpha_match[0][1:-1]) if alpha_match else None
    
    return {
        'alpha': alpha,
        'total_absolute_diff': total_absolute_diff,
        'average_absolute_diff': average_absolute_diff,
        'total_pixels': total_pixels,
        'slice_2d_rot90': slice_2d_rot90,
        'absolute_diff': absolute_diff,
        'aia_data': aia_data
    }

def process_multiple_byte_files(files, aia_file_path, output_dir):
    """
    Process multiple byte files and generate report and visualizations
    
    Parameters:
    -----------
    files : list
        List of byte file paths
    aia_file_path : str
        Path to the AIA FITS file
    output_dir : str
        Directory to save output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    results = []
    for file in files:
        try:
            result = process_byte_file(file, aia_file_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Sort results by alpha value
    results.sort(key=lambda x: x['alpha'])
    
    # Generate report
    report_path = os.path.join(output_dir, 'alpha_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("Alpha Comparison Report\n")
        f.write("=" * 30 + "\n")
        for result in results:
            f.write(f"Alpha: {result['alpha']}\n")
            f.write(f"Total Absolute Difference: {result['total_absolute_diff']}\n")
            f.write(f"Average Absolute Difference: {result['average_absolute_diff']}\n")
            f.write(f"Total Pixels: {result['total_pixels']}\n")
            f.write("-" * 30 + "\n")
    
    # Visualization 1: Alpha vs Average Absolute Difference
    plt.figure(figsize=(12, 6))
    alphas = [r['alpha'] for r in results]
    avg_diffs = [r['average_absolute_diff'] for r in results]
    
    # Scatter plot with labels
    plt.scatter(alphas, avg_diffs, marker='o')
    
    # Annotate each point with its exact alpha value
    for i, (alpha, avg_diff) in enumerate(zip(alphas, avg_diffs)):
        plt.annotate(f'α = {alpha}', 
                     (alpha, avg_diff), 
                     xytext=(5, 5), 
                     textcoords='offset points',
                     fontsize=8)
    
    plt.title('Alpha Values vs Average Absolute Difference')
    plt.xlabel('Alpha Value')
    plt.ylabel('Average Absolute Difference')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_vs_avg_diff.png'))
    plt.close()
    
    # Visualization 2: Detailed Comparison Plots
    # Determine number of columns (4 in this case: AIA, Volume Proj, Abs Diff, Text)
    plt.figure(figsize=(20, 5 * len(results)))
    for i, result in enumerate(results, 1):
        # Original AIA Image
        plt.subplot(len(results), 4, 4*i-3)
        plt.imshow(result['aia_data'], vmin=-500.0, vmax=1500.0, cmap='gray', origin='lower')
        plt.title(f"Alpha {result['alpha']}: AIA Image")
        plt.axis('off')
        
        # 3D Volume Projection
        plt.subplot(len(results), 4, 4*i-2)
        plt.imshow(result['slice_2d_rot90'], cmap='gray')
        plt.title(f"Alpha {result['alpha']}: 3D Volume Projection")
        plt.axis('off')
        
        # Absolute Difference
        plt.subplot(len(results), 4, 4*i-1)
        plt.imshow(result['absolute_diff'], cmap='hot')
        plt.title(f"Alpha {result['alpha']}: Absolute Difference")
        plt.axis('off')
        
        # Textual Information
        plt.subplot(len(results), 4, 4*i)
        plt.text(0.5, 0.5, 
                 f"Alpha: {result['alpha']}\n"
                 f"Total Abs Diff: {result['total_absolute_diff']:.2f}\n"
                 f"Avg Abs Diff: {result['average_absolute_diff']:.6f}", 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_comparison.png'))
    plt.close()
    
    print(f"Report and visualizations saved in {output_dir}")

# Example usage
base_dir = "data/2023.01.10"
aia_file_path = os.path.join(base_dir, "aia.2023.01.10.fits")
fileName = "50_30_20_vol_12_5_3"
output_dir = os.path.join(base_dir, f"output/Mag_Field_5_600_S/{fileName}")


# List of byte files (replace with your actual file paths)
files = [
    os.path.join(base_dir, f"Mag_Field_5_[-0.002]_600_S/{fileName}_nJO1C/vol_12_5_3_nJO1C.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[-0.004]_600_S/{fileName}_WHiVo/vol_12_5_3_WHiVo.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[-0.006]_600_S/{fileName}_abADx/vol_12_5_3_abADx.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[-0.008]_600_S/{fileName}_i0gW3/vol_12_5_3_i0gW3.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[-0.012]_600_S/{fileName}_1UgyD/vol_12_5_3_1UgyD.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[-0.01]_600_S/{fileName}_axjwK/vol_12_5_3_axjwK.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0.002]_600_S/{fileName}_KrWv1/vol_12_5_3_KrWv1.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0.004]_600_S/{fileName}_88sgx/vol_12_5_3_88sgx.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0.006]_600_S/{fileName}_23QtF/vol_12_5_3_23QtF.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0.008]_600_S/{fileName}_KzvL7/vol_12_5_3_KzvL7.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0.012]_600_S/{fileName}_JUaR3/vol_12_5_3_JUaR3.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0.01]_600_S/{fileName}_IKQDe/vol_12_5_3_IKQDe.byte"),
    os.path.join(base_dir, f"Mag_Field_5_[0]_600_S/{fileName}_8XOqR/vol_12_5_3_8XOqR.byte"),
]

# Run the analysis
process_multiple_byte_files(files, aia_file_path, output_dir)