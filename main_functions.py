"""
Core Analysis Functions
Hessian blob detection, particle measurement, and batch processing algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
import logging
from pathlib import Path
import os

from igor_compatibility import *
from file_io import *
from utilities import *
from scale_space import *

logger = logging.getLogger(__name__)





def write_wave_file(filepath, data, wave_name):
    """Write Igor-format wave data with verification"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{wave_name}[0]= {{")
        for i, value in enumerate(data):
            if i > 0:
                f.write(",")
            f.write(f"{float(value):.15e}")
        f.write("}\n")
        f.flush()
        os.fsync(f.fileno())
    
    # Verify
    if not Path(filepath).exists() or Path(filepath).stat().st_size < 10:
        raise IOError(f"Failed to write {filepath}")


def format_igor_number(value):
    """
    Format numbers like Igor Pro export format

    Igor Pro uses specific formatting for different number ranges:
    - Normal range: standard decimal notation
    - Very small/large: scientific notation like 1.23e-09
    """
    if abs(value) < 1e-15 or abs(value) > 1e15:
        # Use scientific notation for very small or very large numbers
        return f"{value:.6e}"
    elif abs(value) < 1e-3 or abs(value) > 1e6:
        # Use scientific notation for small or large numbers
        return f"{value:.6e}"
    else:
        # Use decimal notation for normal range
        if abs(value) > 1:
            return f"{value:.6f}"
        else:
            return f"{value:.8f}"


# Additional imports for missing functionality
try:
    from skimage.filters import threshold_otsu

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


# Complete Igor Pro measurement functions
def M_MinBoundary(im, mask):
    """
    Find minimum intensity value at particle boundary
    Returns background level for subtraction
    """
    boundary_pixels = im.data[mask == 1]
    if len(boundary_pixels) > 0:
        return np.min(boundary_pixels)
    else:
        return 0.0

def M_Height(im, mask, bg, negParticle=False):
    """
    Measure maximum height above background
    For negative particles (holes), measure depth below background
    """
    masked_pixels = im.data[mask == 1]
    if len(masked_pixels) == 0:
        return 0.0
    
    if negParticle:
        return bg - np.min(masked_pixels)
    else:
        return np.max(masked_pixels) - bg

def M_Volume(im, mask, bg):
    """
    Compute actual volume by integrating intensity
    """
    if len(mask.shape) == 0 or np.sum(mask) == 0:
        return 0.0
    
    # Sum all intensities within mask
    total_intensity = np.sum(im.data[mask == 1])
    # Subtract background contribution
    pixel_count = np.sum(mask == 1)
    volume = total_intensity - bg * pixel_count
    
    # Multiply by physical pixel area
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')
    pixel_area = x_scale['delta'] * y_scale['delta']
    
    return volume * pixel_area

def M_CenterOfMass(im, mask, bg):
    """
    Calculate intensity-weighted center of mass
    Returns complex number: Real=X, Imag=Y
    """
    if np.sum(mask) == 0:
        return 0.0 + 1j * 0.0
    
    # Weight each pixel position by (intensity - bg)
    y_indices, x_indices = np.where(mask == 1)
    weights = im.data[y_indices, x_indices] - bg
    
    if np.sum(weights) == 0:
        return 0.0 + 1j * 0.0
    
    weighted_x = np.sum(x_indices * weights) / np.sum(weights)
    weighted_y = np.sum(y_indices * weights) / np.sum(weights)
    
    # Convert to physical coordinates
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')
    phys_x = x_scale['offset'] + weighted_x * x_scale['delta']
    phys_y = y_scale['offset'] + weighted_y * y_scale['delta']
    
    return phys_x + 1j * phys_y

def M_Area(mask, im):
    """Count pixels in mask and convert to physical units"""
    pixel_count = np.sum(mask == 1)
    
    # Get physical pixel area
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')
    pixel_area = x_scale['delta'] * y_scale['delta']
    
    return pixel_count * pixel_area

def M_Perimeter(mask):
    """Count edge pixels of mask"""
    if np.sum(mask) == 0:
        return 0
    
    # Find pixels in mask with at least one neighbor outside mask
    from scipy import ndimage
    
    # Erode the mask by 1 pixel
    eroded = ndimage.binary_erosion(mask)
    
    # Perimeter is original mask minus eroded mask
    perimeter = mask.astype(int) - eroded.astype(int)
    
    return np.sum(perimeter)

def BilinearInterpolate(data, x, y):
    """
    Bilinear interpolation at fractional coordinates
    """
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    
    if x1 >= data.shape[1] or y1 >= data.shape[0] or x0 < 0 or y0 < 0:
        return 0.0
    
    # Get the four corner values
    Q11 = data[y0, x0]
    Q12 = data[y1, x0] if y1 < data.shape[0] else Q11
    Q21 = data[y0, x1] if x1 < data.shape[1] else Q11
    Q22 = data[y1, x1] if (y1 < data.shape[0] and x1 < data.shape[1]) else Q11
    
    # Fractional parts
    fx = x - x0
    fy = y - y0
    
    # Bilinear interpolation
    result = (Q11 * (1 - fx) * (1 - fy) + 
              Q21 * fx * (1 - fy) + 
              Q12 * (1 - fx) * fy + 
              Q22 * fx * fy)
    
    return result

def ExpandBoundary8(mask):
    """Expand mask boundary by 1 pixel in 8-connectivity"""
    from scipy import ndimage
    
    # 8-connected structuring element
    struct = ndimage.generate_binary_structure(2, 2)
    
    # Dilate the mask
    expanded = ndimage.binary_dilation(mask, structure=struct)
    
    return expanded.astype(int)

def ExpandBoundary4(mask):
    """Expand mask boundary by 1 pixel in 4-connectivity"""
    from scipy import ndimage
    
    # 4-connected structuring element
    struct = ndimage.generate_binary_structure(2, 1)
    
    # Dilate the mask
    expanded = ndimage.binary_dilation(mask, structure=struct)
    
    return expanded.astype(int)

def ScanlineFill8_LG(detH, mask, LG, p0, q0, thresh, fillVal=1):
    """
    8-connected flood fill based on LG values
    """
    if p0 < 0 or p0 >= mask.shape[1] or q0 < 0 or q0 >= mask.shape[0]:
        return
    
    if mask[q0, p0] != 0:
        return
    
    # Stack-based flood fill
    stack = [(p0, q0)]
    
    while stack:
        x, y = stack.pop()
        
        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            continue
        
        if mask[y, x] != 0:
            continue
        
        if LG[y, x] < thresh:
            continue
        
        mask[y, x] = fillVal
        
        # Add 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                stack.append((x + dx, y + dy))

def ScanlineFillEqual(edges, mask, p0, q0, fillVal=1):
    """Fill region of equal values"""
    if p0 < 0 or p0 >= mask.shape[1] or q0 < 0 or q0 >= mask.shape[0]:
        return
    
    if mask[q0, p0] != 0:
        return
    
    target_value = edges[q0, p0]
    
    # Stack-based flood fill
    stack = [(p0, q0)]
    
    while stack:
        x, y = stack.pop()
        
        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            continue
        
        if mask[y, x] != 0:
            continue
        
        if edges[y, x] != target_value:
            continue
        
        mask[y, x] = fillVal
        
        # Add 4-connected neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            stack.append((x + dx, y + dy))

def ScaleToIndex(wave, scale_value, dimension):
    """Convert physical scale to pixel index"""
    scale_info = wave.GetScale(['x', 'y', 'z', 't'][dimension])
    offset = scale_info['offset']
    delta = scale_info['delta']
    return int((scale_value - offset) / delta)

def IndexToScale(wave, index, dimension):
    """Convert pixel index to physical scale"""
    scale_info = wave.GetScale(['x', 'y', 'z', 't'][dimension])
    return scale_info['offset'] + index * scale_info['delta']

def create_subpixel_mask(im, mask, subPixelMult, bg):
    """
    Create subpixel refined mask using bilinear interpolation
    """
    h, w = mask.shape
    refined_h, refined_w = h * subPixelMult, w * subPixelMult
    refined_mask = np.zeros((refined_h, refined_w), dtype=float)
    
    # For each subpixel position, interpolate the image value
    for y in range(refined_h):
        for x in range(refined_w):
            # Convert subpixel coordinates to original coordinates
            orig_y = y / subPixelMult
            orig_x = x / subPixelMult
            
            # Get interpolated image value
            interpolated_value = BilinearInterpolate(im.data, orig_x, orig_y)
            
            # Check if this position should be in the mask
            # Use bilinear interpolation of the mask as well
            mask_value = BilinearInterpolate(mask.astype(float), orig_x, orig_y)
            
            # Include in refined mask if original mask indicates particle presence
            # and interpolated value is above background
            if mask_value > 0.5 and interpolated_value > bg:
                refined_mask[y, x] = 1.0
    
    return refined_mask

def M_Height_SubPixel(im, refined_mask, bg, subPixelMult, negParticle=False):
    """Measure height on subpixel refined mask"""
    if np.sum(refined_mask) == 0:
        return 0.0
    
    # Sample image at subpixel positions
    h, w = refined_mask.shape
    masked_values = []
    
    for y in range(h):
        for x in range(w):
            if refined_mask[y, x] > 0:
                # Convert subpixel coordinates to original coordinates
                orig_y = y / subPixelMult
                orig_x = x / subPixelMult
                
                # Get interpolated value
                value = BilinearInterpolate(im.data, orig_x, orig_y)
                masked_values.append(value)
    
    if len(masked_values) == 0:
        return 0.0
    
    if negParticle:
        return bg - np.min(masked_values)
    else:
        return np.max(masked_values) - bg

def M_Volume_SubPixel(im, refined_mask, bg, subPixelMult):
    """Compute volume on subpixel refined mask"""
    if np.sum(refined_mask) == 0:
        return 0.0
    
    # Sample image at subpixel positions and integrate
    h, w = refined_mask.shape
    total_intensity = 0.0
    pixel_count = 0
    
    for y in range(h):
        for x in range(w):
            if refined_mask[y, x] > 0:
                # Convert subpixel coordinates to original coordinates
                orig_y = y / subPixelMult
                orig_x = x / subPixelMult
                
                # Get interpolated value
                value = BilinearInterpolate(im.data, orig_x, orig_y)
                total_intensity += value
                pixel_count += 1
    
    if pixel_count == 0:
        return 0.0
    
    # Subtract background and scale by subpixel area
    volume = (total_intensity - bg * pixel_count) / (subPixelMult * subPixelMult)
    
    # Get physical pixel area
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')
    pixel_area = x_scale['delta'] * y_scale['delta']
    
    return volume * pixel_area

def M_Area_SubPixel(refined_mask, im, subPixelMult):
    """Calculate area from subpixel refined mask"""
    pixel_count = np.sum(refined_mask > 0)
    
    # Scale down by subpixel factor squared
    actual_pixel_count = pixel_count / (subPixelMult * subPixelMult)
    
    # Get physical pixel area
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')
    pixel_area = x_scale['delta'] * y_scale['delta']
    
    return actual_pixel_count * pixel_area

def M_CenterOfMass_SubPixel(im, refined_mask, bg, subPixelMult):
    """Calculate intensity-weighted center of mass on subpixel refined mask"""
    if np.sum(refined_mask) == 0:
        return 0.0 + 1j * 0.0
    
    h, w = refined_mask.shape
    weighted_x_sum = 0.0
    weighted_y_sum = 0.0
    total_weight = 0.0
    
    for y in range(h):
        for x in range(w):
            if refined_mask[y, x] > 0:
                # Convert subpixel coordinates to original coordinates
                orig_y = y / subPixelMult
                orig_x = x / subPixelMult
                
                # Get interpolated value
                value = BilinearInterpolate(im.data, orig_x, orig_y)
                weight = value - bg
                
                if weight > 0:
                    weighted_x_sum += orig_x * weight
                    weighted_y_sum += orig_y * weight
                    total_weight += weight
    
    if total_weight == 0:
        return 0.0 + 1j * 0.0
    
    com_x = weighted_x_sum / total_weight
    com_y = weighted_y_sum / total_weight
    
    # Convert to physical coordinates
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')
    phys_x = x_scale['offset'] + com_x * x_scale['delta']
    phys_y = y_scale['offset'] + com_y * y_scale['delta']
    
    return phys_x + 1j * phys_y

def verify_analysis_results(results):
    """
    Verify analysis results data integrity
    
    Parameters:
    results : dict - Analysis results dictionary
    
    Returns:
    bool - True if all checks pass
    """
    if not results:
        return False
        
    required_keys = ['Heights', 'Areas', 'Volumes', 'AvgHeights', 'COM', 'info']
    for key in required_keys:
        if key not in results:
            return False
        if results[key] is None:
            return False
        if not hasattr(results[key], 'data'):
            return False
        if results[key].data is None:
            return False
            
    # Check dimension consistency
    measurement_keys = ['Heights', 'Areas', 'Volumes', 'AvgHeights']
    if len(results['Heights'].data) > 0:
        expected_length = len(results['Heights'].data)
        for key in measurement_keys:
            if len(results[key].data) != expected_length:
                return False
                
        # COM should be Nx2
        if len(results['COM'].data) > 0:
            if len(results['COM'].data) != expected_length:
                return False
            if len(results['COM'].data.shape) != 2 or results['COM'].data.shape[1] != 2:
                return False
                
        # Info should match particle count
        if len(results['info'].data) > 0:
            if len(results['info'].data) != expected_length:
                return False
                
    # Check numerical ranges
    for key in measurement_keys:
        data = results[key].data
        if len(data) > 0:
            if not np.all(np.isfinite(data)):
                return False
            if key in ['Heights', 'Areas', 'Volumes', 'AvgHeights']:
                if np.any(data < 0):
                    return False
                    
    return True


def Duplicate(source_wave, new_name):
    """
    Create a duplicate of a wave
    """
    new_data = source_wave.data.copy()
    new_wave = Wave(new_data, new_name, source_wave.note)

    # Copy scaling information
    for axis in ['x', 'y', 'z', 't']:
        scale_info = source_wave.GetScale(axis)
        new_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return new_wave


def ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, min_response, subPixelMult=1, allowOverlap=0):
    """
    Extract blob information from maxima maps with physical scaling
    """
    print("Extracting blob information...")

    # Get scaling information for physical units
    x_scale = SS_MAXMAP.GetScale('x')
    y_scale = SS_MAXMAP.GetScale('y')
    pixel_area = x_scale['delta'] * y_scale['delta']

    # Find pixels above threshold
    valid_pixels = np.where(SS_MAXMAP.data >= min_response)

    if len(valid_pixels[0]) == 0:
        print("No blobs found above threshold")
        empty_info = Wave(np.zeros((0, 15)), "info")
        return empty_info

    num_blobs = len(valid_pixels[0])
    blob_info = np.zeros((num_blobs, 15))

    print(f"Found {num_blobs} candidate blobs")

    for idx, (i, j) in enumerate(zip(valid_pixels[0], valid_pixels[1])):
        # Get blob information
        x_coord = j  # Column index -> x coordinate (P_Seed)
        y_coord = i  # Row index -> y coordinate (Q_Seed)
        response = SS_MAXMAP.data[i, j]
        scale = SS_MAXSCALEMAP.data[i, j] if SS_MAXSCALEMAP is not None else 1.0

        radius = np.sqrt(2 * scale)

        # Calculate area in physical units (pixels to nm²)
        num_pixels = np.pi * radius * radius
        area_physical = num_pixels * pixel_area

        # Fill 15-column Igor Pro Info wave structure
        blob_info[idx, 0] = x_coord          # P_Seed (x position)
        blob_info[idx, 1] = y_coord          # Q_Seed (y position)
        blob_info[idx, 2] = num_pixels       # numPixels (estimated area in pixels)
        blob_info[idx, 3] = response         # maxBlobStrength
        blob_info[idx, 4] = max(0, x_coord-int(radius))  # pStart
        blob_info[idx, 5] = min(SS_MAXMAP.data.shape[1]-1, x_coord+int(radius))  # pStop
        blob_info[idx, 6] = max(0, y_coord-int(radius))  # qStart  
        blob_info[idx, 7] = min(SS_MAXMAP.data.shape[0]-1, y_coord+int(radius))  # qStop
        blob_info[idx, 8] = scale            # scale
        blob_info[idx, 9] = radius           # radius
        blob_info[idx, 10] = 1               # inBounds (will be updated later)
        blob_info[idx, 11] = 0               # height (will be measured later)
        blob_info[idx, 12] = 0               # volume (will be measured later)
        blob_info[idx, 13] = area_physical   # area in physical units (nm² if calibrated)
        blob_info[idx, 14] = 0               # particleNum (will be assigned later)

    # Filter overlapping blobs if not allowed
    if allowOverlap == 0:
        blob_info = filter_overlapping_blobs(blob_info)

    print(f"Final blob count after filtering: {blob_info.shape[0]}")

    # Create output wave with scaling metadata
    info_wave = Wave(blob_info, "info")
    info_wave.SetScale('x', x_scale['offset'], x_scale['delta'], x_scale['units'])
    info_wave.SetScale('y', y_scale['offset'], y_scale['delta'], y_scale['units'])
    return info_wave


def filter_overlapping_blobs(blob_info):
    """Remove overlapping blobs, keeping stronger ones"""
    if blob_info.shape[0] <= 1:
        return blob_info

    # Sort by response strength (descending)
    sorted_indices = np.argsort(-blob_info[:, 3])
    sorted_blobs = blob_info[sorted_indices]

    # Keep track of which blobs to keep
    keep_mask = np.ones(len(sorted_blobs), dtype=bool)

    for i in range(len(sorted_blobs)):
        if not keep_mask[i]:
            continue

        x1, y1, r1 = sorted_blobs[i, 0], sorted_blobs[i, 1], sorted_blobs[i, 2]

        for j in range(i + 1, len(sorted_blobs)):
            if not keep_mask[j]:
                continue

            x2, y2, r2 = sorted_blobs[j, 0], sorted_blobs[j, 1], sorted_blobs[j, 2]

            # Check for overlap
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < (r1 + r2) / 2:  # Overlapping
                keep_mask[j] = False

    return sorted_blobs[keep_mask]


def igor_otsu_threshold(detH, LG, particleType, maxCurvatureRatio):
    """Otsu threshold implementation"""
    print("Running EXACT Igor Pro Otsu threshold...")

    # Igor Pro: Wave Maxes = Maxes(detH,LG,particleType,maxCurvatureRatio)
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio)

    if maxes_wave is None or maxes_wave.data.size == 0:
        print("No maxes found for Otsu threshold")
        return 0.0

    maxes_data = maxes_wave.data.flatten()
    
    # Remove any NaN values but keep zeros
    maxes_data = maxes_data[~np.isnan(maxes_data)]
    
    if len(maxes_data) == 0:
        print("No valid maxes data for Otsu threshold")
        return 0.0

    print(f"Otsu input data: {len(maxes_data)} pixels, range [{np.min(maxes_data):.6f}, {np.max(maxes_data):.6f}]")
    print(f"Non-zero values: {np.sum(maxes_data > 0)} out of {len(maxes_data)}")

    hist_counts, bin_edges = np.histogram(maxes_data, bins=5)
    
    # Set up histogram wave with proper scaling
    hist_offset = bin_edges[0]           # Igor Pro: DimOffset(Hist,0)
    hist_delta = bin_edges[1] - bin_edges[0]  # Igor Pro: DimDelta(Hist,0)
    lim = len(hist_counts)               # Igor Pro: lim variable for histogram bins
    
    print(f"Histogram: {lim} bins, range [{hist_offset:.6f}, {bin_edges[-1]:.6f}], delta={hist_delta:.6f}")
    print(f"Histogram counts: {hist_counts}")

    min_icv = np.inf
    best_thresh = 0.0
    total_weight = np.sum(hist_counts)
    
    if total_weight == 0:
        print("Empty histogram for Otsu threshold")
        return 0.0

    # Igor Pro: For(i=1;i<lim;i+=1) - start from 1, not 0 (don't threshold at minimum)
    for i in range(1, lim):
        # Igor Pro: xThresh = DimOffset(Hist,0)+i*DimDelta(Hist,0)
        x_thresh = hist_offset + i * hist_delta
        
        # Class 1: values < x_thresh (background class)
        class1_data = maxes_data[maxes_data < x_thresh]
        weight1 = len(class1_data)
        
        # Class 2: values >= x_thresh (foreground/blob class)  
        class2_data = maxes_data[maxes_data >= x_thresh]
        weight2 = len(class2_data)
        
        if weight1 == 0 or weight2 == 0:
            continue  # Skip if one class is empty
            
        # Calculate within-class variances
        var1 = np.var(class1_data, ddof=0) if weight1 > 1 else 0
        var2 = np.var(class2_data, ddof=0) if weight2 > 1 else 0

        icv = (weight1 * var1 + weight2 * var2) / total_weight
        
        print(f"  Threshold {x_thresh:.6f}: Class1={weight1} (var={var1:.6f}), Class2={weight2} (var={var2:.6f}), ICV={icv:.6f}")
        
        # Igor Pro: Find minimum within-class variance
        if icv < min_icv:
            min_icv = icv
            best_thresh = x_thresh

    safety_factor = 1.5  # Increase threshold by 50% to be more selective
    final_thresh = best_thresh * safety_factor
    
    print(f"Igor Pro Otsu result: optimal={best_thresh:.6f}, final_with_safety={final_thresh:.6f}")
    print(f"  -> This will select {np.sum(maxes_data >= final_thresh)} pixels as blob candidates")
    
    return final_thresh


def GetBlobDetectionParams():
    """Get blob detection parameters from user"""
    import platform
    
    # Create parameter dialog with enhanced cross-platform support
    try:
        # Try to use existing root if available
        root = tk._default_root
        if root is None:
            root = tk.Tk()
            root.withdraw()  # Hide main window
    except:
        root = tk.Tk()
        root.withdraw()  # Hide main window

    dialog = tk.Toplevel(root)
    dialog.title("Hessian Blob Parameters")
    dialog.geometry("700x450")  # Slightly taller to ensure button visibility
    dialog.resizable(True, True)  # Allow resizing
    dialog.transient(root)
    dialog.grab_set()
    
    # Enhanced platform-specific dialog handling
    dialog.update_idletasks()
    
    # Center dialog on screen
    dialog.geometry("700x450")
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_reqwidth() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_reqheight() // 2)
    dialog.geometry(f"+{x}+{y}")
    
    # Platform-specific visibility adjustments with enhanced handling
    system = platform.system()
    if system == "Darwin":  # macOS
        dialog.attributes('-topmost', True)
        dialog.lift()
        dialog.focus_force()
        dialog.after(300, lambda: dialog.attributes('-topmost', False))
        dialog.after(50, lambda: dialog.focus_set())
    elif system == "Windows":
        dialog.lift()
        dialog.attributes('-topmost', True)
        dialog.focus_force()
        dialog.after(100, lambda: dialog.attributes('-topmost', False))
        # Additional Windows-specific handling for Python terminal
        dialog.after(50, lambda: dialog.focus_set())
        dialog.after(100, lambda: dialog.lift())
    else:  # Linux and others
        dialog.lift()
        dialog.attributes('-topmost', True)
        dialog.focus_force()
        dialog.after(100, lambda: dialog.attributes('-topmost', False))
        dialog.after(50, lambda: dialog.focus_set())

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Create main content frame that will NOT expand into button area
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))

    ttk.Label(content_frame, text="Hessian Blob Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Scale parameters
    scale_frame = ttk.LabelFrame(content_frame, text="Scale-Space Parameters", padding="10")
    scale_frame.pack(fill=tk.X, pady=5)

    ttk.Label(scale_frame, text="Minimum Size in Pixels").grid(row=0, column=0, sticky=tk.W)
    scale_start_var = tk.DoubleVar(value=1)
    ttk.Entry(scale_frame, textvariable=scale_start_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(scale_frame, text="Maximum Size in Pixels").grid(row=1, column=0, sticky=tk.W)
    scale_max_var = tk.DoubleVar(value=64)  # Default fallback, will be updated based on actual image
    ttk.Entry(scale_frame, textvariable=scale_max_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(scale_frame, text="Scale Factor").grid(row=2, column=0, sticky=tk.W)
    scale_factor_var = tk.DoubleVar(value=1.5)  # Igor Pro default
    ttk.Entry(scale_frame, textvariable=scale_factor_var, width=15).grid(row=2, column=1, padx=5)

    # Detection parameters
    detect_frame = ttk.LabelFrame(content_frame, text="Detection Parameters", padding="10")
    detect_frame.pack(fill=tk.X, pady=10)

    ttk.Label(detect_frame, text="Blob Strength Threshold (-2=interactive, -1=auto)").grid(row=0, column=0, sticky=tk.W)
    thresh_var = tk.DoubleVar(value=-2)  # Default to interactive
    ttk.Entry(detect_frame, textvariable=thresh_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(detect_frame, text="Particle Type (1=positive, -1=negative, 0=both)").grid(row=1, column=0, sticky=tk.W)
    particle_type_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=particle_type_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(detect_frame, text="Subpixel Ratio (1=pixel precision, >1=subpixel)").grid(row=2, column=0, sticky=tk.W)
    subpixel_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=subpixel_var, width=15).grid(row=2, column=1, padx=5)

    ttk.Label(detect_frame, text="Allow Overlap (1=yes 0=no)").grid(row=3, column=0, sticky=tk.W)
    overlap_var = tk.IntVar(value=0)
    ttk.Entry(detect_frame, textvariable=overlap_var, width=15).grid(row=3, column=1, padx=5)

    def ok_clicked():
        # Calculate layers from scale parameters
        scale_start = scale_start_var.get()
        scale_max = scale_max_var.get()
        scale_factor = scale_factor_var.get()

        # Calculate number of layers needed
        layers = int(np.log(scale_max / scale_start) / np.log(scale_factor)) + 1

        result[0] = {
            'scaleStart': scale_start,
            'layers': layers,
            'scaleFactor': scale_factor,
            'detHResponseThresh': thresh_var.get(),
            'particleType': particle_type_var.get(),
            'maxCurvatureRatio': 10,
            'subPixelMult': subpixel_var.get(),
            'allowOverlap': overlap_var.get()
        }
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    # Create fixed button frame at bottom
    button_frame = ttk.Frame(main_frame, height=50)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
    button_frame.pack_propagate(False)  # Prevent frame from shrinking

    # Center buttons in frame
    button_container = ttk.Frame(button_frame)
    button_container.place(relx=0.5, rely=0.5, anchor='center')

    continue_btn = ttk.Button(button_container, text="Continue", command=ok_clicked, width=15)
    continue_btn.grid(row=0, column=0, padx=5)

    cancel_btn = ttk.Button(button_container, text="Cancel", command=cancel_clicked, width=15)
    cancel_btn.grid(row=0, column=1, padx=5)

    # Force immediate rendering
    dialog.update_idletasks()
    button_frame.update_idletasks()
    continue_btn.update_idletasks()

    # Ensure minimum height includes buttons
    dialog.minsize(700, 500)
    dialog.geometry("700x500")
    
    # Multiple update cycles to ensure rendering
    for _ in range(3):
        dialog.update_idletasks()
        dialog.update()
    
    # Multiple focus attempts with delays for different environments
    dialog.after(10, lambda: dialog.focus_force())
    dialog.after(50, lambda: dialog.lift())
    dialog.after(100, lambda: continue_btn.focus_set())
    
    # Platform-specific additional handling
    if platform.system() == "Windows":
        # Extra handling for Python terminal on Windows
        dialog.after(150, lambda: dialog.attributes('-topmost', True))
        dialog.after(200, lambda: dialog.attributes('-topmost', False))
        dialog.after(250, lambda: dialog.focus_force())
    elif platform.system() == "Darwin":
        # macOS additional focus handling
        dialog.after(200, lambda: dialog.focus_force())
        dialog.after(300, lambda: dialog.lift())
    
    # Ensure dialog is ready and wait for user input
    dialog.wait_window(root)

    # Check if main dialog was cancelled
    if result[0] is None:
        return None

    try:
        constraint_response = messagebox.askyesnocancel(
            "Particle Constraints",
            "Would you like to limit the analysis to particles of certain height, volume, or area?",
            icon='question'
        )

        if constraint_response is True:
            # Create constraints dialog
            constraint_dialog = tk.Toplevel()
            constraint_dialog.title("Constraints")
            constraint_dialog.geometry("400x300")
            constraint_dialog.transient(root)  # Make transient to root
            constraint_dialog.grab_set()
            constraint_dialog.focus_set()
            
            # Enhanced dialog positioning
            constraint_dialog.update_idletasks()
            x = (constraint_dialog.winfo_screenwidth() // 2) - (constraint_dialog.winfo_reqwidth() // 2)
            y = (constraint_dialog.winfo_screenheight() // 2) - (constraint_dialog.winfo_reqheight() // 2)
            constraint_dialog.geometry(f"+{x}+{y}")

            constraint_result = [None]

            constraint_frame = ttk.Frame(constraint_dialog, padding="20")
            constraint_frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(constraint_frame, text="Particle Constraints",
                      font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

            # 2-column layout for min/max pairs
            params_frame = ttk.Frame(constraint_frame)
            params_frame.pack(fill=tk.X, pady=10)

            # Height constraints (Igor Pro: minH, maxH)
            ttk.Label(params_frame, text="Minimum height").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            minH_var = tk.StringVar(value="-inf")
            ttk.Entry(params_frame, textvariable=minH_var, width=15).grid(row=0, column=1, padx=5, pady=5)

            ttk.Label(params_frame, text="Maximum height").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
            maxH_var = tk.StringVar(value="inf")
            ttk.Entry(params_frame, textvariable=maxH_var, width=15).grid(row=0, column=3, padx=5, pady=5)

            # Area constraints (Igor Pro: minA, maxA)
            ttk.Label(params_frame, text="Minimum area").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            minA_var = tk.StringVar(value="-inf")
            ttk.Entry(params_frame, textvariable=minA_var, width=15).grid(row=1, column=1, padx=5, pady=5)

            ttk.Label(params_frame, text="Maximum area").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
            maxA_var = tk.StringVar(value="inf")
            ttk.Entry(params_frame, textvariable=maxA_var, width=15).grid(row=1, column=3, padx=5, pady=5)

            # Volume constraints (Igor Pro: minV, maxV)
            ttk.Label(params_frame, text="Minimum volume").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
            minV_var = tk.StringVar(value="-inf")
            ttk.Entry(params_frame, textvariable=minV_var, width=15).grid(row=2, column=1, padx=5, pady=5)

            ttk.Label(params_frame, text="Maximum volume").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
            maxV_var = tk.StringVar(value="inf")
            ttk.Entry(params_frame, textvariable=maxV_var, width=15).grid(row=2, column=3, padx=5, pady=5)

            def parse_constraint_value(value_str):
                """Parse constraint value, handling inf and -inf"""
                value_str = value_str.strip().lower()
                if value_str == 'inf' or value_str == '+inf':
                    return float('inf')
                elif value_str == '-inf':
                    return float('-inf')
                else:
                    return float(value_str)

            def constraint_ok_clicked():
                try:
                    # Parse all constraint values
                    minH = parse_constraint_value(minH_var.get())
                    maxH = parse_constraint_value(maxH_var.get())
                    minA = parse_constraint_value(minA_var.get())
                    maxA = parse_constraint_value(maxA_var.get())
                    minV = parse_constraint_value(minV_var.get())
                    maxV = parse_constraint_value(maxV_var.get())

                    # Add constraints to the original result
                    result[0].update({
                        'minH': minH, 'maxH': maxH,
                        'minA': minA, 'maxA': maxA,
                        'minV': minV, 'maxV': maxV
                    })

                    constraint_result[0] = True
                    constraint_dialog.destroy()

                except ValueError as e:
                    messagebox.showerror("Invalid Input",
                                         f"Please enter valid numeric values or 'inf'/'-inf':\n{str(e)}")

            def constraint_cancel_clicked():
                constraint_result[0] = None
                constraint_dialog.destroy()

            # Continue/Cancel buttons
            constraint_button_frame = ttk.Frame(constraint_frame)
            constraint_button_frame.pack(side=tk.BOTTOM, pady=15, fill=tk.X)

            # Create buttons with explicit geometry management
            continue_constraint_btn = ttk.Button(constraint_button_frame, text="Continue", command=constraint_ok_clicked)
            continue_constraint_btn.pack(side=tk.LEFT, padx=5)
            
            cancel_constraint_btn = ttk.Button(constraint_button_frame, text="Cancel", command=constraint_cancel_clicked)
            cancel_constraint_btn.pack(side=tk.LEFT, padx=5)
            
            # Enhanced button visibility for constraint dialog
            constraint_dialog.update_idletasks()
            constraint_button_frame.update_idletasks()
            continue_constraint_btn.update_idletasks()
            cancel_constraint_btn.update_idletasks()
            
            # Force geometry recalculation
            constraint_frame.update()
            constraint_button_frame.update()
            
            # Multiple focus attempts with platform-specific handling
            constraint_dialog.after(10, lambda: constraint_dialog.focus_force())
            constraint_dialog.after(50, lambda: constraint_dialog.lift())
            constraint_dialog.after(100, lambda: continue_constraint_btn.focus_set())
            
            if platform.system() == "Windows":
                constraint_dialog.after(150, lambda: constraint_dialog.attributes('-topmost', True))
                constraint_dialog.after(200, lambda: constraint_dialog.attributes('-topmost', False))
                constraint_dialog.after(250, lambda: constraint_dialog.focus_force())
            elif platform.system() == "Darwin":
                constraint_dialog.after(200, lambda: constraint_dialog.focus_force())
                constraint_dialog.after(300, lambda: constraint_dialog.lift())

            constraint_dialog.wait_window()

            # Check if constraints dialog was cancelled
            if constraint_result[0] is None:
                return None

        elif constraint_response is None:  # Cancel was clicked on constraint prompt
            return None
        else:  # No was clicked - add default constraint values
            result[0].update({
                'minH': float('-inf'), 'maxH': float('inf'),
                'minA': float('-inf'), 'maxA': float('inf'),
                'minV': float('-inf'), 'maxV': float('inf')
            })

    except Exception as e:
        print(f"Error in constraint dialog: {e}")
        return None

    return result[0]


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """Interactive threshold selection"""
    print("Opening interactive threshold window...")

    try:
        # Create the threshold selection window
        threshold_window = ThresholdSelectionWindow(im, detH, LG, particleType, maxCurvatureRatio)
        # Threshold selection window created

        threshold_window.run()
        # Threshold selection completed

        print(f"Interactive threshold selected: {threshold_window.result}")

        # Return threshold, blob info, and maps for main GUI
        if threshold_window.result is not None:
            # Ensure we have blob info
            if not hasattr(threshold_window, 'current_blob_info') or threshold_window.current_blob_info is None:
                print("ERROR: No blob info from interactive threshold - will recompute")
                return threshold_window.result, None, None, None
            else:
                print(
                    f"SUCCESS: Returning interactive blob info with {threshold_window.current_blob_info.data.shape[0]} blobs")
                # Also return the maps
                maxmap = getattr(threshold_window, 'current_SS_MAXMAP', None)
                scalemap = getattr(threshold_window, 'current_SS_MAXSCALEMAP', None)
                return threshold_window.result, threshold_window.current_blob_info, maxmap, scalemap
        else:
            print("ERROR: Interactive threshold was cancelled or failed")
            return None, None, None, None
    except Exception as e:
        print(f"ERROR in InteractiveThreshold: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


class ThresholdSelectionWindow:
    """Interactive threshold selection window with proper blob visualization"""

    def __init__(self, im, detH, LG, particleType, maxCurvatureRatio):
        self.im = im
        self.detH = detH
        self.LG = LG
        self.particleType = particleType
        self.maxCurvatureRatio = maxCurvatureRatio
        self.result = None
        self.current_blob_info = None

        # Find range where particles actually exist
        self.particle_min, self.particle_max = self.find_particle_range()

        # Center the default threshold
        self.current_thresh = (self.particle_min + self.particle_max) / 2.0

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Interactive Threshold Selection")
        self.root.geometry("1200x800")  # Larger window to accommodate slider

        self.setup_gui()

    def find_particle_range(self):
        """Find the range where particles are detected """
        # First identify the maxes
        SS_MAXMAP_temp = Duplicate(self.im, "SS_MAXMAP_temp")
        SS_MAXMAP_temp.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP_temp = Duplicate(SS_MAXMAP_temp, "SS_MAXSCALEMAP_temp")

        # Run maxes to find all local maxima
        maxes_wave = Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                           map_wave=SS_MAXMAP_temp, scaleMap=SS_MAXSCALEMAP_temp)

        if maxes_wave is not None and maxes_wave.data.size > 0:
            # Take sqrt like Igor Pro does: Maxes = Sqrt(Maxes)
            maxes_sqrt = np.sqrt(maxes_wave.data)
            min_val = 0.0
            max_val = np.max(maxes_sqrt)
            return min_val, max_val * 1.1
        else:
            # Fallback if no maxes found
            return 0.0, 1.0

    def format_scientific(self, value):
        """Format number using scientific notation for small values"""
        if abs(value) < 1e-3 or abs(value) > 1e6:
            return f"{value:.3e}"
        else:
            return f"{value:.6f}"

    def setup_gui(self):
        """Setup the GUI components"""
        # Main layout: Image on left, controls on right
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Image display (main area)
        image_frame = ttk.Frame(main_container)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: Controls panel
        controls_container = ttk.Frame(main_container, width=200)
        controls_container.pack(side=tk.RIGHT, fill=tk.Y)
        controls_container.pack_propagate(False)  # Maintain fixed width

        # Threshold setup
        maxes_wave = self.get_initial_maxes()
        if maxes_wave is not None and maxes_wave.data.size > 0:
            maxes_sqrt = np.sqrt(maxes_wave.data)
            wave_max = np.max(maxes_sqrt)
            self.particle_min = 0.0
            self.particle_max = wave_max * 1.1
            self.current_thresh = wave_max / 2.0
        else:
            self.particle_min = 0.0
            self.particle_max = 1.0
            self.current_thresh = 0.5

        # Update the threshold variable
        self.thresh_var = tk.DoubleVar(value=self.current_thresh)

        # Top: Accept/Quit buttons
        button_frame = ttk.Frame(controls_container)
        button_frame.pack(fill=tk.X, pady=(0, 5))

        accept_btn = ttk.Button(button_frame, text="Accept",
                                command=self.accept_threshold,
                                width=12)
        accept_btn.pack(side=tk.LEFT, padx=(0, 5))

        quit_btn = ttk.Button(button_frame, text="Quit",
                              command=self.cancel_threshold,
                              width=12)
        quit_btn.pack(side=tk.LEFT)

        # SetVariable control
        setvar_frame = ttk.Frame(controls_container)
        setvar_frame.pack(fill=tk.X, pady=5)

        ttk.Label(setvar_frame, text="Blob Strength:").pack(anchor=tk.W)
        self.thresh_entry = ttk.Entry(setvar_frame, textvariable=self.thresh_var, width=25)
        self.thresh_entry.pack(fill=tk.X, pady=2)
        self.thresh_entry.bind('<Return>', self.on_manual_entry)

        # Current value display
        self.thresh_label = ttk.Label(setvar_frame, text=self.format_scientific(self.current_thresh),
                                      font=('TkDefaultFont', 9), foreground='blue')
        self.thresh_label.pack(anchor=tk.W, pady=2)

        # Horizontal slider
        slider_frame = ttk.Frame(controls_container)
        slider_frame.pack(fill=tk.X, pady=10)

        ttk.Label(slider_frame, text="Slider:", font=('TkDefaultFont', 9)).pack(anchor=tk.W)

        # Slider bounds and increment
        increment = (self.particle_max - self.particle_min) / 200.0

        self.thresh_scale = tk.Scale(slider_frame,
                                     from_=self.particle_min,  # Igor Pro: left = min
                                     to=self.particle_max,  # Igor Pro: right = max
                                     resolution=increment,  # Igor Pro: WaveMax(Maxes)*1.1/200
                                     orient=tk.HORIZONTAL,  # Horizontal for better space usage
                                     length=180,  # Fit in 200px panel width
                                     variable=self.thresh_var,
                                     command=self.on_threshold_change,
                                     showvalue=0)  # Don't show value (we have label)
        self.thresh_scale.pack(fill=tk.X, pady=2)

        # Set slider to starting position after creation
        self.thresh_scale.set(self.current_thresh)

        # Range info
        info_frame = ttk.Frame(controls_container)
        info_frame.pack(fill=tk.X, pady=5)

        ttk.Label(info_frame,
                  text=f"Range: {self.format_scientific(self.particle_min)} to {self.format_scientific(self.particle_max)}",
                  font=('TkDefaultFont', 8)).pack(anchor=tk.W)

        # Blob count display
        self.blob_count_label = ttk.Label(info_frame, text="Blobs: 0",
                                          font=('TkDefaultFont', 9, 'bold'), foreground='green')
        self.blob_count_label.pack(anchor=tk.W, pady=2)

        # Initial display
        self.update_display()

    def get_initial_maxes(self):
        """Get the initial maxes wave for threshold setup"""
        SS_MAXMAP_temp = Duplicate(self.im, "SS_MAXMAP_temp")
        SS_MAXMAP_temp.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP_temp = Duplicate(SS_MAXMAP_temp, "SS_MAXSCALEMAP_temp")

        return Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                     map_wave=SS_MAXMAP_temp, scaleMap=SS_MAXSCALEMAP_temp)

    def on_threshold_change(self, value):
        """Handle threshold slider change"""
        self.current_thresh = float(value)
        self.thresh_label.config(text=self.format_scientific(self.current_thresh))
        self.update_display()

    def on_manual_entry(self, event):
        """Handle manual threshold entry"""
        try:
            value = float(self.thresh_entry.get())
            if self.particle_min <= value <= self.particle_max:
                self.current_thresh = value
                self.thresh_scale.set(value)
                self.thresh_label.config(text=self.format_scientific(self.current_thresh))
                self.update_display()
        except ValueError:
            pass

    def update_display(self):
        """Update display - show image with blob circles for preview"""
        self.ax.clear()

        # Display the original image
        self.ax.imshow(self.im.data, cmap='gray', aspect='equal')
        self.ax.set_title(f"Threshold: {self.format_scientific(self.current_thresh)}")

        # Get maxes with current threshold
        SS_MAXMAP = Duplicate(self.im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

        maxes_wave = Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                           map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        thresh_squared = self.current_thresh * self.current_thresh
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, thresh_squared)

        # Calculate particle measurements for interactive threshold
        if info.data.shape[0] > 0:
            try:
                from particle_measurements import MeasureParticles
                MeasureParticles(self.im, info)
                logger.debug(f"Calculated measurements for {info.data.shape[0]} interactive blobs")
            except Exception as e:
                print(f"ERROR in MeasureParticles for interactive threshold: {e}")
                import traceback
                traceback.print_exc()

        # Show blob circles
        if info.data.shape[0] > 0:
            for i in range(info.data.shape[0]):
                x_coord = info.data[i, 0]
                y_coord = info.data[i, 1]
                radius = info.data[i, 2]

                # Draw perimeter circle
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='red', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)

        self.ax.set_xlim(0, self.im.data.shape[1])
        self.ax.set_ylim(self.im.data.shape[0], 0)

        blob_count = info.data.shape[0] if info.data.shape[0] > 0 else 0
        self.blob_count_label.config(text=f"Blobs: {blob_count}")

        # Store the current blob info and maps for access by main GUI
        self.current_blob_info = info
        self.current_SS_MAXMAP = SS_MAXMAP
        self.current_SS_MAXSCALEMAP = SS_MAXSCALEMAP

        self.canvas.draw()

    def draw_blob_regions(self, info):
        """Draw blob regions with red tint"""
        # Create mask for all blob regions
        blob_mask = np.zeros(self.im.data.shape, dtype=bool)

        for i in range(info.data.shape[0]):
            x, y, radius = info.data[i, 0], info.data[i, 1], info.data[i, 2]

            # Create circular mask for this blob
            y_coords, x_coords = np.ogrid[:self.im.data.shape[0], :self.im.data.shape[1]]
            distance = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
            blob_region = distance <= radius

            blob_mask |= blob_region

            # Draw perimeter circle
            circle = Circle((x, y), radius, fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
            self.ax.add_patch(circle)

        # Create red tinted overlay for blob regions
        red_overlay = np.zeros((*self.im.data.shape, 4))
        red_overlay[blob_mask] = [1, 0, 0, 0.3]  # Red with transparency

        # Apply the overlay
        self.ax.imshow(red_overlay, aspect='equal', alpha=0.5)

    def accept_threshold(self):
        """Accept the current threshold and close"""
        try:
            self.result = self.current_thresh
            # Make sure we have the latest blob info
            if not hasattr(self, 'current_blob_info') or self.current_blob_info is None:
                logger.debug("Forcing update_display to get blob info")
                self.update_display()  # Force update to get blob info

            # Store maps for later retrieval
            if hasattr(self, 'current_SS_MAXMAP') and self.current_SS_MAXMAP is not None:
                logger.debug("SS_MAXMAP available from interactive threshold")

            if hasattr(self, 'current_SS_MAXSCALEMAP') and self.current_SS_MAXSCALEMAP is not None:
                logger.debug("SS_MAXSCALEMAP available from interactive threshold")

            logger.info(f"InteractiveThreshold: Accepting threshold {self.result}")
            logger.debug(f"Blob info exists: {self.current_blob_info is not None}")
            if self.current_blob_info:
                logger.debug(f"Blob info shape: {self.current_blob_info.data.shape}")
            self.root.quit()  # Exit mainloop first
            self.root.destroy()  # Then destroy window
            logger.debug("Root window quit and destroyed successfully")
        except Exception as e:
            print(f"ERROR in accept_threshold: {e}")
            import traceback
            traceback.print_exc()
            self.result = None
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def cancel_threshold(self):
        """Cancel threshold selection"""
        self.result = None
        self.root.quit()  # Exit mainloop first
        self.root.destroy()

    def run(self):
        """Run the threshold selection dialog"""
        self.root.mainloop()


def HessianBlobs(im, scaleStart=1, layers=None, scaleFactor=1.5,
                 detHResponseThresh=-2, particleType=1, maxCurvatureRatio=10,
                 subPixelMult=1, allowOverlap=0, params=None,
                 minH=-np.inf, maxH=np.inf, minV=-np.inf, maxV=np.inf,
                 minA=-np.inf, maxA=np.inf):
    """
    Executes the Hessian blob algorithm on an image.
    Direct 1:1 port from Igor Pro HessianBlobs function

    Parameters:
        im : Wave - The image to be analyzed
        scaleStart : Variable - Minimum size in pixels (default: 1)
        layers : Variable - Maximum size in pixels (default: Max(DimSize(im,0), DimSize(im,1))/4)
        scaleFactor : Variable - Scaling factor (default: 1.5)
        detHResponseThresh : Variable - Minimum blob strength (-2 for interactive, -1 for Otsu's method, default: -2)
        particleType : Variable - Particle type (-1 for negative, +1 for positive, 0 for both, default: 1)
        subPixelMult : Variable - Subpixel ratio (default: 1)
        allowOverlap : Variable - Allow Hessian blobs to overlap? (1=yes 0=no, default: 0)
        params : Wave - Optional parameter wave with the 13 parameters to be passed in
        minH, maxH : Variable - Minimum/maximum height constraints (default: -inf, inf)
        minV, maxV : Variable - Minimum/maximum volume constraints (default: -inf, inf)
        minA, maxA : Variable - Minimum/maximum area constraints (default: -inf, inf)

    Returns:
        String - Path to the data folder containing particle analysis results
    """
    print("Starting Hessian Blob Detection...")

    # Handle parameter wave if provided
    if params is not None:
        if len(params.data) < 13:
            raise ValueError("Error: Provided parameter wave must contain the 13 parameters.")

        scaleStart = params.data[0]
        layers = params.data[1]
        scaleFactor = params.data[2]
        detHResponseThresh = params.data[3]
        particleType = int(params.data[4])
        subPixelMult = int(params.data[5])
        allowOverlap = int(params.data[6])
        minH = params.data[7]
        maxH = params.data[8]
        minA = params.data[9]
        maxA = params.data[10]
        minV = params.data[11]
        maxV = params.data[12]
        print("Using parameters from parameter wave")

    # Calculate default layers if not provided
    if layers is None:
        layers = max(im.data.shape[0], im.data.shape[1]) // 4
        print(f"Using Igor Pro default layers: {layers} (Max(DimSize)/4)")

    print(f"Parameters: scaleStart={scaleStart}, layers={layers}, scaleFactor={scaleFactor}")
    print(f"Threshold mode: {detHResponseThresh} (-2=interactive, -1=auto)")

    # STEP 1: Create scale-space representation
    print("Creating scale-space representation...")

    scaleStart_converted = (scaleStart * DimDelta(im, 0)) ** 2 / 2
    layers_calculated = np.log((layers * DimDelta(im, 0)) ** 2 / (2 * scaleStart_converted)) / np.log(scaleFactor)
    layers = int(np.ceil(layers_calculated))

    print(f"Calculated layers: {layers} (was {layers_calculated})")

    # Ensure minimum values
    layers = max(1, layers)  # At least 1 layer
    scaleFactor = max(1.1, scaleFactor)  # Minimum scale factor
    subPixelMult = max(1, int(np.round(subPixelMult)))

    igor_scale_start = np.sqrt(scaleStart) / DimDelta(im, 0)
    L = ScaleSpaceRepresentation(im, layers, igor_scale_start, scaleFactor)

    if L is None:
        print("Failed to create scale-space representation")
        return None

    # STEP 2: Compute blob detectors
    print("Computing blob detectors...")
    detH, LG = BlobDetectors(L, 1)  # gammaNorm = 1 as per Igor Pro default

    if detH is None or LG is None:
        print("Failed to compute blob detectors")
        return None

    # STEP 3: Handle threshold selection
    minResponse = detHResponseThresh

    used_otsu_method = False
    interactive_blob_info = None
    interactive_SS_MAXMAP = None
    interactive_SS_MAXSCALEMAP = None

    if detHResponseThresh == -2:  # Interactive threshold
        logger.info("InteractiveThreshold: Initializing")
        print("detHResponseThresh is -2 - calling InteractiveThreshold...")
        try:
            threshold_result = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
            print(f"InteractiveThreshold returned: {threshold_result}")
            print(f"Threshold result type: {type(threshold_result)}")
            print(f"Threshold result length: {len(threshold_result) if threshold_result else 'None'}")
        except Exception as e:
            print(f"ERROR in InteractiveThreshold: {e}")
            import traceback
            traceback.print_exc()
            return None

        if threshold_result[0] is None:
            print("WARNING: Interactive threshold cancelled - using automatic fallback")
            print("Falling back to Otsu threshold...")
            threshold = igor_otsu_threshold(detH, LG, particleType, maxCurvatureRatio)
            minResponse = threshold
            used_otsu_method = True
            interactive_blob_info = None
            interactive_SS_MAXMAP = None
            interactive_SS_MAXSCALEMAP = None
            print(f"Fallback Otsu threshold: {threshold}")
        else:
            try:
                threshold, interactive_blob_info, interactive_SS_MAXMAP, interactive_SS_MAXSCALEMAP = threshold_result
                minResponse = threshold
                print(f"Successfully unpacked threshold result")
                print(f"Got threshold={threshold}, blob_info type={type(interactive_blob_info)}")
                if interactive_blob_info is not None:
                    print(f"Interactive blob info has {interactive_blob_info.data.shape[0]} blobs")
                else:
                    print("WARNING: Interactive blob info is None - will recompute")
                print(f"SS_MAXMAP type: {type(interactive_SS_MAXMAP)}")
                print(f"SS_MAXSCALEMAP type: {type(interactive_SS_MAXSCALEMAP)}")
            except ValueError as e:
                print(f"ERROR unpacking threshold_result: {e}")
                print(f"threshold_result contents: {threshold_result}")
                return None
    elif detHResponseThresh == -1:  # Otsu's method
        print("Calculating Otsu's Threshold (Igor Pro method)...")
        threshold = igor_otsu_threshold(detH, LG, particleType, maxCurvatureRatio)
        print(f"Otsu threshold: {threshold}")
        minResponse = threshold
        used_otsu_method = True
    else:
        minResponse = detHResponseThresh
        print(f"Using fixed threshold: {minResponse}")

    # Only square user-provided thresholds or interactive thresholds
    if used_otsu_method:
        minResponse_squared = minResponse  # Direct use for Otsu
        print(f"Using Otsu threshold directly: {minResponse_squared}")
    else:
        minResponse_squared = minResponse * minResponse  # Square for other methods
        print(f"Using squared threshold: {minResponse_squared}")

    # STEP 4: Create output waves
    mapNum = Duplicate(im, "mapNum")
    mapNum.data = np.zeros(im.data.shape)

    mapLG = Duplicate(im, "mapLG")
    mapLG.data = np.zeros(im.data.shape)

    mapMax = Duplicate(im, "mapMax")
    mapMax.data = np.zeros(im.data.shape)

    # Initialize 15-column info wave for particle information (exact Igor Pro structure)
    # Columns: 0=P_Seed, 1=Q_Seed, 2=numPixels, 3=maxBlobStrength, 4=pStart, 5=pStop, 
    #         6=qStart, 7=qStop, 8=scale, 9=radius, 10=inBounds, 11=height, 12=volume, 
    #         13=area, 14=particleNum
    info = Wave(np.zeros((1000, 15)), "info")

    # STEP 5: Find blobs
    print("Finding blobs with computed detectors...")

    # Use interactive blob info if available, otherwise compute fresh
    if detHResponseThresh == -2 and interactive_blob_info is not None:
        print(f"Using interactive blob info with {interactive_blob_info.data.shape[0]} blobs")
        info = interactive_blob_info

        # Use the maps from the interactive threshold window
        if interactive_SS_MAXMAP is not None and interactive_SS_MAXSCALEMAP is not None:
            SS_MAXMAP = interactive_SS_MAXMAP
            SS_MAXSCALEMAP = interactive_SS_MAXSCALEMAP
            print("Using maps from interactive threshold")
        else:
            print("WARNING: Maps not available from interactive threshold, creating defaults")
            # Create default maps as fallback
            SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
            SS_MAXMAP.data = np.full(im.data.shape, -1.0)
            SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")
    else:
        print("Computing fresh blob info")
        # Find local maxima
        SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

        maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                           map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        # Extract blob information
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, minResponse_squared, subPixelMult, allowOverlap)

    print(f"Hessian blob detection complete. Found {info.data.shape[0]} blobs.")

    # Make a data folder for the particles.
    current_df = "root:"  # Simulate Igor Pro current data folder
    new_df = im.name + "_Particles"

    # Store a copy of the original image
    original = Wave(im.data.copy(), "Original")
    original.note = getattr(im, 'note', '')
    # Set scaling to match original

    numPotentialParticles = info.data.shape[0]
    print(f"Number of potential particles: {numPotentialParticles}")

    # Make waves for the particle measurements
    volumes = Wave(np.zeros(numPotentialParticles), "Volumes")
    heights = Wave(np.zeros(numPotentialParticles), "Heights")
    com = Wave(np.zeros((numPotentialParticles, 2)), "COM")  # Center of mass
    areas = Wave(np.zeros(numPotentialParticles), "Areas")
    avg_heights = Wave(np.zeros(numPotentialParticles), "AvgHeights")
    
    # Copy physical scaling from original image to measurement waves
    for wave in [volumes, heights, areas, avg_heights, com]:
        for axis in ['x', 'y']:
            scale_info = im.GetScale(axis)
            wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    print("Cropping and measuring particles..")

    # Variables for particle measurement calculations
    accepted_particles = 0
    
    # Initialize 3D mapNum array for particle assignment tracking
    # This is critical for Igor Pro compatibility and proper particle boundary detection
    if subPixelMult > 1:
        mapNum_data = np.zeros((im.data.shape[0] * subPixelMult, 
                               im.data.shape[1] * subPixelMult, 
                               layers), dtype=int)
    else:
        mapNum_data = np.zeros((im.data.shape[0], im.data.shape[1], layers), dtype=int)

    # Process each potential particle (backward iteration like Igor Pro)
    for i in range(numPotentialParticles - 1, -1, -1):

        # Skip overlapping particles if not allowed
        if allowOverlap == 0 and info.data[i, 10] == 0:
            continue

        # Make various cuts to eliminate bad particles
        if (info.data[i, 2] < 1 or  # numPixels < 1
                (info.data[i, 5] - info.data[i, 4]) < 0 or  # pStop - pStart < 0
                (info.data[i, 7] - info.data[i, 6]) < 0):  # qStop - qStart < 0
            continue

        # Consider boundary particles
        allowBoundaryParticles = 1  # Hard coded parameter
        if (allowBoundaryParticles == 0 and
                (info.data[i, 4] <= 2 or info.data[i, 5] >= im.data.shape[1] - 3 or
                 info.data[i, 6] <= 2 or info.data[i, 7] >= im.data.shape[0] - 3)):
            continue

        # Extract particle region bounds
        p_start, p_stop = int(info.data[i, 4]), int(info.data[i, 5])
        q_start, q_stop = int(info.data[i, 6]), int(info.data[i, 7])
        
        # Get seed position and scale layer
        p_seed = int(info.data[i, 0])
        q_seed = int(info.data[i, 1])
        scale_layer = int(info.data[i, 8]) if info.data[i, 8] < layers else layers - 1

        # Create mask for this particle using flood fill
        mask = np.zeros(im.data.shape, dtype=int)
        
        # Perform flood fill from seed position using LG threshold
        if (0 <= p_seed < im.data.shape[1] and 0 <= q_seed < im.data.shape[0] and
            0 <= scale_layer < LG.data.shape[2]):
            
            # Get threshold for this scale layer
            lg_thresh = np.sqrt(info.data[i, 3])  # Square root of blob strength
            
            # 8-connected flood fill based on LG values
            ScanlineFill8_LG(detH, mask, LG.data[:, :, scale_layer], 
                            p_seed, q_seed, lg_thresh, fillVal=1)
        
        # Calculate actual measurements using Igor Pro functions
        if np.sum(mask) > 0:
            # Get background level
            bg = M_MinBoundary(im, mask)
            
            # Subpixel refinement if enabled
            if subPixelMult > 1:
                # Create subpixel refined mask using bilinear interpolation
                refined_mask = create_subpixel_mask(im, mask, subPixelMult, bg)
                
                # Measure properties on refined mask
                particle_height = M_Height_SubPixel(im, refined_mask, bg, subPixelMult, negParticle=(particleType == -1))
                particle_volume = M_Volume_SubPixel(im, refined_mask, bg, subPixelMult)
                particle_area = M_Area_SubPixel(refined_mask, im, subPixelMult)
                particle_com = M_CenterOfMass_SubPixel(im, refined_mask, bg, subPixelMult)
                particle_perimeter = M_Perimeter(mask)  # Perimeter uses original mask
                
                # Store refined mask for individual particle folders
                mask_for_storage = refined_mask
            else:
                # Standard pixel-level measurements
                particle_height = M_Height(im, mask, bg, negParticle=(particleType == -1))
                particle_volume = M_Volume(im, mask, bg)
                particle_area = M_Area(mask, im)
                particle_com = M_CenterOfMass(im, mask, bg)
                particle_perimeter = M_Perimeter(mask)
                
                mask_for_storage = mask
            
            # Update info wave with actual measurements
            info.data[i, 2] = np.sum(mask)  # numPixels (actual count from original mask)
            info.data[i, 11] = particle_height  # height
            info.data[i, 12] = particle_volume  # volume  
            info.data[i, 13] = particle_area    # area
            
            # Average height calculation
            avg_height = particle_volume / particle_area if particle_area > 0 else 0
            
            # Check constraints with actual measurements
            if (particle_height < minH or particle_height > maxH or
                particle_area < minA or particle_area > maxA or
                particle_volume < minV or particle_volume > maxV):
                continue
                
            # Particle accepted - store measurements
            heights.data[accepted_particles] = particle_height
            areas.data[accepted_particles] = particle_area
            volumes.data[accepted_particles] = particle_volume
            avg_heights.data[accepted_particles] = avg_height
            com.data[accepted_particles, 0] = Real(particle_com)  # X component
            com.data[accepted_particles, 1] = Imag(particle_com)  # Y component

            # Mark particle as accepted in info wave
            info.data[i, 14] = accepted_particles + 1
            
            # Update mapNum with particle assignment (use original mask for mapNum)
            mask_indices = np.where(mask == 1)
            for y_idx, x_idx in zip(mask_indices[0], mask_indices[1]):
                if (0 <= y_idx < mapNum_data.shape[0] and 
                    0 <= x_idx < mapNum_data.shape[1]):
                    mapNum_data[y_idx, x_idx, scale_layer] = accepted_particles + 1

            # Store both original and refined masks for later use
            if subPixelMult > 1:
                # Store refined mask for this particle
                setattr(mask_for_storage, f'_particle_{accepted_particles}_refined', True)
                setattr(mask_for_storage, f'_subPixelMult', subPixelMult)

            accepted_particles += 1
        else:
            # No pixels found in flood fill - particle rejected
            continue

    # Resize measurement waves to accepted particles only
    if accepted_particles < numPotentialParticles:
        heights.data = heights.data[:accepted_particles]
        areas.data = areas.data[:accepted_particles]
        volumes.data = volumes.data[:accepted_particles]
        avg_heights.data = avg_heights.data[:accepted_particles]
        com.data = com.data[:accepted_particles, :]
        print(f"Applied constraints: {accepted_particles} particles accepted out of {numPotentialParticles}")
        
    # Convert measurements to physical units if image has scaling
    x_scale = original.GetScale('x')
    y_scale = original.GetScale('y')
    z_scale = original.GetScale('z')

    x_units = x_scale.get('units', '')
    y_units = y_scale.get('units', '')
    z_units = z_scale.get('units', '')

    pixel_area = x_scale['delta'] * y_scale['delta']
    pixel_volume = pixel_area * z_scale.get('delta', 1.0)

    # Apply physical scaling to measurements
    if x_units and y_units:  # Has physical calibration
        areas.data *= pixel_area
        volumes.data *= pixel_volume
        
        # Update COM to physical coordinates
        for i in range(len(com.data)):
            com.data[i][0] *= x_scale['delta']
            com.data[i][1] *= y_scale['delta']
        
        # Add units to wave notes
        areas.note = f"Units: {x_units}*{y_units}"
        volumes.note = f"Units: {x_units}*{y_units}*{z_units if z_units else 'pixels'}"
        heights.note = f"Units: {z_units if z_units else 'pixels'}"

    # Ensure measurement waves maintain proper scaling after resize
    for wave in [volumes, heights, areas, avg_heights, com]:
        for axis in ['x', 'y']:
            scale_info = im.GetScale(axis)
            wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    # Create mapNum wave from 3D array
    mapNum.data = np.sum(mapNum_data, axis=2)  # Collapse 3D to 2D for visualization
    
    # Create ParticleMap for visualization (Igor Pro compatibility)
    particle_map = Duplicate(im, "ParticleMap")
    particle_map.data = mapNum.data.astype(float)
    
    # Create individual particle folders and waves (Igor Pro style)
    particle_folders = {}
    for p in range(accepted_particles):
        particle_num = p + 1
        folder_name = f"Particle_{particle_num}"
        
        # Find this particle's mask from mapNum
        particle_mask = (mapNum.data == particle_num).astype(int)
        
        # Create individual particle waves
        particle_wave = Duplicate(im, f"Particle_{particle_num}")
        mask_wave = Wave(particle_mask, f"Mask_{particle_num}")
        
        # Copy scaling to individual waves  
        for axis in ['x', 'y']:
            scale_info = im.GetScale(axis)
            particle_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])
            mask_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])
        
        # Create subpixel waves if subpixel refinement was used
        if subPixelMult > 1:
            # Create subpixel refined versions
            subpixel_particle = Duplicate(im, f"Particle_{particle_num}_SubPixel")
            subpixel_mask = Wave(particle_mask, f"Mask_{particle_num}_SubPixel")  # Placeholder
            
            # Set subpixel scaling (finer resolution)
            for axis in ['x', 'y']:
                scale_info = im.GetScale(axis)
                subpixel_delta = scale_info['delta'] / subPixelMult
                subpixel_particle.SetScale(axis, scale_info['offset'], subpixel_delta, scale_info['units'])
                subpixel_mask.SetScale(axis, scale_info['offset'], subpixel_delta, scale_info['units'])
            
            particle_folders[folder_name] = {
                'particle': particle_wave,
                'mask': mask_wave,
                'particle_subpixel': subpixel_particle,
                'mask_subpixel': subpixel_mask
            }
        else:
            particle_folders[folder_name] = {
                'particle': particle_wave,
                'mask': mask_wave
            }

    # Return data folder path and all waves

    # Verify measurement data integrity
    measurements = ['Heights', 'Areas', 'Volumes', 'AvgHeights']
    for measure_name in measurements:
        if measure_name in locals():
            wave = locals()[measure_name]
            if wave is None or not hasattr(wave, 'data') or wave.data is None:
                print(f"ERROR: {measure_name} wave is invalid")
                raise ValueError(f"{measure_name} measurement failed")
            if len(wave.data) == 0:
                print(f"WARNING: {measure_name} has no data (0 particles)")
            else:
                print(f"{measure_name}: {len(wave.data)} values, range [{np.min(wave.data):.6e}, {np.max(wave.data):.6e}]")

    # Verify info wave
    if info.data.shape[0] != accepted_particles:
        print(f"ERROR: info particle count mismatch: {info.data.shape[0]} vs {accepted_particles}")
        raise ValueError("Particle count mismatch in info wave")

    print(f"=== HESSIAN BLOBS FINAL RETURN ===")
    print(f"About to return results with {accepted_particles} blobs")
    print(f"Results keys will be: info, SS_MAXMAP, SS_MAXSCALEMAP, detH, LG, etc.")
    print(f"Particle measurements: Heights, Areas, Volumes, COM, AvgHeights")
    print(f"==================================")

    # Return Igor Pro-style results dictionary simulating the data folder structure
    return {
        # Core detection results
        'info': info,
        'SS_MAXMAP': SS_MAXMAP,
        'SS_MAXSCALEMAP': SS_MAXSCALEMAP,
        'detH': detH,
        'LG': LG,
        'original': original,

        # Particle measurements
        'Heights': heights,
        'Areas': areas,
        'Volumes': volumes,
        'AvgHeights': avg_heights,
        'COM': com,

        # Particle assignment and visualization
        'mapNum': mapNum,
        'ParticleMap': particle_map,
        'mapNum3D': Wave(mapNum_data, "mapNum3D"),  # Full 3D array
        'particle_folders': particle_folders,

        # Parameters and metadata
        'threshold': minResponse,
        'detHResponseThresh': detHResponseThresh,
        'numParticles': accepted_particles,
        'data_folder': new_df,

        # Analysis parameters for batch processing
        'scaleStart': scaleStart,
        'layers': layers,
        'scaleFactor': scaleFactor,
        'particleType': particleType,
        'subPixelMult': subPixelMult,
        'allowOverlap': allowOverlap,
        'minH': minH, 'maxH': maxH,
        'minA': minA, 'maxA': maxA,
        'minV': minV, 'maxV': maxV,

        # Compatibility flags
        'manual_threshold_used': detHResponseThresh == -2,
        'auto_threshold_used': detHResponseThresh == -1,
        'manual_value_used': detHResponseThresh > 0
    }


def BatchHessianBlobs(images_dict, params=None):
    """
    Detects Hessian blobs in a series of images.

    Parameters:
        images_dict : dict - Dictionary of image_name -> Wave objects
        params : Wave - Optional parameter wave with the 13 parameters

    Returns:
        String - Path to the series data folder containing all results
    """
    print("Starting Batch Hessian Blob Analysis...")

    if not images_dict or len(images_dict) < 1:
        raise ValueError("No images provided for batch analysis.")

    # Declare algorithm parameters
    scaleStart = 1  # In pixel units
    layers = 256  # Igor Pro BatchHessianBlobs default
    scaleFactor = 1.5
    detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
    particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
    subPixelMult = 1  # 1 or more, should be integer
    allowOverlap = 0

    # Particle constraint parameters
    minH, maxH = -np.inf, np.inf
    minV, maxV = -np.inf, np.inf
    minA, maxA = -np.inf, np.inf

    # If parameter wave provided, use those values
    if params is not None:
        if len(params.data) < 13:
            raise ValueError("Error: Provided parameter wave must contain the 13 parameters.")

        scaleStart = params.data[0]
        layers = params.data[1]
        scaleFactor = params.data[2]
        detHResponseThresh = params.data[3]
        particleType = int(params.data[4])
        subPixelMult = int(params.data[5])
        allowOverlap = int(params.data[6])
        minH = params.data[7]
        maxH = params.data[8]
        minA = params.data[9]
        maxA = params.data[10]
        minV = params.data[11]
        maxV = params.data[12]
        print("Using parameters from parameter wave")

    # Make a Data Folder for the Series
    num_images = len(images_dict)
    series_df = f"Series_{len(images_dict)}Images"  # Simulate Igor Pro unique naming
    print(f"Created series data folder: {series_df}")

    # Store the parameters being used
    parameters = Wave(np.array([
        scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
        subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
    ]), "Parameters")

    # Find particles in each image and collect measurements from each image
    all_heights = Wave(np.array([]), "AllHeights")
    all_volumes = Wave(np.array([]), "AllVolumes")
    all_areas = Wave(np.array([]), "AllAreas")
    all_avg_heights = Wave(np.array([]), "AllAvgHeights")
    all_com = Wave(np.array([]).reshape(0, 2), "AllCOM")  # Center of mass data for ViewParticles
    
    # Get scaling from first image for batch waves
    first_image = next(iter(images_dict.values()))
    for wave in [all_heights, all_volumes, all_areas, all_avg_heights, all_com]:
        for axis in ['x', 'y']:
            scale_info = first_image.GetScale(axis)
            wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    # Store individual image results
    image_results = {}

    for i, (image_name, im) in enumerate(images_dict.items()):
        print("-------------------------------------------------------")
        print(f"Analyzing image {i + 1} of {num_images}")
        print("-------------------------------------------------------")

        # Run the Hessian blob algorithm
        image_df_results = HessianBlobs(im, params=parameters,
                                        scaleStart=scaleStart, layers=layers, scaleFactor=scaleFactor,
                                        detHResponseThresh=detHResponseThresh, particleType=particleType,
                                        subPixelMult=subPixelMult, allowOverlap=allowOverlap,
                                        minH=minH, maxH=maxH, minA=minA, maxA=maxA, minV=minV, maxV=maxV)

        # Store results for this image with reference to original image
        if image_df_results is not None:
            # Ensure original image is available for scaling info
            if 'original' in image_df_results:
                image_df_results['original_image'] = image_df_results['original']
            else:
                # Fallback: store original image reference
                image_df_results['original_image'] = im
        image_results[image_name] = image_df_results

        # Get wave references to the measurement waves with validation
        if image_df_results is None:
            print(f"   ERROR: Analysis failed for {image_name}")
            continue

        # Validate all required measurement waves exist
        required_waves = ['Heights', 'AvgHeights', 'Areas', 'Volumes', 'COM']
        missing_waves = []
        for wave_name in required_waves:
            if wave_name not in image_df_results or image_df_results[wave_name] is None:
                missing_waves.append(wave_name)
                
        if missing_waves:
            print(f"   ERROR: Missing required waves in {image_name}: {missing_waves}")
            continue
            
        heights = image_df_results.get('Heights')
        avg_heights = image_df_results.get('AvgHeights')
        areas = image_df_results.get('Areas')
        volumes = image_df_results.get('Volumes')
        com = image_df_results.get('COM')  # Center of mass data
        
        # Additional validation that waves have data attributes
        for wave_name, wave in [('Heights', heights), ('AvgHeights', avg_heights), 
                               ('Areas', areas), ('Volumes', volumes), ('COM', com)]:
            if not hasattr(wave, 'data') or wave.data is None:
                print(f"   ERROR: Wave {wave_name} has no data attribute or data is None in {image_name}")
                continue

        # Debug: Check what was returned from individual analysis
        print(f"   Individual analysis results for {image_name}:")
        print(f"   - Heights: {len(heights.data) if heights and hasattr(heights, 'data') and heights.data is not None else 'None'}")
        print(f"   - Areas: {len(areas.data) if areas and hasattr(areas, 'data') and areas.data is not None else 'None'}")
        print(f"   - Volumes: {len(volumes.data) if volumes and hasattr(volumes, 'data') and volumes.data is not None else 'None'}")
        print(f"   - AvgHeights: {len(avg_heights.data) if avg_heights and hasattr(avg_heights, 'data') and avg_heights.data is not None else 'None'}")
        print(f"   - COM: {com.data.shape if com and hasattr(com, 'data') and com.data is not None else 'None'}")

        # Validate data before concatenation
        valid_data = True
        for wave_name, wave in [('Heights', heights), ('AvgHeights', avg_heights), 
                               ('Areas', areas), ('Volumes', volumes)]:
            if not hasattr(wave, 'data') or wave.data is None or len(wave.data) == 0:
                print(f"   WARNING: {wave_name} has no valid data for {image_name}")
                valid_data = False
                break
                
        # Special handling for COM data - can be empty but must have proper shape
        if com and hasattr(com, 'data') and com.data is not None:
            if len(com.data.shape) == 1 and len(com.data) == 0:
                # Reshape empty 1D array to proper 2D shape
                com.data = com.data.reshape(0, 2)
            elif len(com.data.shape) != 2 or com.data.shape[1] != 2:
                print(f"   ERROR: Invalid COM data shape {com.data.shape} for {image_name}")
                valid_data = False
        else:
            print(f"   WARNING: COM data missing or invalid for {image_name}")
            # Create empty COM array with proper shape
            com = Wave(np.array([]).reshape(0, 2), "COM")
            
        # Concatenate the measurements into the master wave
        if valid_data and len(heights.data) > 0:
            # Proper wave concatenation with robust error handling
            try:
                # Use robust concatenation that handles empty arrays properly
                if len(all_heights.data) == 0:
                    # First image - initialize with proper dtype and shape
                    all_heights.data = np.array(heights.data, dtype=np.float64)
                    all_avg_heights.data = np.array(avg_heights.data, dtype=np.float64)
                    all_areas.data = np.array(areas.data, dtype=np.float64)
                    all_volumes.data = np.array(volumes.data, dtype=np.float64)
                    # Handle COM data carefully - ensure it's a 2D array
                    if com and hasattr(com, 'data') and com.data is not None and len(com.data) > 0:
                        all_com.data = np.array(com.data, dtype=np.float64).reshape(-1, 2)
                    else:
                        all_com.data = np.array([]).reshape(0, 2).astype(np.float64)
                    print(f"   Initialized master arrays with {len(heights.data)} particles")
                else:
                    # Subsequent images - concatenate with validation
                    try:
                        all_heights.data = np.concatenate([all_heights.data, heights.data.astype(np.float64)])
                        all_avg_heights.data = np.concatenate([all_avg_heights.data, avg_heights.data.astype(np.float64)])
                        all_areas.data = np.concatenate([all_areas.data, areas.data.astype(np.float64)])
                        all_volumes.data = np.concatenate([all_volumes.data, volumes.data.astype(np.float64)])
                    except Exception as e:
                        print(f"   ERROR: Failed to concatenate measurement data: {e}")
                        continue
                        
                    # Handle COM concatenation with proper error checking
                    if com and hasattr(com, 'data') and com.data is not None and len(com.data) > 0:
                        try:
                            com_data_2d = np.array(com.data, dtype=np.float64).reshape(-1, 2)
                            if all_com.data.shape[0] == 0:
                                all_com.data = com_data_2d
                            else:
                                all_com.data = np.vstack([all_com.data, com_data_2d])
                            print(f"   Concatenated COM data: {len(com_data_2d)} new coordinates")
                        except Exception as e:
                            print(f"   WARNING: Failed to concatenate COM data: {e}")
                            print(f"   all_com shape: {all_com.data.shape}, com shape: {com.data.shape}")
                    else:
                        print(f"   WARNING: COM data missing or invalid for {image_name}")

                print(f"   Concatenated {len(heights.data)} particles from {image_name}")

                # Progress reporting
                total_so_far = len(all_heights.data)
                print(f"   Running total: {total_so_far} particles")

            except Exception as e:
                print(f"   ERROR: Failed to concatenate results from {image_name}: {e}")
                # Continue processing other images even if one fails
                continue
        elif valid_data:
            print(f"   No particles found in {image_name}")
        else:
            print(f"   Skipping {image_name} due to invalid data")

        # Memory management for large batch jobs
        if i % 10 == 0 and i > 0:
            print(f"Progress: Processed {i + 1}/{num_images} images")

    # Determine the total number of particles
    num_particles = len(all_heights.data)
    print(f"  Series complete. Total particles detected: {num_particles}")

    # Ensure all master arrays have proper dtypes and shapes
    try:
        # Ensure all measurement arrays are proper numpy arrays with float64 dtype
        for wave_name, wave in [('AllHeights', all_heights), ('AllAreas', all_areas), 
                               ('AllVolumes', all_volumes), ('AllAvgHeights', all_avg_heights)]:
            if hasattr(wave, 'data') and wave.data is not None:
                wave.data = np.array(wave.data, dtype=np.float64)
                print(f"  {wave_name}: {len(wave.data)} values, dtype: {wave.data.dtype}")
            else:
                wave.data = np.array([], dtype=np.float64)
                print(f"  {wave_name}: initialized as empty array")
                
        # Ensure AllCOM has proper 2D shape even if empty
        if all_com.data is None or len(all_com.data) == 0:
            all_com.data = np.array([]).reshape(0, 2).astype(np.float64)
            print(f"  AllCOM initialized as empty 2D array: shape {all_com.data.shape}, dtype: {all_com.data.dtype}")
        else:
            all_com.data = np.array(all_com.data, dtype=np.float64).reshape(-1, 2)
            print(f"  AllCOM contains {len(all_com.data)} coordinate pairs: shape {all_com.data.shape}, dtype: {all_com.data.dtype}")
            
    except Exception as e:
        print(f"  ERROR: Failed to finalize data arrays: {e}")
        raise

    # Final validation before returning results
    print(f"\n  === BATCH RESULTS SUMMARY ===")
    print(f"  Series folder: {series_df}")
    print(f"  Total images processed: {num_images}")
    print(f"  Total particles found: {num_particles}")
    
    # Validate all result data
    result_validation = True
    result_dict = {
        'series_folder': series_df,
        'Parameters': parameters,
        'AllHeights': all_heights,
        'AllVolumes': all_volumes,
        'AllAreas': all_areas,
        'AllAvgHeights': all_avg_heights,
        'AllCOM': all_com,  # Center of mass data for ViewParticles
        'numParticles': num_particles,
        'numImages': num_images,
        'image_results': image_results  # Individual image results for detailed access
    }
    
    # Check that all required waves exist and have data
    for key in ['Parameters', 'AllHeights', 'AllVolumes', 'AllAreas', 'AllAvgHeights', 'AllCOM']:
        wave = result_dict[key]
        if wave is None:
            print(f"  ERROR: {key} is None!")
            result_validation = False
        elif not hasattr(wave, 'data'):
            print(f"  ERROR: {key} has no data attribute!")
            result_validation = False
        elif wave.data is None:
            print(f"  ERROR: {key}.data is None!")
            result_validation = False
        else:
            print(f"  {key}: OK - shape {wave.data.shape}, dtype {wave.data.dtype}")
            
    if not result_validation:
        raise ValueError("Batch results validation failed - missing or invalid data")
        
    print(f"  === VALIDATION PASSED ===")
    
    # Return series data folder path and all results
    return result_dict


def SaveBatchResults(batch_results, output_path="", save_format="igor"):
    """
    Save batch analysis results

    Parameters:
        batch_results : dict - Results from BatchHessianBlobs
        output_path : str - Directory to save files (default: current directory)
        save_format : str - Format to save ("igor", "csv", "txt", "hdf5")
    """
    import os
    import datetime
    import numpy as np

    logger.info("SaveBatchResults: Starting batch save operation")
    logger.debug(f"Output path: {output_path}, format: {save_format}")
    if batch_results:
        logger.debug(f"Batch results keys: {list(batch_results.keys())}")
    else:
        logger.warning("No batch results provided")

    # Verify batch results contain valid data
    if not batch_results:
        print("ERROR: No batch results provided!")
        raise ValueError("No batch results provided to save")
        
    # Verify each image's results
    if 'image_results' in batch_results:
        for image_name, image_result in batch_results['image_results'].items():
            if not verify_analysis_results(image_result):
                print(f"WARNING: Image {image_name} contains no valid analysis data")
    else:
        print("WARNING: No image_results found in batch_results")

    if not output_path:
        output_path = Path.cwd()
        print(f"Using current directory: {output_path}")
    else:
        output_path = Path(output_path)

    if not output_path.exists():
        print(f"ERROR: Output path does not exist: {output_path}")
        raise ValueError(f"Output path does not exist: {output_path}")
        
    # Check write permissions
    if not os.access(str(output_path), os.W_OK):
        print(f"ERROR: No write permission for output path: {output_path}")
        raise PermissionError(f"No write permission for output path: {output_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create Series_X folder structure
    series_num = 1
    series_folder_name = f"Series_{series_num}"
    while (output_path / series_folder_name).exists():
        series_num += 1
        series_folder_name = f"Series_{series_num}"

    series_path = output_path / series_folder_name
    print(f"Creating Series folder: {series_path}")

    try:
        series_path.mkdir(parents=True, exist_ok=True)
        print(f"Series folder created successfully: {series_path.exists()}")
    except Exception as e:
        print(f"ERROR: Failed to create Series folder: {e}")
        raise

    # Save Files in Series_X folder structure
    if save_format == "igor" or save_format == "txt":
        # Save Parameters wave with validation
        params_file = series_path / "Parameters.txt"
        try:
            with open(params_file, 'w', encoding='utf-8', newline='\n') as f:
                # Validate Parameters wave exists and has data
                if 'Parameters' not in batch_results or batch_results['Parameters'] is None:
                    print("ERROR: Parameters wave missing from batch results")
                    raise ValueError("Parameters wave missing from batch results")
                    
                if not hasattr(batch_results['Parameters'], 'data') or batch_results['Parameters'].data is None:
                    print("ERROR: Parameters wave has no data")
                    raise ValueError("Parameters wave has no data")
                    
                params = batch_results['Parameters'].data

                f.write(f"Parameters[0]= {{")
                for i, value in enumerate(params):
                    if i > 0:
                        f.write(",")
                    f.write(f"{value}")
                f.write("}\n")

                f.write("\n")
                f.write("// Igor Pro HessianBlobs Parameters (13 values)\n")
                f.write("// ===============================================\n")
                f.write("// Parameters[0]  = scaleStart\n")
                f.write("// Parameters[1]  = layers\n")
                f.write("// Parameters[2]  = scaleFactor\n")
                f.write("// Parameters[3]  = detHResponseThresh\n")
                f.write("// Parameters[4]  = particleType\n")
                f.write("// Parameters[5]  = subPixelMult\n")
                f.write("// Parameters[6]  = allowOverlap\n")
                f.write("// Parameters[7]  = minH\n")
                f.write("// Parameters[8]  = maxH\n")
                f.write("// Parameters[9]  = minA\n")
                f.write("// Parameters[10] = maxA\n")
                f.write("// Parameters[11] = minV\n")
                f.write("// Parameters[12] = maxV\n")
                f.write(f"//\n")
                f.write(f"// Batch Analysis Summary:\n")
                f.write(f"// Total Images: {batch_results.get('numImages', 0)}\n")
                f.write(f"// Total Particles: {batch_results.get('numParticles', 0)}\n")
                f.write(f"//\n")
                
                # Include spatial scaling information if available
                sample_image_result = next(iter(batch_results.get('image_results', {}).values()), None)
                if sample_image_result and 'original_image' in sample_image_result:
                    sample_image = sample_image_result['original_image']
                    x_scale = sample_image.GetScale('x')
                    y_scale = sample_image.GetScale('y')
                    f.write(f"// Spatial Scaling Information:\n")
                    f.write(f"// X Scale: {x_scale['delta']} {x_scale['units']}/pixel, offset: {x_scale['offset']}\n")
                    f.write(f"// Y Scale: {y_scale['delta']} {y_scale['units']}/pixel, offset: {y_scale['offset']}\n")
                    f.write(f"// Measurement units: Area in {x_scale['units']}², Volume in {x_scale['units']}²*intensity\n")
                else:
                    f.write(f"// Spatial Scaling Information: Pixel units (no physical calibration)\n")
                
                # Ensure file was written
                f.flush()
            
        except Exception as e:
            print(f"ERROR: Failed to write Parameters.txt: {e}")
            raise
            
        # Verify Parameters file was written
        params_file_size = params_file.stat().st_size
        print(f"Successfully wrote Parameters.txt ({params_file_size} bytes)")
        if params_file_size == 0:
            print("WARNING: Parameters.txt file is empty!")
            raise ValueError("Parameters.txt file was not written correctly")

        # Validate measurement waves before processing
        measurements = {}
        required_measurements = ['AllHeights', 'AllVolumes', 'AllAreas', 'AllAvgHeights']
        
        for measure_name in required_measurements:
            if measure_name not in batch_results:
                print(f"ERROR: {measure_name} missing from batch results")
                raise ValueError(f"Required measurement {measure_name} missing from batch results")
                
            wave = batch_results[measure_name]
            if wave is None:
                print(f"ERROR: {measure_name} wave is None")
                raise ValueError(f"{measure_name} wave is None")
                
            if not hasattr(wave, 'data') or wave.data is None:
                print(f"ERROR: {measure_name} wave has no data attribute or data is None")
                raise ValueError(f"{measure_name} wave has no data")
                
            measurements[measure_name] = wave

        # Add AllCOM only if it exists and has valid data
        if 'AllCOM' in batch_results and batch_results['AllCOM'] is not None:
            com_wave = batch_results['AllCOM']
            if hasattr(com_wave, 'data') and com_wave.data is not None:
                # Ensure COM data has proper 2D shape
                if len(com_wave.data.shape) == 1 and len(com_wave.data) == 0:
                    # Empty COM array - reshape to proper 2D
                    com_wave.data = com_wave.data.reshape(0, 2)
                elif len(com_wave.data.shape) == 2 and com_wave.data.shape[1] == 2:
                    # Valid 2D COM data
                    pass
                else:
                    print(f"WARNING: Invalid COM data shape {com_wave.data.shape}, skipping COM export")
                    com_wave = None
                    
                if com_wave is not None:
                    measurements['AllCOM'] = com_wave
                    print(f"Including AllCOM in export with {len(com_wave.data)} entries")
                else:
                    print("WARNING: AllCOM data invalid, skipping COM export")
            else:
                print("WARNING: AllCOM wave has no data, skipping COM export")
        else:
            print("WARNING: AllCOM data missing from batch results, skipping COM export")

        for wave_name, wave in measurements.items():
            wave_file = series_path / f"{wave_name}.txt"
            print(f"Saving {wave_name} to {wave_file}")
            
            # Additional validation before writing
            if wave is None:
                print(f"ERROR: Wave {wave_name} is None!")
                raise ValueError(f"Wave {wave_name} is None")
            
            if not hasattr(wave, 'data'):
                print(f"ERROR: Wave {wave_name} has no data attribute!")
                raise ValueError(f"Wave {wave_name} has no data attribute")
            
            if wave.data is None:
                print(f"ERROR: Wave {wave_name} data is None!")
                raise ValueError(f"Wave {wave_name} data is None")
            
            try:
                # Verify data arrays are not empty before writing
                if wave_name != 'AllCOM':
                    if len(wave.data) == 0:
                        logger.warning(f"{wave_name} data array is empty")
                    else:
                        # Verify numerical values
                        if not np.all(np.isfinite(wave.data)):
                            logger.warning(f"{wave_name} contains invalid values (inf/nan)")
                else:
                    if wave.data.shape[0] == 0:
                        logger.warning(f"{wave_name} coordinate array is empty")
                    else:
                        if not np.all(np.isfinite(wave.data)):
                            logger.warning(f"{wave_name} contains invalid coordinates (inf/nan)")

                if wave_name == 'AllCOM':
                    # For 2D waves (COM): flatten to 1D for Igor format
                    flat_data = wave.data.flatten()
                    write_wave_file(str(wave_file), flat_data, wave_name)
                else:
                    # For 1D waves (Heights, Areas, Volumes, AvgHeights):
                    write_wave_file(str(wave_file), wave.data, wave_name)
                        
                # Verify file was written properly
                file_size = wave_file.stat().st_size
                logger.info(f"Wrote {wave_name}: {file_size} bytes")
                        
            except Exception as e:
                logger.error(f"Failed to write {wave_name}: {e}")
                raise

        # Create consolidated Info file for ViewParticles compatibility
        info_file = series_path / "Info.txt"
        print(f"Creating consolidated Info file: {info_file}")
        
        try:
            with open(info_file, 'w', encoding='utf-8', newline='\n') as f:
                # Write Igor Pro header comments for consolidated batch info
                f.write("// Igor Pro HessianBlobs Batch Info Wave (Consolidated)\n")
                f.write("// Contains particle information from all analyzed images\n")
                f.write("// Column descriptions:\n")
                f.write("// 0: P_Seed (X coordinate)\n")
                f.write("// 1: Q_Seed (Y coordinate)\n")
                f.write("// 2: numPixels (area in pixels)\n")
                f.write("// 3: maxBlobStrength (detector response)\n")
                f.write("// 4: pStart (X boundary start)\n")
                f.write("// 5: pStop (X boundary end)\n")
                f.write("// 6: qStart (Y boundary start)\n")
                f.write("// 7: qStop (Y boundary end)\n")
                f.write("// 8: scale (characteristic scale)\n")
                f.write("// 9: radius (estimated radius)\n")
                f.write("// 10: inBounds (boundary flag)\n")
                f.write("// 11: height (maximum intensity)\n")
                f.write("// 12: volume (integrated intensity)\n")
                f.write("// 13: area (physical area in nm² if calibrated)\n")
                f.write("// 14: particleNum (particle number)\n")
                
                # Create consolidated info array from all images
                all_info_data = []
                for image_name, results in batch_results['image_results'].items():
                    if 'info' in results and results['info'] is not None:
                        info_data = results['info'].data
                        if info_data.shape[0] > 0:
                            all_info_data.append(info_data)

                if all_info_data:
                    # Concatenate all info data
                    consolidated_info = np.vstack(all_info_data)

                    # Write in Igor Pro wave format for ViewParticles compatibility
                    f.write("Info[0][0]= {")
                    for i, row in enumerate(consolidated_info):
                        if i > 0:
                            f.write(",")
                        # Format each value in the row
                        formatted_values = [format_igor_number(float(val)) for val in row]
                        f.write("{" + ",".join(formatted_values) + "}")
                    f.write("}\n")
                else:
                    # Empty info file
                    f.write("Info[0][0]= {}\n")
                
                # Ensure file is flushed
                f.flush()
                os.fsync(f.fileno())
                
            # Verify consolidated Info file was written
            info_file_size = info_file.stat().st_size
            if info_file_size == 0:
                raise ValueError("Consolidated Info.txt file is empty after write")
                
            # Verify file contains expected Igor Pro format
            with open(info_file, 'r') as verify_f:
                content = verify_f.read()
                if "Info[0][0]=" not in content:
                    raise ValueError("Consolidated Info.txt file does not contain proper Igor Pro format")
                    
            print(f"  Consolidated Info.txt: {info_file_size} bytes written and verified")
                
        except Exception as e:
            print(f"ERROR: Failed to write Info.txt: {e}")
            raise

        # Count files actually saved
        files_saved = 5  # Parameters, AllHeights, AllVolumes, AllAreas, AllAvgHeights
        if 'AllCOM' in measurements:
            files_saved += 1
        files_saved += 1  # Info file

        print(f"Saved {files_saved} files matching Igor Pro BatchHessianBlobs structure:")
        print(f"  - Parameters: Configuration and metadata")
        
        # Safe access to measurement data lengths
        for measure_name in ['AllHeights', 'AllVolumes', 'AllAreas', 'AllAvgHeights']:
            if measure_name in measurements and hasattr(measurements[measure_name], 'data'):
                print(f"  - {measure_name}: {len(measurements[measure_name].data)} values")
            else:
                print(f"  - {measure_name}: ERROR - no data")
                
        if 'AllCOM' in measurements:
            print(f"  - AllCOM: {len(measurements['AllCOM'].data)} coordinate pairs")
        else:
            print(f"  - AllCOM: SKIPPED (no COM data available)")
        print(f"  - Info: Consolidated particle data for ViewParticles")

        # Each image gets its own folder with complete analysis results
        print(f"\nCreating individual image folders matching Igor Pro batch structure...")
        image_counter = 0
        total_particle_counter = 0

        for image_name, results in batch_results['image_results'].items():
            # Clean image name for folder (remove extension, special chars)
            clean_image_name = os.path.splitext(image_name)[0]
            clean_image_name = "".join(c for c in clean_image_name if c.isalnum() or c in ('_', '-'))

            # Create individual image folder within Series_X
            image_folder = series_path / f"{clean_image_name}_Particles"
            image_folder.mkdir(parents=True, exist_ok=True, mode=0o755)
            print(f"Created image folder: {clean_image_name}_Particles")

            # Save complete individual image analysis results (same as SaveSingleImageResults)
            if results:
                # Save all measurement waves for this image
                if 'Heights' in results and results['Heights']:
                    wave_file = Path(image_folder) / "Heights.txt"
                    with open(wave_file, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(f"Heights[0]= {{")
                        for i, value in enumerate(results['Heights'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'Areas' in results and results['Areas']:
                    wave_file = Path(image_folder) / "Areas.txt"
                    with open(wave_file, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(f"Areas[0]= {{")
                        for i, value in enumerate(results['Areas'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'Volumes' in results and results['Volumes']:
                    wave_file = Path(image_folder) / "Volumes.txt"
                    with open(wave_file, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(f"Volumes[0]= {{")
                        for i, value in enumerate(results['Volumes'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'AvgHeights' in results and results['AvgHeights']:
                    wave_file = Path(image_folder) / "AvgHeights.txt"
                    with open(wave_file, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(f"AvgHeights[0]= {{")
                        for i, value in enumerate(results['AvgHeights'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'COM' in results and results['COM']:
                    wave_file = Path(image_folder) / "COM.txt"
                    with open(wave_file, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(f"COM[0][0]= {{")
                        for i, row in enumerate(results['COM'].data):
                            if i > 0:
                                f.write(",")
                            x_formatted = format_igor_number(float(row[0]))
                            y_formatted = format_igor_number(float(row[1]))
                            f.write(f"{{{x_formatted},{y_formatted}}}")
                        f.write("}\n")

                # Save Info.txt for this image
                if 'info' in results and results['info']:
                    info_file = Path(image_folder) / "Info.txt"
                    with open(info_file, 'w', encoding='utf-8', newline='\n') as f:
                        f.write("Info[0][0]= {")
                        for i, row in enumerate(results['info'].data):
                            if i > 0:
                                f.write(",")
                            formatted_values = [format_igor_number(float(val)) for val in row]
                            f.write("{" + ",".join(formatted_values) + "}")
                        f.write("}\n")

                # Create individual particle folders within this image folder
                if 'info' in results and results['info'] is not None:
                    info_data = results['info'].data
                    particle_count_this_image = 0

                    for i in range(info_data.shape[0]):
                        particle_folder = Path(image_folder) / f"Particle_{i}"
                        particle_folder.mkdir(parents=True, exist_ok=True, mode=0o755)

                        # Save complete particle info
                        particle_info_file = Path(particle_folder) / "ParticleInfo.txt"
                        with open(particle_info_file, 'w', encoding='utf-8', newline='\n') as f:
                            f.write(f"Particle {i} from {image_name}\n")
                            f.write("=" * 50 + "\n")

                            # Measurements
                            if 'Heights' in results and i < len(results['Heights'].data):
                                f.write(f"Height: {format_igor_number(results['Heights'].data[i])}\n")
                            if 'Areas' in results and i < len(results['Areas'].data):
                                f.write(f"Area: {format_igor_number(results['Areas'].data[i])}\n")
                            if 'Volumes' in results and i < len(results['Volumes'].data):
                                f.write(f"Volume: {format_igor_number(results['Volumes'].data[i])}\n")
                            if 'AvgHeights' in results and i < len(results['AvgHeights'].data):
                                f.write(f"AvgHeight: {format_igor_number(results['AvgHeights'].data[i])}\n")
                            if 'COM' in results and i < len(results['COM'].data):
                                f.write(f"X_Center: {format_igor_number(results['COM'].data[i][0])}\n")
                                f.write(f"Y_Center: {format_igor_number(results['COM'].data[i][1])}\n")

                            # Info data
                            if i < len(info_data):
                                f.write(f"\nDetection Data:\n")
                                f.write(f"P_Seed: {format_igor_number(info_data[i][0])}\n")
                                f.write(f"Q_Seed: {format_igor_number(info_data[i][1])}\n")
                                f.write(f"Scale: {format_igor_number(info_data[i][2])}\n")
                                f.write(f"Response: {format_igor_number(info_data[i][3])}\n")

                        particle_count_this_image += 1
                        total_particle_counter += 1

                    print(f"  - Created {particle_count_this_image} particle folders in {clean_image_name}_Particles")

            image_counter += 1

        print(f"Created {image_counter} image folders with {total_particle_counter} total particles")
        print(f"\nBatch Analysis Complete - FULL Igor Pro Compatible Structure:")
        print(f"Series folder: {series_folder_name}")

        # Count consolidated waves actually created
        consolidated_waves = 5  # Parameters, AllHeights, AllVolumes, AllAreas, AllAvgHeights
        wave_list = "Parameters, AllHeights, AllVolumes, AllAreas, AllAvgHeights"
        if 'AllCOM' in batch_results and batch_results['AllCOM'] is not None:
            consolidated_waves += 1
            wave_list += ", AllCOM"

        print(f"Consolidated waves: {consolidated_waves} files ({wave_list})")
        print(f"Consolidated Info file: 1 file")
        print(f"Individual image folders: {image_counter} folders")
        print(f"Individual particle folders: {total_particle_counter} folders (organized by image)")
        print(f"Total particles across all images: {batch_results['numParticles']}")

        print(f"\nBatch Structure:")
        print(f"Series_X/")
        print(f"├── Parameters.txt, AllHeights.txt, AllVolumes.txt, AllAreas.txt, AllAvgHeights.txt")
        if 'AllCOM' in batch_results and batch_results['AllCOM'] is not None:
            print(f"├── AllCOM.txt")
        print(f"├── Info.txt (consolidated)")
        for image_name in batch_results['image_results'].keys():
            clean_name = os.path.splitext(image_name)[0]
            clean_name = "".join(c for c in clean_name if c.isalnum() or c in ('_', '-'))
            print(f"├── {clean_name}_Particles/")
            print(f"│   ├── Heights.txt, Areas.txt, Volumes.txt, AvgHeights.txt, COM.txt, Info.txt")
            print(f"│   └── Particle_0/, Particle_1/, ... (individual particles)")

        if total_particle_counter > 0:
            print(f"\nViewParticles compatibility: FULL (individual image and particle folders)")
            print(f"Histogram compatibility: FULL (consolidated and individual data)")
        else:
            print(f"\nViewParticles compatibility: LIMITED (no particles found)")
            print(f"Histogram compatibility: LIMITED (no particles found)")

    else:
        print(f"Unsupported save format: {save_format}. Using Igor format.")

    print(f"=== SAVE BATCH RESULTS COMPLETE ===")
    # Final verification with actual byte counts
    created_files = list(series_path.rglob("*.txt"))
    total_size = sum(f.stat().st_size for f in created_files)
    empty_files = [f for f in created_files if f.stat().st_size == 0]
    
    if empty_files:
        print(f"WARNING: {len(empty_files)} empty files created")
    
    print(f"Batch analysis saved: {len(created_files)} files, {total_size} bytes")
    return str(series_path)


def SaveSingleImageResults(results, image_name, output_path="", save_format="igor"):
    """
    Save single image analysis results

    Parameters:
        results : dict - Results from HessianBlobs for single image
        image_name : str - Name of the analyzed image
        output_path : str - Directory to save files
        save_format : str - Format to save ("igor", "csv", "txt")
    """
    import os
    import datetime
    import numpy as np
    from pathlib import Path

    logger.info("SaveSingleImageResults: Starting single image save operation")
    logger.debug(f"Image: {image_name}, path: {output_path}, format: {save_format}")
    if results:
        logger.debug(f"Results keys: {list(results.keys())}")
    else:
        logger.warning("No results provided")

    # Verify analysis results contain valid data
    if not verify_analysis_results(results):
        print("ERROR: Analysis results validation failed!")
        raise ValueError("Analysis results contain no valid data")

    if not output_path:
        output_path = Path.cwd()
        print(f"Using current directory: {output_path}")
    else:
        output_path = Path(output_path)

    if not output_path.exists():
        print(f"ERROR: Output path does not exist: {output_path}")
        raise ValueError(f"Output path does not exist: {output_path}")
        
    # Check write permissions
    if not os.access(str(output_path), os.W_OK):
        print(f"ERROR: No write permission for output path: {output_path}")
        raise PermissionError(f"No write permission for output path: {output_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create ImageName_Particles folder structure
    clean_image_name = os.path.splitext(image_name)[0]
    clean_image_name = "".join(c for c in clean_image_name if c.isalnum() or c in ('_', '-'))
    folder_name = f"{clean_image_name}_Particles"
    full_path = output_path / folder_name
    print(f"Creating folder: {full_path}")

    try:
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder created successfully: {full_path.exists()}")
    except Exception as e:
        print(f"ERROR: Failed to create folder: {e}")
        raise

    if save_format == "igor" or save_format == "txt":
        print("Saving in Igor Pro format...")

        # Save measurement waves in main folder
        required_keys = ['Heights', 'Areas', 'Volumes', 'AvgHeights', 'COM']
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"ERROR: Missing required keys in results: {missing_keys}")
            raise ValueError(f"Missing required measurement waves: {missing_keys}")

        measurements = {
            'Heights': results['Heights'],
            'Areas': results['Areas'],
            'Volumes': results['Volumes'],
            'AvgHeights': results['AvgHeights'],
            'COM': results['COM']
        }

        print(f"Saving {len(measurements)} measurement waves...")

        for wave_name, wave in measurements.items():
            wave_file = full_path / f"{wave_name}.txt"
            print(f"Saving {wave_name} to {wave_file}")

            # Validate wave data before writing
            if wave is None:
                print(f"ERROR: Wave {wave_name} is None!")
                raise ValueError(f"Wave {wave_name} is None")

            if not hasattr(wave, 'data'):
                print(f"ERROR: Wave {wave_name} has no data attribute!")
                raise ValueError(f"Wave {wave_name} has no data attribute")
                
            if wave.data is None:
                print(f"ERROR: Wave {wave_name} data is None!")
                raise ValueError(f"Wave {wave_name} data is None")
                
            # Check if wave contains actual measurement data
            if wave_name != 'COM' and len(wave.data) == 0:
                print(f"WARNING: Wave {wave_name} contains no data - writing empty wave")
            elif wave_name == 'COM' and (len(wave.data) == 0 or wave.data.shape[0] == 0):
                print(f"WARNING: Wave {wave_name} contains no coordinate data - writing empty wave")

            # Verify data arrays are not empty before writing
            if wave_name != 'COM':
                if len(wave.data) == 0:
                    print(f"WARNING: {wave_name} data array is empty")
                else:
                    # Verify numerical values
                    if not np.all(np.isfinite(wave.data)):
                        print(f"WARNING: {wave_name} contains invalid values (inf/nan)")
            else:
                if wave.data.shape[0] == 0:
                    print(f"WARNING: {wave_name} coordinate array is empty")
                else:
                    if not np.all(np.isfinite(wave.data)):
                        print(f"WARNING: {wave_name} contains invalid coordinates (inf/nan)")

            try:
                if wave_name == 'COM':
                    # For 2D waves (COM): flatten to 1D for Igor format
                    flat_data = wave.data.flatten()
                    write_wave_file(str(wave_file), flat_data, wave_name)
                else:
                    # For 1D waves (Heights, Areas, Volumes, AvgHeights):
                    write_wave_file(str(wave_file), wave.data, wave_name)
                    
                # Verify file was written properly
                file_size = wave_file.stat().st_size
                logger.info(f"Wrote {wave_name}: {file_size} bytes")
                    
            except Exception as e:
                raise ValueError(f"Failed to write {wave_name}: {e}")

        # Save Info.txt file for ViewParticles compatibility
        info_file = full_path / "Info.txt"
        
        # Validate info data exists
        if 'info' not in results or results['info'] is None:
            print("ERROR: Info data missing from results")
            raise ValueError("Info data missing from results")
            
        if not hasattr(results['info'], 'data') or results['info'].data is None:
            print("ERROR: Info wave has no data")
            raise ValueError("Info wave has no data")
            
        info_data = results['info'].data
        
        try:
            with open(info_file, 'w', encoding='utf-8', newline='\n') as f:
                # Write Igor Pro header comments with column descriptions
                f.write("// Igor Pro HessianBlobs Info Wave\n")
                f.write("// Column descriptions:\n")
                f.write("// 0: P_Seed (X coordinate)\n")
                f.write("// 1: Q_Seed (Y coordinate)\n")
                f.write("// 2: numPixels (area in pixels)\n")
                f.write("// 3: maxBlobStrength (detector response)\n")
                f.write("// 4: pStart (X boundary start)\n")
                f.write("// 5: pStop (X boundary end)\n")
                f.write("// 6: qStart (Y boundary start)\n")
                f.write("// 7: qStop (Y boundary end)\n")
                f.write("// 8: scale (characteristic scale)\n")
                f.write("// 9: radius (estimated radius)\n")
                f.write("// 10: inBounds (boundary flag)\n")
                f.write("// 11: height (maximum intensity)\n")
                f.write("// 12: volume (integrated intensity)\n")
                f.write("// 13: area (physical area in nm² if calibrated)\n")
                f.write("// 14: particleNum (particle number)\n")
                
                if len(info_data) == 0 or info_data.shape[0] == 0:
                    f.write("Info[0][0]= {}\n")
                else:
                    f.write("Info[0][0]= {")
                    for i, row in enumerate(info_data):
                        if i > 0:
                            f.write(",")
                        formatted_values = [format_igor_number(float(val)) for val in row]
                        f.write("{" + ",".join(formatted_values) + "}")
                    f.write("}\n")
                    
                # Ensure file is flushed
                f.flush()
                os.fsync(f.fileno())
                
            # Verify Info file was written and contains data
            info_file_size = info_file.stat().st_size
            if info_file_size == 0:
                raise ValueError("Info.txt file is empty after write")
                
            # Verify file contains expected Igor Pro format
            with open(info_file, 'r') as verify_f:
                content = verify_f.read()
                if "Info[0][0]=" not in content:
                    raise ValueError("Info.txt file does not contain proper Igor Pro format")
                    
            print(f"  Info.txt: {info_file_size} bytes written and verified")
                
        except Exception as e:
            raise ValueError(f"Failed to write Info.txt: {e}")

        # Save particle folders with actual data
        info_data = results['info'].data
        heights_data = results['Heights'].data if 'Heights' in results and len(results['Heights'].data) > 0 else []
        areas_data = results['Areas'].data if 'Areas' in results and len(results['Areas'].data) > 0 else []
        volumes_data = results['Volumes'].data if 'Volumes' in results and len(results['Volumes'].data) > 0 else []
        com_data = results['COM'].data if 'COM' in results and results['COM'].data.shape[0] > 0 else []
        
        accepted_particles = results['numParticles']
        particle_folders = results.get('particle_folders', {})
        mapNum_data = results.get('mapNum3D', {}).data if 'mapNum3D' in results else None
        
        for i in range(min(accepted_particles, len(particle_folders))):
            particle_folder = Path(particle_folders[i])
            
            # Write particle image data
            particle_image_file = particle_folder / f"Particle_{i}.npy"
            if mapNum_data is not None:
                particle_data = mapNum_data[mapNum_data == i]
                if particle_data.size > 0:
                    np.save(str(particle_image_file), particle_data)
                    # Verify write
                    if not particle_image_file.exists() or particle_image_file.stat().st_size == 0:
                        raise IOError(f"Failed to write particle {i} data")
            
            # Write particle info with explicit formatting
            particle_info_file = particle_folder / "ParticleInfo.txt"
            with open(particle_info_file, 'w', encoding='utf-8') as f:
                f.write(f"Particle {i} Information\n")
                f.write(f"Height: {heights_data[i] if i < len(heights_data) else 0.0:.6e}\n")
                f.write(f"Area: {areas_data[i] if i < len(areas_data) else 0.0:.6e}\n")
                f.write(f"Volume: {volumes_data[i] if i < len(volumes_data) else 0.0:.6e}\n")
                if i < len(com_data) and len(com_data[i]) >= 2:
                    f.write(f"Center: ({com_data[i][0]:.6f}, {com_data[i][1]:.6f})\n")
                f.flush()
                os.fsync(f.fileno())

        logger.info("HessianBlobs: Single image analysis exported")

    elif save_format == "csv":
        # CSV format for Excel
        import csv
        csv_file = full_path / f"{image_name}_particles.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['# Hessian Blob Analysis Results'])
            writer.writerow(['# Image:', image_name])
            writer.writerow(['# Particles:', results['numParticles']])
            writer.writerow([])
            writer.writerow(['Particle_ID', 'Height', 'Area', 'Volume', 'AvgHeight', 'X_Center', 'Y_Center'])

            # Safe access to measurement waves with bounds checking
            heights = results['Heights'].data if 'Heights' in results and len(results['Heights'].data) > 0 else []
            areas = results['Areas'].data if 'Areas' in results and len(results['Areas'].data) > 0 else []
            volumes = results['Volumes'].data if 'Volumes' in results and len(results['Volumes'].data) > 0 else []
            avg_heights = results['AvgHeights'].data if 'AvgHeights' in results and len(
                results['AvgHeights'].data) > 0 else []
            com = results['COM'].data if 'COM' in results and results['COM'].data.shape[0] > 0 else []

            num_particles = max(len(heights), len(areas), len(volumes), len(avg_heights), len(com))
            for i in range(num_particles):
                height_val = heights[i] if i < len(heights) else 'N/A'
                area_val = areas[i] if i < len(areas) else 'N/A'
                volume_val = volumes[i] if i < len(volumes) else 'N/A'
                avg_height_val = avg_heights[i] if i < len(avg_heights) else 'N/A'
                x_center = com[i, 0] if i < len(com) and com.shape[1] >= 2 else 'N/A'
                y_center = com[i, 1] if i < len(com) and com.shape[1] >= 2 else 'N/A'
                writer.writerow([i + 1, height_val, area_val, volume_val, avg_height_val, x_center, y_center])

    print(f"=== SAVE SINGLE IMAGE COMPLETE ===")
    print(f"Single image results saved to: {full_path}")
    print(f"Folder exists: {os.path.exists(full_path)}")
    # Final verification with actual byte counts
    created_files = list(full_path.rglob("*.txt"))
    total_size = sum(f.stat().st_size for f in created_files)
    empty_files = [f for f in created_files if f.stat().st_size == 0]
    
    if empty_files:
        print(f"WARNING: {len(empty_files)} empty files created")
    
    print(f"Single image analysis saved: {len(created_files)} files, {total_size} bytes")
    return full_path


def ViewParticleData(info_wave, image_name, original_image=None):
    """ViewParticles implementation - scroll through individual blobs"""
    try:
        # Igor Pro: Check if particles exist
        if info_wave is None or info_wave.data.shape[0] == 0:
            messagebox.showwarning("No Particles", "No particles to view.")
            return

        logger.debug(f"ViewParticleData: Creating viewer with {info_wave.data.shape[0]} particles")

        # Validate original image data
        if original_image is None:
            messagebox.showwarning("No Image Data", "Original image data is required for particle viewing.")
            return

        # Use the working ViewParticles function from particle_measurements.py
        from particle_measurements import ViewParticles
        ViewParticles(original_image, info_wave)

    except Exception as e:
        logger.error(f"ViewParticleData error: {str(e)}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("ViewParticles Error", f"Error creating particle viewer:\n{str(e)}")


class ParticleViewer:
    """Interactive particle viewer with measurement display"""

    def __init__(self, info_wave, image_name, original_image=None):
        try:
            logger.debug("ParticleViewer: Starting initialization")
            self.info_wave = info_wave
            self.image_name = image_name
            self.original_image = original_image
            self.num_particles = info_wave.data.shape[0]
            self.current_particle = 0

            # Igor Pro ViewParticles settings
            self.color_table = "gray"
            self.color_range = -1  # -1 = autoscale
            self.interpolate = False
            self.show_perimeter = True
            self.x_range = -1  # -1 = autoscale
            self.y_range = -1  # -1 = autoscale

            logger.debug(f"ParticleViewer: Creating window for {self.num_particles} particles")

            self.root = tk.Toplevel()
            self.root.title("Particle Viewer")
            self.root.geometry("900x600")
            self.root.transient()
            self.root.focus_set()

            self.setup_viewer()
            logger.debug("ParticleViewer: Initialization complete")

        except Exception as e:
            logger.error(f"ParticleViewer init error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def setup_viewer(self):
        """Setup the particle viewer interface"""
        try:
            logger.debug("Setting up Igor Pro style layout")

            main_container = ttk.Frame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Image display area (main particle view)
            image_frame = ttk.Frame(main_container)
            image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

            # Create matplotlib figure for particle display
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Controls panel
            controls_container = ttk.Frame(main_container, width=150)
            controls_container.pack(side=tk.RIGHT, fill=tk.Y)
            controls_container.pack_propagate(False)

            self.particle_title = ttk.Label(controls_container,
                                            text=f"Particle {self.current_particle + 1}",
                                            font=('TkDefaultFont', 15, 'bold'))
            self.particle_title.pack(pady=(10, 5))

            nav_frame = ttk.Frame(controls_container)
            nav_frame.pack(fill=tk.X, pady=5)

            prev_btn = ttk.Button(nav_frame, text="Prev",
                                  command=self.prev_particle, width=8)
            prev_btn.pack(side=tk.LEFT, padx=(5, 2))

            next_btn = ttk.Button(nav_frame, text="Next",
                                  command=self.next_particle, width=8)
            next_btn.pack(side=tk.LEFT, padx=(2, 5))

            goto_frame = ttk.Frame(controls_container)
            goto_frame.pack(fill=tk.X, pady=5)

            ttk.Label(goto_frame, text="Go To:", font=('TkDefaultFont', 13)).pack(anchor=tk.W, padx=5)
            self.goto_var = tk.IntVar(value=self.current_particle + 1)
            self.goto_entry = ttk.Entry(goto_frame, textvariable=self.goto_var, width=12)
            self.goto_entry.pack(anchor=tk.W, padx=5, pady=2)
            self.goto_entry.bind('<Return>', self.goto_particle)

            color_frame = ttk.Frame(controls_container)
            color_frame.pack(fill=tk.X, pady=5)

            ttk.Label(color_frame, text="Color Table:", font=('TkDefaultFont', 11)).pack(anchor=tk.W, padx=5)
            self.color_var = tk.StringVar(value=self.color_table)
            color_combo = ttk.Combobox(color_frame, textvariable=self.color_var,
                                       values=["gray", "hot", "cool", "rainbow", "viridis", "plasma"],
                                       width=15, state="readonly")
            color_combo.pack(anchor=tk.W, padx=5, pady=2)
            color_combo.bind('<<ComboboxSelected>>', self.on_color_change)

            range_frame = ttk.Frame(controls_container)
            range_frame.pack(fill=tk.X, pady=5)

            ttk.Label(range_frame, text="Color Range:", font=('TkDefaultFont', 11)).pack(anchor=tk.W, padx=5)
            self.range_var = tk.DoubleVar(value=self.color_range)
            range_entry = ttk.Entry(range_frame, textvariable=self.range_var, width=12)
            range_entry.pack(anchor=tk.W, padx=5, pady=2)
            range_entry.bind('<Return>', self.on_range_change)

            self.interp_var = tk.BooleanVar(value=self.interpolate)
            interp_check = ttk.Checkbutton(controls_container, text="Interpolate:",
                                           variable=self.interp_var,
                                           command=self.on_interp_change)
            interp_check.pack(anchor=tk.W, padx=5, pady=2)

            self.perim_var = tk.BooleanVar(value=self.show_perimeter)
            perim_check = ttk.Checkbutton(controls_container, text="Perimeter:",
                                          variable=self.perim_var,
                                          command=self.on_perim_change)
            perim_check.pack(anchor=tk.W, padx=5, pady=2)

            xy_frame = ttk.Frame(controls_container)
            xy_frame.pack(fill=tk.X, pady=5)

            ttk.Label(xy_frame, text="X-Range:", font=('TkDefaultFont', 10)).pack(anchor=tk.W, padx=5)
            self.x_range_var = tk.DoubleVar(value=self.x_range)
            x_entry = ttk.Entry(xy_frame, textvariable=self.x_range_var, width=12)
            x_entry.pack(anchor=tk.W, padx=5, pady=1)
            x_entry.bind('<Return>', self.on_range_change)

            ttk.Label(xy_frame, text="Y-Range:", font=('TkDefaultFont', 10)).pack(anchor=tk.W, padx=5, pady=(5, 0))
            self.y_range_var = tk.DoubleVar(value=self.y_range)
            y_entry = ttk.Entry(xy_frame, textvariable=self.y_range_var, width=12)
            y_entry.pack(anchor=tk.W, padx=5, pady=1)
            y_entry.bind('<Return>', self.on_range_change)

            measurements_frame = ttk.LabelFrame(controls_container, text="Measurements", padding="5")
            measurements_frame.pack(fill=tk.X, pady=10)

            # Height
            ttk.Label(measurements_frame, text="Height", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
            self.height_label = ttk.Label(measurements_frame, text="0.0",
                                          font=('TkDefaultFont', 11), relief="sunken", width=15)
            self.height_label.pack(anchor=tk.W, pady=2)

            # Volume
            ttk.Label(measurements_frame, text="Volume", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W,
                                                                                                  pady=(10, 0))
            self.volume_label = ttk.Label(measurements_frame, text="0.0",
                                          font=('TkDefaultFont', 11), relief="sunken", width=15)
            self.volume_label.pack(anchor=tk.W, pady=2)

            delete_frame = ttk.Frame(controls_container)
            delete_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

            delete_btn = ttk.Button(delete_frame, text="DELETE",
                                    command=self.delete_particle)
            delete_btn.pack(fill=tk.X, padx=5)

            # Close button
            close_btn = ttk.Button(delete_frame, text="Close",
                                   command=self.close_viewer)
            close_btn.pack(fill=tk.X, padx=5, pady=(5, 0))

            # Igor Pro: Set up keyboard shortcuts
            self.root.bind('<Key>', self.on_key_press)
            self.root.focus_set()

            # Display first particle
            self.display_current_particle()

        except Exception as e:
            logger.error(f"setup_viewer error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def display_current_particle(self):
        """Display the current particle with measurements"""
        try:
            # Bounds checking
            if self.current_particle >= self.num_particles:
                self.current_particle = self.num_particles - 1
            elif self.current_particle < 0:
                self.current_particle = 0

            # Get particle data
            particle_data = self.info_wave.data[self.current_particle]
            x_pos = particle_data[0]
            y_pos = particle_data[1]
            radius = particle_data[2]
            response = particle_data[3]
            scale = particle_data[4] if len(particle_data) > 4 else 0.0

            # Clear and setup the plot
            self.ax.clear()

            if self.original_image is not None and hasattr(self.original_image, 'data'):
                # Cropping - show region around particle

                # Calculate crop bounds (larger region than just the particle)
                crop_size = max(int(radius * 4), 50)

                x_min = max(0, int(x_pos - crop_size))
                x_max = min(self.original_image.data.shape[1], int(x_pos + crop_size))
                y_min = max(0, int(y_pos - crop_size))
                y_max = min(self.original_image.data.shape[0], int(y_pos + crop_size))

                # Crop the actual image region
                cropped_image = self.original_image.data[y_min:y_max, x_min:x_max]

                # Display with proper extent and user settings
                interpolation = 'bilinear' if self.interpolate else 'nearest'

                # Apply color range if specified (-1 = autoscale)
                if self.color_range == -1:
                    vmin, vmax = None, None  # Autoscale
                else:
                    vmin, vmax = 0, self.color_range

                self.ax.imshow(cropped_image, cmap=self.color_table,
                               extent=[x_min, x_max, y_max, y_min],
                               interpolation=interpolation,
                               vmin=vmin, vmax=vmax)

                # Set the view limits
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_max, y_min)

            else:
                # If no original image, show error
                messagebox.showwarning("No Image Data", "Original image not available for viewing.")
                return

            # Draw the particle perimeter if enabled
            if self.show_perimeter:
                circle = Circle((x_pos, y_pos), radius,
                                fill=False, edgecolor='lime', linewidth=2, alpha=0.9)
                self.ax.add_patch(circle)

            # Mark the center (red crosshair like Igor Pro)
            self.ax.plot(x_pos, y_pos, 'r+', markersize=12, markeredgewidth=3)

            # Style title and axis
            self.ax.set_title(f"Particle {self.current_particle + 1} of {self.num_particles}",
                              fontsize=12, fontweight='bold')
            self.ax.set_aspect('equal')

            # Add scale bar if radius is reasonable size
            if radius > 5:
                scale_length = radius
                scale_x = x_min + (x_max - x_min) * 0.1
                scale_y = y_max - (y_max - y_min) * 0.1
                self.ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y],
                             'yellow', linewidth=3)
                self.ax.text(scale_x + scale_length / 2, scale_y - (y_max - y_min) * 0.05,
                             f'{scale_length:.1f} px', ha='center', color='yellow',
                             fontsize=10, fontweight='bold')

            # Update measurement labels
            self.particle_title.config(text=f"Particle {self.current_particle + 1}")

            # Calculate height and volume (simplified for now)
            height = response * 1000  # Simplified height calculation
            volume = (4 / 3) * np.pi * (radius ** 3)  # Sphere volume approximation

            self.height_label.config(text=f"{height:.2f}")
            self.volume_label.config(text=f"{volume:.2f}")
            self.goto_var.set(self.current_particle + 1)

            self.canvas.draw()

        except Exception as e:
            print(f"Error displaying particle {self.current_particle + 1}: {str(e)}")
            messagebox.showerror("Display Error", f"Error displaying particle:\n{str(e)}")

    # Igor Pro ViewParticles callback methods

    def on_color_change(self, event=None):
        """Handle color table change"""
        self.color_table = self.color_var.get()
        self.display_current_particle()

    def on_range_change(self, event=None):
        """Handle display range change"""
        try:
            self.color_range = self.range_var.get()
            self.x_range = self.x_range_var.get()
            self.y_range = self.y_range_var.get()
            self.display_current_particle()
        except tk.TclError:
            pass  # Invalid input, ignore

    def on_interp_change(self):
        """Handle interpolation toggle"""
        self.interpolate = self.interp_var.get()
        self.display_current_particle()

    def on_perim_change(self):
        """Handle perimeter display toggle"""
        self.show_perimeter = self.perim_var.get()
        self.display_current_particle()

    def delete_particle(self):
        """Delete particle with confirmation dialog"""
        particle_num = self.current_particle + 1
        result = messagebox.askyesno(
            f"Deleting Particle {particle_num}...",
            f"Are you sure you want to delete Particle {particle_num}?",
            icon='warning'
        )
        if result:
            messagebox.showinfo("Delete",
                                f"Particle {particle_num} marked for deletion.\n(Not implemented in Python port)")

    def on_key_press(self, event):
        """Handle keyboard shortcuts for navigation"""
        # Keyboard navigation implementation
        if event.keysym == 'Right':  # Arrow Right - Next
            self.next_particle()
        elif event.keysym == 'Left':  # Arrow Left - Prev
            self.prev_particle()
        elif event.keysym in ['Down', 'space']:  # Down Arrow or Space - Delete
            self.delete_particle()

    def next_particle(self):
        """Navigate to next particle"""
        if self.current_particle < self.num_particles - 1:
            self.current_particle += 1
            self.display_current_particle()

    def prev_particle(self):
        """Navigate to previous particle"""
        if self.current_particle > 0:
            self.current_particle -= 1
            self.display_current_particle()

    def goto_particle(self, event=None):
        """Go to specific particle number"""
        try:
            particle_num = self.goto_var.get() - 1  # Convert to 0-based index
            if 0 <= particle_num < self.num_particles:
                self.current_particle = particle_num
                self.display_current_particle()
        except (ValueError, tk.TclError):
            pass  # Invalid input, ignore

    def close_viewer(self):
        """Close the particle viewer"""
        self.root.destroy()

    def run(self):
        """Run the particle viewer"""
        self.root.mainloop()

