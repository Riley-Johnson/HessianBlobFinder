"""
Utilities Module
Contains various utility functions used throughout the blob detection algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import messagebox
from scipy import ndimage

from igor_compatibility import *
from file_io import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def Maxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """
    Find local maxima in the detector response using scale-space blob detection.
    
    This function implements a two-phase local maxima detection algorithm that identifies
    blob-like structures in scale-space. The algorithm uses asymmetric neighbor comparison
    to ensure robust maximum detection while avoiding tie-breaking issues.

    Parameters:
    detH : Wave - The determinant of Hessian blob detector response (3D scale-space)
    LG : Wave - The Laplacian of Gaussian blob detector response (3D scale-space) 
    particleType : int - Type of particles to detect (-1 for dark blobs, 1 for bright blobs, 0 for both)
    maxCurvatureRatio : float - Maximum ratio of principal curvatures for edge suppression
    map_wave : Wave - Output map for particle identification (optional)
    scaleMap : Wave - Output map for optimal scale information (optional)

    Returns:
    Wave - 2D wave containing maximum detector responses at each spatial position
    """
    print(f"Computing local maxima for particle type: {particleType}")
    print("  particleType = 1: positive blobs (bright spots)")
    print("  particleType = -1: negative blobs (dark spots)")
    print("  particleType = 0: both types")

    # Initialize output wave with same spatial dimensions as input
    maxes_data = np.zeros(detH.data.shape[:2])

    if map_wave is not None:
        map_wave.data = np.full(detH.data.shape[:2], -1.0)
    if scaleMap is not None:
        scaleMap.data = np.zeros(detH.data.shape[:2])

    # Get dimensions
    limI = DimSize(detH, 0)  # Height
    limJ = DimSize(detH, 1)  # Width
    limK = DimSize(detH, 2)  # Scale layers

    print(f"Image dimensions: {limI} x {limJ} x {limK}")

    blob_count = 0

    # Iterate through all spatial positions
    for i in range(1, limI - 1):  # Skip boundary pixels
        for j in range(1, limJ - 1):

            # Find scale-space maxima at this spatial location
            for k in range(0, limK):  # Include all scales including boundaries

                # Handle boundary scale layers with reduced neighbor sets
                # Scale boundaries require special neighbor configuration due to limited 3D context
                
                detH_val = detH.data[i, j, k]
                LG_val = LG.data[i, j, k]

                # Skip if detector response is too weak
                if detH_val <= 0:
                    continue

                # Apply particle type filtering based on Laplacian of Gaussian sign
                if particleType == 1 and LG_val > 0:
                    continue  # Skip positive responses when looking for negative blobs
                if particleType == -1 and LG_val < 0:
                    continue  # Skip negative responses when looking for positive blobs
                # particleType == 0: both types allowed, no filtering

                # Phase 1: Strict greater-than test against asymmetrically chosen neighbors
                # This phase ensures the center value is strictly greater than a subset of neighbors
                strictlyGreater = True
                
                # Define asymmetric neighbor subset for Phase 1 comparison
                if k == 0:
                    # Boundary case: first scale layer (6 asymmetric neighbors)
                    phase1_neighbors = [
                        (i-1, j-1, k), (i-1, j, k), (i, j-1, k),
                        (i, j, k+1), (i, j-1, k+1), (i-1, j, k+1)
                    ]
                elif k == limK - 1:
                    # Boundary case: last scale layer (6 asymmetric neighbors)  
                    phase1_neighbors = [
                        (i-1, j-1, k-1), (i-1, j, k-1), (i, j-1, k-1),
                        (i, j, k-1), (i, j-1, k), (i-1, j, k)
                    ]
                else:
                    # Interior scales: full asymmetric neighbor set (7 neighbors)
                    phase1_neighbors = [
                        (i-1, j-1, k-1), (i-1, j-1, k), (i-1, j, k-1),
                        (i, j-1, k-1), (i, j, k-1), (i, j-1, k), (i-1, j, k)
                    ]
                
                # Apply strict greater-than test to Phase 1 neighbors
                for ni, nj, nk in phase1_neighbors:
                    if (ni >= 0 and ni < limI and nj >= 0 and nj < limJ and nk >= 0 and nk < limK):
                        if detH_val <= detH.data[ni, nj, nk]:
                            strictlyGreater = False
                            break
                
                # Phase 2: Greater-than-or-equal test against remaining neighbors  
                # This phase ensures the center value is greater than or equal to all other neighbors
                greaterOrEqual = True
                
                if strictlyGreater:
                    # Define complementary neighbor set for Phase 2 (remaining 26-neighborhood)
                    if k == 0:
                        # Boundary case: first scale layer complementary neighbors (20 neighbors)
                        phase2_neighbors = [
                            # Layer k+1 remaining neighbors (2 neighbors)
                            (i-1, j-1, k+1), (i+1, j, k+1),
                            # Layer k remaining neighbors (5 neighbors)
                            (i, j+1, k), (i+1, j-1, k), (i+1, j, k), (i+1, j+1, k), (i-1, j+1, k),
                            # Layer k+2 if exists (all 9 neighbors)
                            (i-1, j-1, k+2), (i-1, j, k+2), (i-1, j+1, k+2),
                            (i, j-1, k+2), (i, j, k+2), (i, j+1, k+2),
                            (i+1, j-1, k+2), (i+1, j, k+2), (i+1, j+1, k+2)
                        ]
                    elif k == limK - 1:
                        # Boundary case: last scale layer complementary neighbors (20 neighbors)
                        phase2_neighbors = [
                            # Layer k-1 remaining neighbors (3 neighbors)
                            (i-1, j+1, k-1), (i, j+1, k-1), (i+1, j-1, k-1), (i+1, j, k-1), (i+1, j+1, k-1),
                            # Layer k remaining neighbors (7 neighbors)
                            (i, j+1, k), (i+1, j-1, k), (i+1, j, k), (i+1, j+1, k), (i-1, j+1, k),
                            # Layer k-2 if exists (all 9 neighbors)
                            (i-1, j-1, k-2), (i-1, j, k-2), (i-1, j+1, k-2),
                            (i, j-1, k-2), (i, j, k-2), (i, j+1, k-2),
                            (i+1, j-1, k-2), (i+1, j, k-2), (i+1, j+1, k-2)
                        ]
                    else:
                        # Interior scales: complementary neighbor set (19 neighbors)
                        phase2_neighbors = [
                            # Layer k-1 remaining neighbors (5 neighbors)
                            (i-1, j+1, k-1), (i, j+1, k-1), (i+1, j-1, k-1), (i+1, j, k-1), (i+1, j+1, k-1),
                            # Layer k remaining neighbors (5 neighbors)
                            (i, j+1, k), (i+1, j-1, k), (i+1, j, k), (i+1, j+1, k), (i-1, j+1, k),
                            # Layer k+1 all neighbors (9 neighbors)
                            (i-1, j-1, k+1), (i-1, j, k+1), (i-1, j+1, k+1),
                            (i, j-1, k+1), (i, j, k+1), (i, j+1, k+1),
                            (i+1, j-1, k+1), (i+1, j, k+1), (i+1, j+1, k+1)
                        ]
                    
                    # Apply greater-than-or-equal test to Phase 2 neighbors
                    for ni, nj, nk in phase2_neighbors:
                        if (ni >= 0 and ni < limI and nj >= 0 and nj < limJ and nk >= 0 and nk < limK):
                            if detH_val < detH.data[ni, nj, nk]:
                                greaterOrEqual = False
                                break

                # Accept point as local maximum only if both phase conditions are satisfied
                if not (strictlyGreater and greaterOrEqual):
                    continue

                # Apply edge suppression using curvature ratio test
                if maxCurvatureRatio > 0:
                    curvature_ratio = (LG_val * LG_val) / detH_val
                    threshold_ratio = ((maxCurvatureRatio + 1) ** 2) / maxCurvatureRatio
                    
                    if curvature_ratio >= threshold_ratio:
                        continue  # Reject edge-like responses

                # Record valid local maximum
                blob_count += 1

                # Update maximum response at this spatial location
                maxes_data[i, j] = max(maxes_data[i, j], detH_val)

                # Update optional output maps
                if map_wave is not None:
                    map_wave.data[i, j] = max(map_wave.data[i, j], detH_val)

                if scaleMap is not None:
                    # Calculate scale value using dimension parameters
                    scale_value = DimOffset(detH, 2) * (DimDelta(detH, 2) ** k)
                    if detH_val > maxes_data[i, j] or scaleMap.data[i, j] == 0:
                        scaleMap.data[i, j] = scale_value

    print(f"Found {blob_count} local maxima")

    # Create output wave
    maxes_wave = Wave(maxes_data, "maxes")

    # Copy scaling from input
    maxes_wave.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0))
    maxes_wave.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1))

    return maxes_wave


def LocalMaxima3D(data, i, j, k):
    """
    Check if point (i,j,k) is a local maximum in 3D data
    Helper function for Maxes
    """
    if i == 0 or i >= data.shape[0] - 1:
        return False
    if j == 0 or j >= data.shape[1] - 1:
        return False
    if k == 0 or k >= data.shape[2] - 1:
        return False

    center_val = data[i, j, k]

    # Check all 26 neighbors in 3D
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue

                neighbor_val = data[i + di, j + dj, k + dk]

                # Use asymmetric comparison for tie-breaking
                if di < 0 or (di == 0 and dj < 0) or (di == 0 and dj == 0 and dk < 0):
                    if center_val <= neighbor_val:
                        return False
                else:
                    if center_val < neighbor_val:
                        return False

    return True


def ApplyCurvatureTest(detH, i, j, k, maxCurvatureRatio):
    """
    Apply curvature ratio test at location (i,j,k)
    Simplified version - full implementation would compute Hessian eigenvalues
    """
    if maxCurvatureRatio <= 0:
        return True  # Skip test

    # Simplified test - in practice would compute full Hessian matrix
    # and check ratio of eigenvalues
    return True


def SubPixelRefinement(detH, i, j, k):
    """
    Perform sub-pixel refinement of blob location
    Uses quadratic interpolation around the maximum
    """
    # Get neighborhood values
    if (i <= 0 or i >= detH.data.shape[0] - 1 or
            j <= 0 or j >= detH.data.shape[1] - 1 or
            k <= 0 or k >= detH.data.shape[2] - 1):
        return i, j, k  # Can't refine at boundaries

    # Simple quadratic interpolation in each dimension
    # X direction
    f_left = detH.data[i, j - 1, k]
    f_center = detH.data[i, j, k]
    f_right = detH.data[i, j + 1, k]

    if f_left + f_right - 2 * f_center != 0:
        dx = 0.5 * (f_left - f_right) / (f_left + f_right - 2 * f_center)
    else:
        dx = 0

    # Y direction
    f_up = detH.data[i - 1, j, k]
    f_down = detH.data[i + 1, j, k]

    if f_up + f_down - 2 * f_center != 0:
        dy = 0.5 * (f_up - f_down) / (f_up + f_down - 2 * f_center)
    else:
        dy = 0

    # Z direction (scale)
    f_prev = detH.data[i, j, k - 1]
    f_next = detH.data[i, j, k + 1]

    if f_prev + f_next - 2 * f_center != 0:
        dz = 0.5 * (f_prev - f_next) / (f_prev + f_next - 2 * f_center)
    else:
        dz = 0

    # Clamp refinements to reasonable range
    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)
    dz = np.clip(dz, -0.5, 0.5)

    return i + dy, j + dx, k + dz


def FilterBySize(blob_info, min_size, max_size):
    """
    Filter blobs by size constraints
    """
    if blob_info.shape[0] == 0:
        return blob_info

    # Column 2 contains radius information
    radii = blob_info[:, 2]

    # Create mask for blobs within size range
    size_mask = (radii >= min_size) & (radii <= max_size)

    return blob_info[size_mask]


def FilterByResponse(blob_info, min_response):
    """
    Filter blobs by minimum response strength
    """
    if blob_info.shape[0] == 0:
        return blob_info

    # Column 3 contains response strength
    responses = blob_info[:, 3]

    # Create mask for blobs above threshold
    response_mask = responses >= min_response

    return blob_info[response_mask]


def ScanFill(image, startI, startJ, layer, val, dest, destLayer, fill):
    """
    Flood fill algorithm for connected component analysis
    Direct port from Igor Pro ScanFill function
    
    Parameters:
    image : Wave - Source image to scan
    startI, startJ : int - Starting coordinates 
    layer : int - Layer in source image
    val : float - Value to match for filling
    dest : Wave - Destination image for filling
    destLayer : int - Layer in destination image  
    fill : float - Fill value to use
    
    Returns:
    complex - Count of filled pixels + j*isBP (boundary pixel flag)
    """
    if image is None or dest is None:
        return complex(0, 0)
    
    # Get dimensions
    limI, limJ = image.data.shape[:2]
    
    # Boundary check
    if startI < 0 or startI >= limI or startJ < 0 or startJ >= limJ:
        return complex(0, 1)  # isBP=1 for boundary pixel
    
    # Check if start pixel matches target value
    if len(image.data.shape) == 3:
        start_val = image.data[startI, startJ, layer] if layer < image.data.shape[2] else 0
    else:
        start_val = image.data[startI, startJ]
    
    if abs(start_val - val) > 1e-10:  # Not matching target value
        return complex(0, 0)
    
    # Initialize flood fill using stack-based approach
    stack = [(startI, startJ)]
    visited = set()
    count = 0
    isBP = 0
    
    while stack:
        i, j = stack.pop()
        
        # Skip if already visited or out of bounds
        if (i, j) in visited or i < 0 or i >= limI or j < 0 or j >= limJ:
            continue
            
        # Check if current pixel matches target value
        if len(image.data.shape) == 3:
            current_val = image.data[i, j, layer] if layer < image.data.shape[2] else 0
        else:
            current_val = image.data[i, j]
            
        if abs(current_val - val) > 1e-10:
            continue
            
        # Mark as visited and fill
        visited.add((i, j))
        count += 1
        
        # Set boundary pixel flag if on image edge
        if i == 0 or i == limI-1 or j == 0 or j == limJ-1:
            isBP = 1
        
        # Fill destination pixel
        if len(dest.data.shape) == 3:
            if destLayer < dest.data.shape[2]:
                dest.data[i, j, destLayer] = fill
        else:
            dest.data[i, j] = fill
        
        # Add 4-connected neighbors to stack (Igor Pro uses 4-connectivity)
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        for ni, nj in neighbors:
            if (ni, nj) not in visited:
                stack.append((ni, nj))
    
    return complex(count, isBP)


def ImageHistogram(wave, num_bins=256):
    """
    Compute histogram of image values
    """
    data = wave.data.flatten()
    hist, bin_edges = np.histogram(data, bins=num_bins)
    return hist, bin_edges


def ThresholdOtsu(wave):
    """
    Compute Otsu threshold for a wave
    """
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(wave.data)
    except ImportError:
        # Fallback implementation
        data = wave.data.flatten()
        return np.mean(data) + np.std(data)


def ConnectedComponents(binary_image):
    """
    Find connected components in binary image
    """
    labeled_array, num_features = ndimage.label(binary_image)
    return labeled_array, num_features


def ImageMoments(wave, order=2):
    """
    Compute image moments up to specified order
    """
    data = wave.data
    y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    moments = {}

    for p in range(order + 1):
        for q in range(order + 1 - p):
            if p + q <= order:
                moment = np.sum((x_coords ** p) * (y_coords ** q) * data)
                moments[f'm{p}{q}'] = moment

    return moments


def CenterOfMass(wave):
    """
    Compute center of mass of image
    """
    data = wave.data
    total_mass = np.sum(data)

    if total_mass == 0:
        return 0, 0

    y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    cm_x = np.sum(x_coords * data) / total_mass
    cm_y = np.sum(y_coords * data) / total_mass

    return cm_x, cm_y


def RadialProfile(wave, center_x, center_y, max_radius=None):
    """
    Compute radial profile around a center point
    """
    data = wave.data
    height, width = data.shape

    if max_radius is None:
        max_radius = min(height, width) // 2

    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Calculate distances from center
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Create radial bins
    radial_bins = np.arange(0, max_radius + 1)
    profile = []

    for r in radial_bins:
        mask = (distances >= r - 0.5) & (distances < r + 0.5)
        if np.any(mask):
            profile.append(np.mean(data[mask]))
        else:
            profile.append(0)

    return np.array(profile), radial_bins


def FitGaussianBlob(data, x0, y0, radius):
    """
    Fit a 2D Gaussian to a blob region
    Simplified implementation
    """
    # Extract region around blob
    height, width = data.shape

    x_min = max(0, int(x0 - radius * 2))
    x_max = min(width, int(x0 + radius * 2))
    y_min = max(0, int(y0 - radius * 2))
    y_max = min(height, int(y0 + radius * 2))

    if x_max <= x_min or y_max <= y_min:
        return None

    region = data[y_min:y_max, x_min:x_max]

    # Simple moment-based fitting
    total = np.sum(region)
    if total == 0:
        return None

    y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]

    # Center of mass
    cm_x = np.sum(x_coords * region) / total
    cm_y = np.sum(y_coords * region) / total

    # Convert back to full image coordinates
    cm_x += x_min
    cm_y += y_min

    # Estimate amplitude
    amplitude = np.max(region)

    # Estimate sigma from radius
    sigma = radius / 2.0

    return {
        'amplitude': amplitude,
        'x0': cm_x,
        'y0': cm_y,
        'sigma_x': sigma,
        'sigma_y': sigma,
        'background': np.min(region)
    }


def ExportResultsCSV(results, filename):
    """
    Export blob detection results to CSV format
    """
    import csv

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['X', 'Y', 'Radius', 'Response', 'Scale', 'Boundary',
                  'Reserved1', 'Reserved2', 'Area', 'Volume', 'Height']
        writer.writerow(header)

        # Write data
        for row in results:
            writer.writerow(row)


def ValidateParameters(params):
    """
    Validate blob detection parameters
    """
    required_keys = ['scaleStart', 'layers', 'scaleFactor', 'detHResponseThresh',
                     'particleType', 'maxCurvatureRatio', 'subPixelMult', 'allowOverlap']

    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    # Validate ranges
    if params['scaleStart'] <= 0:
        raise ValueError("scaleStart must be positive")

    if params['layers'] <= 0:
        raise ValueError("layers must be positive")

    if params['scaleFactor'] <= 1:
        raise ValueError("scaleFactor must be > 1")

    if params['particleType'] not in [-1, 0, 1]:
        raise ValueError("particleType must be -1, 0, or 1")

    return True


# Additional helper functions

def NonMaximumSuppression(response_map, radius_map, threshold):
    """
    Apply non-maximum suppression to remove overlapping detections
    """
    # Find all pixels above threshold
    candidates = np.where(response_map >= threshold)

    if len(candidates[0]) == 0:
        return np.array([]), np.array([]), np.array([])

    # Create list of detections with (response, y, x, radius)
    detections = []
    for i in range(len(candidates[0])):
        y, x = candidates[0][i], candidates[1][i]
        response = response_map[y, x]
        radius = radius_map[y, x] if radius_map is not None else 1.0
        detections.append((response, y, x, radius))

    # Sort by response (descending)
    detections.sort(reverse=True, key=lambda x: x[0])

    # Apply non-maximum suppression
    keep = []
    for i, detection in enumerate(detections):
        response, y, x, radius = detection

        # Check against all previously kept detections
        should_keep = True
        for kept_detection in keep:
            _, ky, kx, kr = kept_detection
            distance = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
            if distance < (radius + kr) / 2:
                should_keep = False
                break

        if should_keep:
            keep.append(detection)

    # Convert back to arrays
    if keep:
        responses = np.array([d[0] for d in keep])
        y_coords = np.array([d[1] for d in keep])
        x_coords = np.array([d[2] for d in keep])
        return responses, y_coords, x_coords
    else:
        return np.array([]), np.array([]), np.array([])


def CalculateBlobMetrics(image_data, x, y, radius):
    """
    Calculate various metrics for a detected blob
    """
    height, width = image_data.shape

    # Define blob region
    x_min = max(0, int(x - radius * 2))
    x_max = min(width, int(x + radius * 2) + 1)
    y_min = max(0, int(y - radius * 2))
    y_max = min(height, int(y + radius * 2) + 1)

    if x_max <= x_min or y_max <= y_min:
        return {'area': 0, 'volume': 0, 'height': 0}

    # Extract region
    region = image_data[y_min:y_max, x_min:x_max]

    # Create circular mask
    y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    center_x = x - x_min
    center_y = y - y_min

    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    mask = distances <= radius

    if not np.any(mask):
        return {'area': 0, 'volume': 0, 'height': 0}

    blob_pixels = region[mask]
    background = np.mean(region[~mask]) if np.any(~mask) else 0

    # Calculate metrics
    area = np.sum(mask)  # Number of pixels
    height = np.max(blob_pixels) - background
    volume = np.sum(blob_pixels - background)

    return {
        'area': area,
        'volume': max(0, volume),
        'height': max(0, height)
    }