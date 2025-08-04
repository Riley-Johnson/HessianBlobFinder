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

from igor_compatibility import *
from file_io import *
from utilities import *
from scale_space import *



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
    Extract blob information from maxima maps
    """
    print("Extracting blob information...")

    # Find pixels above threshold
    valid_pixels = np.where(SS_MAXMAP.data >= min_response)

    if len(valid_pixels[0]) == 0:
        print("No blobs found above threshold")
        empty_info = Wave(np.zeros((0, 13)), "info")
        return empty_info

    num_blobs = len(valid_pixels[0])
    blob_info = np.zeros((num_blobs, 13))

    print(f"Found {num_blobs} candidate blobs")

    for idx, (i, j) in enumerate(zip(valid_pixels[0], valid_pixels[1])):
        # Get blob information
        x_coord = j  # Column index -> x coordinate
        y_coord = i  # Row index -> y coordinate
        response = SS_MAXMAP.data[i, j]
        scale = SS_MAXSCALEMAP.data[i, j] if SS_MAXSCALEMAP is not None else 1.0

        radius = np.sqrt(2 * scale)

        blob_info[idx, 0] = x_coord  # X position
        blob_info[idx, 1] = y_coord  # Y position
        blob_info[idx, 2] = radius  # Radius
        blob_info[idx, 3] = response  # Response strength
        blob_info[idx, 4] = scale  # Scale
        # Other columns can be filled with additional measurements

    # Filter overlapping blobs if not allowed
    if allowOverlap == 0:
        blob_info = filter_overlapping_blobs(blob_info)

    print(f"Final blob count after filtering: {blob_info.shape[0]}")

    # Create output wave
    info_wave = Wave(blob_info, "info")
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
    # Create parameter dialog
    root = tk.Tk()
    root.withdraw()  # Hide main window

    dialog = tk.Toplevel()
    dialog.title("Hessian Blob Parameters")
    dialog.geometry("700x400")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    # Make dialog more visible
    dialog.lift()  # Bring to front
    dialog.attributes('-topmost', True)  # Keep on top temporarily
    dialog.after(100, lambda: dialog.attributes('-topmost', False))  # Remove topmost after 100ms

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Hessian Blob Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Scale parameters
    scale_frame = ttk.LabelFrame(main_frame, text="Scale-Space Parameters", padding="10")
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
    detect_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
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

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    ttk.Button(button_frame, text="Continue", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    dialog.wait_window()

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
            constraint_dialog.transient()
            constraint_dialog.grab_set()
            constraint_dialog.focus_set()

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
            constraint_button_frame.pack(side=tk.BOTTOM, pady=15)

            ttk.Button(constraint_button_frame, text="Continue", command=constraint_ok_clicked).pack(side=tk.LEFT,
                                                                                                     padx=5)
            ttk.Button(constraint_button_frame, text="Cancel", command=constraint_cancel_clicked).pack(side=tk.LEFT,
                                                                                                       padx=5)

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
                print(f"DEBUG: Calculated measurements for {info.data.shape[0]} interactive blobs")
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
                print("DEBUG: Forcing update_display to get blob info")
                self.update_display()  # Force update to get blob info

            # Store maps for later retrieval
            if hasattr(self, 'current_SS_MAXMAP') and self.current_SS_MAXMAP is not None:
                print("DEBUG: SS_MAXMAP available from interactive threshold")

            if hasattr(self, 'current_SS_MAXSCALEMAP') and self.current_SS_MAXSCALEMAP is not None:
                print("DEBUG: SS_MAXSCALEMAP available from interactive threshold")

            print(f"=== ACCEPT THRESHOLD DEBUG ===")
            print(f"Accepting threshold: {self.result}")
            print(f"Blob info exists: {self.current_blob_info is not None}")
            if self.current_blob_info:
                print(f"Blob info shape: {self.current_blob_info.data.shape}")
            print(f"===============================")
            print("DEBUG: About to quit mainloop and destroy root window...")
            self.root.quit()  # Exit mainloop first
            self.root.destroy()  # Then destroy window
            print("DEBUG: Root window quit and destroyed successfully")
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
        print("=== INTERACTIVE THRESHOLD DEBUG ===")
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

    # Initialize info wave for particle information
    info = Wave(np.zeros((1000, 13)), "info")

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

    print("Cropping and measuring particles..")

    # Variables for particle measurement calculations
    accepted_particles = 0

    # Process each potential particle
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
                (info.data[i, 4] <= 2 or info.data[i, 5] >= im.data.shape[0] - 3 or
                 info.data[i, 6] <= 2 or info.data[i, 7] >= im.data.shape[1] - 3)):
            continue

        # Extract particle region
        p_start, p_stop = int(info.data[i, 4]), int(info.data[i, 5])
        q_start, q_stop = int(info.data[i, 6]), int(info.data[i, 7])

        # Calculate basic measurements
        particle_area = info.data[i, 2]  # numPixels from Info wave
        particle_height = info.data[i, 3]  # maximum blob strength

        # Simple volume calculation
        particle_volume = particle_area * particle_height

        # Center of mass
        com_x = info.data[i, 0]  # P Seed
        com_y = info.data[i, 1]  # Q Seed

        # Average height
        avg_height = particle_height * 0.7

        # Check height constraints
        if particle_height < minH or particle_height > maxH:
            continue

        # Check area constraints
        if particle_area < minA or particle_area > maxA:
            continue

        # Check volume constraints
        if particle_volume < minV or particle_volume > maxV:
            continue

        # Particle accepted
        heights.data[accepted_particles] = particle_height
        areas.data[accepted_particles] = particle_area
        volumes.data[accepted_particles] = particle_volume
        avg_heights.data[accepted_particles] = avg_height
        com.data[accepted_particles, 0] = com_x
        com.data[accepted_particles, 1] = com_y

        # Mark particle as accepted
        info.data[i, 14] = accepted_particles + 1

        accepted_particles += 1

    # Resize measurement waves to accepted particles only
    if accepted_particles < numPotentialParticles:
        heights.data = heights.data[:accepted_particles]
        areas.data = areas.data[:accepted_particles]
        volumes.data = volumes.data[:accepted_particles]
        avg_heights.data = avg_heights.data[:accepted_particles]
        com.data = com.data[:accepted_particles, :]
        print(f"Applied constraints: {accepted_particles} particles accepted out of {numPotentialParticles}")

    # Return data folder path and all waves

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

        # Store results for this image
        image_results[image_name] = image_df_results

        # Get wave references to the measurement waves
        if image_df_results is None:
            print(f"   ERROR: Analysis failed for {image_name}")
            continue

        heights = image_df_results.get('Heights')
        avg_heights = image_df_results.get('AvgHeights')
        areas = image_df_results.get('Areas')
        volumes = image_df_results.get('Volumes')
        com = image_df_results.get('COM')  # Center of mass data

        # Debug: Check what was returned from individual analysis
        print(f"   Individual analysis results for {image_name}:")
        print(f"   - Heights: {len(heights.data) if heights else 'None'}")
        print(f"   - Areas: {len(areas.data) if areas else 'None'}")
        print(f"   - Volumes: {len(volumes.data) if volumes else 'None'}")
        print(f"   - AvgHeights: {len(avg_heights.data) if avg_heights else 'None'}")
        print(f"   - COM: {com.data.shape if com else 'None'}")

        # Concatenate the measurements into the master wave
        if len(heights.data) > 0:
            # Proper wave concatenation with robust error handling
            try:
                # Use robust concatenation that handles empty arrays properly
                if len(all_heights.data) == 0:
                    all_heights.data = heights.data.copy()
                    all_avg_heights.data = avg_heights.data.copy()
                    all_areas.data = areas.data.copy()
                    all_volumes.data = volumes.data.copy()
                    # Handle COM data carefully - ensure it's a 2D array
                    if com and hasattr(com, 'data') and len(com.data) > 0:
                        all_com.data = com.data.copy()
                    else:
                        all_com.data = np.array([]).reshape(0, 2)
                else:
                    all_heights.data = np.concatenate([all_heights.data, heights.data])
                    all_avg_heights.data = np.concatenate([all_avg_heights.data, avg_heights.data])
                    all_areas.data = np.concatenate([all_areas.data, areas.data])
                    all_volumes.data = np.concatenate([all_volumes.data, volumes.data])
                    # Handle COM concatenation with proper error checking
                    if com and hasattr(com, 'data') and len(com.data) > 0:
                        try:
                            if len(all_com.data) == 0:
                                all_com.data = com.data.copy()
                            else:
                                all_com.data = np.vstack([all_com.data, com.data])
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
        else:
            print(f"   No particles found in {image_name}")

        # Memory management for large batch jobs
        if i % 10 == 0 and i > 0:
            print(f"Progress: Processed {i + 1}/{num_images} images")

    # Determine the total number of particles
    num_particles = len(all_heights.data)
    print(f"  Series complete. Total particles detected: {num_particles}")

    # Ensure AllCOM has proper 2D shape even if empty
    if len(all_com.data) == 0:
        all_com.data = np.array([]).reshape(0, 2)
        print(f"  AllCOM initialized as empty 2D array: shape {all_com.data.shape}")
    else:
        print(f"  AllCOM contains {len(all_com.data)} coordinate pairs: shape {all_com.data.shape}")

    # Return series data folder path and all results
    return {
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

    print(f"=== SAVE BATCH RESULTS DEBUG ===")
    print(f"Function called with:")
    print(f"  output_path: {output_path}")
    print(f"  save_format: {save_format}")
    print(f"  batch_results type: {type(batch_results)}")
    print(f"  batch_results keys: {list(batch_results.keys()) if batch_results else 'None'}")

    if not batch_results:
        print("ERROR: No batch results provided!")
        raise ValueError("No batch results provided to save")

    if not output_path:
        output_path = os.getcwd()
        print(f"Using current directory: {output_path}")

    if not os.path.exists(output_path):
        print(f"ERROR: Output path does not exist: {output_path}")
        raise ValueError(f"Output path does not exist: {output_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create Series_X folder structure
    series_num = 1
    series_folder_name = f"Series_{series_num}"
    while os.path.exists(os.path.join(output_path, series_folder_name)):
        series_num += 1
        series_folder_name = f"Series_{series_num}"

    series_path = os.path.join(output_path, series_folder_name)
    print(f"Creating Series folder: {series_path}")

    try:
        os.makedirs(series_path, exist_ok=True)
        print(f"Series folder created successfully: {os.path.exists(series_path)}")
    except Exception as e:
        print(f"ERROR: Failed to create Series folder: {e}")
        raise

    # Save Files in Series_X folder structure
    if save_format == "igor" or save_format == "txt":
        # Save Parameters wave
        params_file = os.path.join(series_path, "Parameters.txt")
        with open(params_file, 'w') as f:
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
            f.write(f"// Total Images: {batch_results['numImages']}\n")
            f.write(f"// Total Particles: {batch_results['numParticles']}\n")

        measurements = {
            'AllHeights': batch_results['AllHeights'],
            'AllVolumes': batch_results['AllVolumes'],
            'AllAreas': batch_results['AllAreas'],
            'AllAvgHeights': batch_results['AllAvgHeights']
        }

        # Add AllCOM only if it exists and has valid data
        if 'AllCOM' in batch_results and batch_results['AllCOM'] is not None:
            measurements['AllCOM'] = batch_results['AllCOM']
            print(f"Including AllCOM in export with {len(batch_results['AllCOM'].data)} entries")
        else:
            print("WARNING: AllCOM data missing from batch results, skipping COM export")

        for wave_name, wave in measurements.items():
            wave_file = os.path.join(series_path, f"{wave_name}.txt")
            with open(wave_file, 'w') as f:
                # Use Igor Pro wave format with proper number formatting
                if wave_name == 'AllCOM':
                    print(f"Saving AllCOM with shape: {wave.data.shape}")
                    try:
                        if len(wave.data) == 0:
                            # Handle empty COM data
                            f.write(f"{wave_name}[0][0]= {{}}\n")
                        else:
                            f.write(f"{wave_name}[0][0]= {{")
                            for i, row in enumerate(wave.data):
                                if i > 0:
                                    f.write(",")
                                # Format COM coordinates
                                x_formatted = format_igor_number(float(row[0]))
                                y_formatted = format_igor_number(float(row[1]))
                                f.write(f"{{{x_formatted},{y_formatted}}}")
                            f.write("}\n")
                        print(f"Successfully saved AllCOM with {len(wave.data)} coordinate pairs")
                    except Exception as e:
                        print(f"ERROR saving AllCOM: {e}")
                        print(f"AllCOM data type: {type(wave.data)}")
                        print(f"AllCOM data shape: {wave.data.shape if hasattr(wave.data, 'shape') else 'No shape'}")
                        # Fallback: save empty COM file
                        f.write(f"{wave_name}[0][0]= {{}}\n")
                        print(f"Saved empty AllCOM file as fallback")
                else:
                    # Handle 1D measurement data
                    f.write(f"{wave_name}[0]= {{")
                    for i, value in enumerate(wave.data):
                        if i > 0:
                            f.write(",")
                        # Use Igor Pro number formatting
                        f.write(format_igor_number(value))
                    f.write("}\n")

        # Create consolidated Info file for ViewParticles compatibility
        info_file = os.path.join(series_path, "Info.txt")
        with open(info_file, 'w') as f:
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

        # Count files actually saved
        files_saved = 5  # Parameters, AllHeights, AllVolumes, AllAreas, AllAvgHeights
        if 'AllCOM' in measurements:
            files_saved += 1
        files_saved += 1  # Info file

        print(f"Saved {files_saved} files matching Igor Pro BatchHessianBlobs structure:")
        print(f"  - Parameters: Configuration and metadata")
        print(f"  - AllHeights: {len(batch_results['AllHeights'].data)} values")
        print(f"  - AllVolumes: {len(batch_results['AllVolumes'].data)} values")
        print(f"  - AllAreas: {len(batch_results['AllAreas'].data)} values")
        print(f"  - AllAvgHeights: {len(batch_results['AllAvgHeights'].data)} values")
        if 'AllCOM' in batch_results and batch_results['AllCOM'] is not None:
            print(f"  - AllCOM: {len(batch_results['AllCOM'].data)} coordinate pairs")
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
            image_folder = os.path.join(series_path, f"{clean_image_name}_Particles")
            os.makedirs(image_folder, exist_ok=True)
            print(f"Created image folder: {clean_image_name}_Particles")

            # Save complete individual image analysis results (same as SaveSingleImageResults)
            if results:
                # Save all measurement waves for this image
                if 'Heights' in results and results['Heights']:
                    wave_file = os.path.join(image_folder, "Heights.txt")
                    with open(wave_file, 'w') as f:
                        f.write(f"Heights[0]= {{")
                        for i, value in enumerate(results['Heights'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'Areas' in results and results['Areas']:
                    wave_file = os.path.join(image_folder, "Areas.txt")
                    with open(wave_file, 'w') as f:
                        f.write(f"Areas[0]= {{")
                        for i, value in enumerate(results['Areas'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'Volumes' in results and results['Volumes']:
                    wave_file = os.path.join(image_folder, "Volumes.txt")
                    with open(wave_file, 'w') as f:
                        f.write(f"Volumes[0]= {{")
                        for i, value in enumerate(results['Volumes'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'AvgHeights' in results and results['AvgHeights']:
                    wave_file = os.path.join(image_folder, "AvgHeights.txt")
                    with open(wave_file, 'w') as f:
                        f.write(f"AvgHeights[0]= {{")
                        for i, value in enumerate(results['AvgHeights'].data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")

                if 'COM' in results and results['COM']:
                    wave_file = os.path.join(image_folder, "COM.txt")
                    with open(wave_file, 'w') as f:
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
                    info_file = os.path.join(image_folder, "Info.txt")
                    with open(info_file, 'w') as f:
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
                        particle_folder = os.path.join(image_folder, f"Particle_{i}")
                        os.makedirs(particle_folder, exist_ok=True)

                        # Save complete particle info
                        particle_info_file = os.path.join(particle_folder, "ParticleInfo.txt")
                        with open(particle_info_file, 'w') as f:
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
    print(f"Batch results saved to: {series_path}")
    print(f"Format: {save_format}")
    print(f"Series folder created: {series_folder_name}")
    print(f"Series folder exists: {os.path.exists(series_path)}")
    print(f"Files in series folder: {os.listdir(series_path) if os.path.exists(series_path) else 'N/A'}")
    print(f"=== SAVE BATCH RESULTS END ===")
    return series_path


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

    print(f"=== SAVE SINGLE IMAGE DEBUG ===")
    print(f"Function called with:")
    print(f"  image_name: {image_name}")
    print(f"  output_path: {output_path}")
    print(f"  save_format: {save_format}")
    print(f"  results type: {type(results)}")
    print(f"  results keys: {list(results.keys()) if results else 'None'}")

    if not results:
        print("ERROR: No results provided!")
        raise ValueError("No results provided to save")

    if not output_path:
        output_path = os.getcwd()
        print(f"Using current directory: {output_path}")

    if not os.path.exists(output_path):
        print(f"ERROR: Output path does not exist: {output_path}")
        raise ValueError(f"Output path does not exist: {output_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create ImageName_Particles folder structure
    folder_name = f"{image_name}_Particles"
    full_path = os.path.join(output_path, folder_name)
    print(f"Creating folder: {full_path}")

    try:
        os.makedirs(full_path, exist_ok=True)
        print(f"Folder created successfully: {os.path.exists(full_path)}")
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
            wave_file = os.path.join(full_path, f"{wave_name}.txt")
            print(f"Saving {wave_name} to {wave_file}")

            if wave is None:
                print(f"ERROR: Wave {wave_name} is None!")
                continue

            if not hasattr(wave, 'data'):
                print(f"ERROR: Wave {wave_name} has no data attribute!")
                continue

            try:
                with open(wave_file, 'w') as f:
                    if wave_name == 'COM':
                        f.write(f"{wave_name}[0][0]= {{")
                        for i, row in enumerate(wave.data):
                            if i > 0:
                                f.write(",")
                            x_formatted = format_igor_number(float(row[0]))
                            y_formatted = format_igor_number(float(row[1]))
                            f.write(f"{{{x_formatted},{y_formatted}}}")
                        f.write("}\n")
                    else:
                        f.write(f"{wave_name}[0]= {{")
                        for i, value in enumerate(wave.data):
                            if i > 0:
                                f.write(",")
                            f.write(format_igor_number(float(value)))
                        f.write("}\n")
                print(f"Successfully wrote {wave_name} ({os.path.getsize(wave_file)} bytes)")
            except Exception as e:
                print(f"ERROR: Failed to write {wave_name}: {e}")
                raise

        # Save Info.txt file for ViewParticles compatibility
        info_file = os.path.join(full_path, "Info.txt")
        info_data = results['info'].data
        with open(info_file, 'w') as f:
            f.write("Info[0][0]= {")
            for i, row in enumerate(info_data):
                if i > 0:
                    f.write(",")
                # Format each value in the row
                formatted_values = [format_igor_number(float(val)) for val in row]
                f.write("{" + ",".join(formatted_values) + "}")
            f.write("}\n")
        print(f"Successfully wrote Info.txt ({os.path.getsize(info_file)} bytes)")

        # Create individual particle folders (Particle_0, Particle_1, etc.)
        info_data = results['info'].data
        print(f"Creating particle folders for {info_data.shape[0]} particles")

        # Check if measurement waves have data
        heights_data = results['Heights'].data if 'Heights' in results and len(results['Heights'].data) > 0 else []
        areas_data = results['Areas'].data if 'Areas' in results and len(results['Areas'].data) > 0 else []
        volumes_data = results['Volumes'].data if 'Volumes' in results and len(results['Volumes'].data) > 0 else []
        com_data = results['COM'].data if 'COM' in results and results['COM'].data.shape[0] > 0 else []

        print(
            f"Measurement wave sizes: Heights={len(heights_data)}, Areas={len(areas_data)}, Volumes={len(volumes_data)}, COM={len(com_data)}")

        for i in range(info_data.shape[0]):
            particle_folder = os.path.join(full_path, f"Particle_{i}")
            os.makedirs(particle_folder, exist_ok=True)

            # Save particle measurements in each particle folder with bounds checking
            particle_info_file = os.path.join(particle_folder, "ParticleInfo.txt")
            with open(particle_info_file, 'w') as f:
                f.write(f"Particle {i} Information\n")
                f.write(f"Height: {heights_data[i] if i < len(heights_data) else 'N/A'}\n")
                f.write(f"Area: {areas_data[i] if i < len(areas_data) else 'N/A'}\n")
                f.write(f"Volume: {volumes_data[i] if i < len(volumes_data) else 'N/A'}\n")

                # Safe COM access with bounds checking
                if i < len(com_data) and len(com_data[i]) >= 2:
                    f.write(f"Center: ({com_data[i][0]:.2f}, {com_data[i][1]:.2f})\n")
                else:
                    f.write("Center: N/A\n")

        # Save Info wave in main folder
        info_file = os.path.join(full_path, "Info.txt")
        with open(info_file, 'w') as f:
            f.write("Particle Detection Results\n")
            f.write(f"Image: {image_name}\n")
            f.write(f"Particles Found: {results['numParticles']}\n")
            f.write("Columns: P_Seed, Q_Seed, NumPixels, MaxBlobStrength, pStart, pStop, qStart, qStop, ")
            f.write("scale, layer, maximal, parentBlob, numBlobs, unused, particleNumber\n")
            info_data = results['info'].data
            for row in info_data:
                f.write("\t".join(map(str, row)) + "\n")

        # Save measurement waves
        measurements = {
            'Heights': results['Heights'],
            'Areas': results['Areas'],
            'Volumes': results['Volumes'],
            'AvgHeights': results['AvgHeights'],
            'COM': results['COM']
        }

        for wave_name, wave in measurements.items():
            wave_file = os.path.join(full_path, f"{wave_name}.txt")
            with open(wave_file, 'w') as f:
                f.write(f"Wave: {wave_name}\n")
                f.write(f"Image: {image_name}\n")
                if wave_name == 'COM':
                    f.write("Columns: X_Center, Y_Center\n")
                    for row in wave.data:
                        f.write(f"{row[0]}\t{row[1]}\n")
                else:
                    f.write("Data:\n")
                    for value in wave.data:
                        f.write(f"{value}\n")

        # Save analysis summary
        summary_file = os.path.join(full_path, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Hessian Blob Analysis Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Image: {image_name}\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Particles: {results['numParticles']}\n\n")

            # Statistics if available
            if 'Heights' in results and len(results['Heights'].data) > 0:
                heights = results['Heights'].data
                areas = results['Areas'].data
                volumes = results['Volumes'].data

                f.write("Particle Statistics:\n")
                f.write(f"Mean Height: {np.mean(heights):.3f}\n")
                f.write(f"Mean Area: {np.mean(areas):.3f}\n")
                f.write(f"Mean Volume: {np.mean(volumes):.3f}\n")
                f.write(f"Height Range: {np.min(heights):.3f} - {np.max(heights):.3f}\n")
                f.write(f"Area Range: {np.min(areas):.3f} - {np.max(areas):.3f}\n")
                f.write(f"Volume Range: {np.min(volumes):.3f} - {np.max(volumes):.3f}\n")

        # Save analysis parameters
        params_file = os.path.join(full_path, "Parameters.txt")
        with open(params_file, 'w') as f:
            f.write(f"Hessian Blob Analysis Parameters for {image_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"scaleStart = {results['scaleStart']}\n")
            f.write(f"layers = {results['layers']}\n")
            f.write(f"scaleFactor = {results['scaleFactor']}\n")
            f.write(f"detHResponseThresh = {results['detHResponseThresh']}\n")
            f.write(f"particleType = {results['particleType']}\n")
            f.write(f"subPixelMult = {results['subPixelMult']}\n")
            f.write(f"allowOverlap = {results['allowOverlap']}\n")
            f.write(f"minH = {results['minH']}, maxH = {results['maxH']}\n")
            f.write(f"minA = {results['minA']}, maxA = {results['maxA']}\n")
            f.write(f"minV = {results['minV']}, maxV = {results['maxV']}\n")
            f.write(f"\nParticles found: {results['numParticles']}\n")

    elif save_format == "csv":
        # CSV format for Excel
        import csv
        csv_file = os.path.join(full_path, f"{image_name}_particles.csv")
        with open(csv_file, 'w', newline='') as f:
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
    print(f"Files in folder: {os.listdir(full_path) if os.path.exists(full_path) else 'N/A'}")
    print(f"=== SAVE SINGLE IMAGE END ===")
    return full_path


def ViewParticleData(info_wave, image_name, original_image=None):
    """ViewParticles implementation - scroll through individual blobs"""
    try:
        # Igor Pro: Check if particles exist
        if info_wave is None or info_wave.data.shape[0] == 0:
            messagebox.showwarning("No Particles", "No particles to view.")
            return

        print(f"DEBUG ViewParticleData: Creating viewer with {info_wave.data.shape[0]} particles")
        print(f"DEBUG ViewParticleData: Image name: {image_name}")
        print(f"DEBUG ViewParticleData: Original image type: {type(original_image)}")

        # Validate original image data
        if original_image is None:
            messagebox.showwarning("No Image Data", "Original image data is required for particle viewing.")
            return

        # Use the working ViewParticles function from particle_measurements.py
        from particle_measurements import ViewParticles
        ViewParticles(original_image, info_wave)

    except Exception as e:
        print(f"DEBUG ViewParticleData error: {str(e)}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("ViewParticles Error", f"Error creating particle viewer:\n{str(e)}")


class ParticleViewer:
    """Interactive particle viewer with measurement display"""

    def __init__(self, info_wave, image_name, original_image=None):
        try:
            print(f"DEBUG ParticleViewer init: Starting initialization")
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

            print(f"DEBUG ParticleViewer init: Creating window for {self.num_particles} particles")

            self.root = tk.Toplevel()
            self.root.title("Particle Viewer")
            self.root.geometry("900x600")
            self.root.transient()
            self.root.focus_set()

            print(f"DEBUG ParticleViewer init: Calling setup_viewer")
            self.setup_viewer()
            print(f"DEBUG ParticleViewer init: Initialization complete")

        except Exception as e:
            print(f"DEBUG ParticleViewer init error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def setup_viewer(self):
        """Setup the particle viewer interface"""
        try:
            print(f"DEBUG setup_viewer: Creating Igor Pro style layout")

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
            print(f"DEBUG setup_viewer error: {str(e)}")
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

