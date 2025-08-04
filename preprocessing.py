"""
Preprocessing Module
Contains image preprocessing functions for blob detection
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from scipy import ndimage
from pathlib import Path

from igor_compatibility import *
from file_io import *
from utilities import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def RemoveStreaks(im, sigma=3):
    """
    Remove horizontal streaks from image

    Parameters:
    im : Wave - Input image to process
    sigma : float - Number of standard deviations for streak identification
    """
    print(f"Removing streaks with sigma = {sigma}")

    if sigma <= 0:
        return  # Skip if disabled

    data = im.data
    height, width = data.shape

    # Calculate streakiness for each row
    streakiness = np.zeros(height)

    for i in range(height):
        row = data[i, :]

        # Calculate horizontal variation (streakiness measure)
        diff = np.diff(row)
        streakiness[i] = np.std(diff)

    # Find mean and std of streakiness
    mean_streak = np.mean(streakiness)
    std_streak = np.std(streakiness)

    # Identify streaky rows
    threshold = mean_streak + sigma * std_streak
    streaky_rows = streakiness > threshold

    print(f"Found {np.sum(streaky_rows)} streaky rows out of {height}")

    # Interpolate streaky rows from neighboring good rows
    for i in range(height):
        if streaky_rows[i]:
            # Find nearest non-streaky rows
            above = i - 1
            below = i + 1

            # Search for good row above
            while above >= 0 and streaky_rows[above]:
                above -= 1

            # Search for good row below
            while below < height and streaky_rows[below]:
                below += 1

            # Interpolate from available good rows
            if above >= 0 and below < height:
                # Interpolate between above and below
                weight_above = (below - i) / (below - above)
                weight_below = (i - above) / (below - above)
                im.data[i, :] = weight_above * data[above, :] + weight_below * data[below, :]
            elif above >= 0:
                # Use row above
                im.data[i, :] = data[above, :]
            elif below < height:
                # Use row below
                im.data[i, :] = data[below, :]

    print("Streak removal complete")


def Flatten(im, order):
    """
    Flatten image by subtracting polynomial fit to each row

    Parameters:
    im : Wave - Input image to process
    order : int - Polynomial order for fitting
    """
    print(f"Flattening image with polynomial order = {order}")

    if order <= 0:
        return  # Skip if disabled

    data = im.data
    height, width = data.shape

    # Process each row
    for i in range(height):
        row = data[i, :]

        # Create x coordinates for fitting
        x = np.arange(width)

        try:
            # Fit polynomial to the row
            coeffs = np.polyfit(x, row, order)

            # Calculate polynomial background
            background = np.polyval(coeffs, x)

            # Subtract background from row
            im.data[i, :] = row - background

        except np.linalg.LinAlgError:
            # If fitting fails, skip this row
            continue

    print("Flattening complete")


def BatchPreprocess():
    """Batch preprocessing interface for multiple image files"""

    class PreprocessingGUI:
        def __init__(self):
            self.root = tk.Toplevel()
            self.root.title("Batch Image Preprocessing")
            self.root.geometry("800x600")
            self.root.transient()
            self.root.grab_set()

            self.input_files = []
            self.output_folder = ""

            self.setup_ui()

        def setup_ui(self):
            """Setup the preprocessing GUI"""
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            ttk.Label(main_frame, text="Batch Image Preprocessing",
                      font=('TkDefaultFont', 14, 'bold')).pack(pady=(0, 20))

            # Input files section
            input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
            input_frame.pack(fill=tk.X, pady=(0, 10))

            buttons_frame = ttk.Frame(input_frame)
            buttons_frame.pack(fill=tk.X)

            ttk.Button(buttons_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
            ttk.Button(buttons_frame, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
            ttk.Button(buttons_frame, text="Clear", command=self.clear_files).pack(side=tk.LEFT, padx=5)

            # File list
            list_frame = ttk.Frame(input_frame)
            list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

            self.file_listbox = tk.Listbox(list_frame, height=6)
            scrollbar_files = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
            self.file_listbox.configure(yscrollcommand=scrollbar_files.set)

            self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar_files.pack(side=tk.RIGHT, fill=tk.Y)

            # Preprocessing parameters
            params_frame = ttk.LabelFrame(main_frame, text="Preprocessing Parameters", padding="10")
            params_frame.pack(fill=tk.X, pady=(0, 10))

            # Streak removal parameters
            streak_frame = ttk.Frame(params_frame)
            streak_frame.pack(fill=tk.X, pady=5)

            ttk.Label(streak_frame, text="Std. Deviations for streak removal (0 = disable):").pack(side=tk.LEFT)
            self.streak_sdevs_var = tk.DoubleVar(value=3)
            ttk.Entry(streak_frame, textvariable=self.streak_sdevs_var, width=10).pack(side=tk.LEFT, padx=5)

            # Flattening parameters
            flatten_frame = ttk.Frame(params_frame)
            flatten_frame.pack(fill=tk.X, pady=5)

            ttk.Label(flatten_frame, text="Polynomial order for flattening (0 = disable):").pack(side=tk.LEFT)
            self.flatten_order_var = tk.IntVar(value=2)
            ttk.Entry(flatten_frame, textvariable=self.flatten_order_var, width=10).pack(side=tk.LEFT, padx=5)

            # Output folder section
            output_frame = ttk.LabelFrame(main_frame, text="Output Folder", padding="10")
            output_frame.pack(fill=tk.X, pady=(0, 10))

            output_buttons_frame = ttk.Frame(output_frame)
            output_buttons_frame.pack(fill=tk.X)

            ttk.Button(output_buttons_frame, text="Select Output Folder",
                       command=self.select_output_folder).pack(side=tk.LEFT, padx=5)

            self.output_label = ttk.Label(output_frame, text="No output folder selected")
            self.output_label.pack(anchor=tk.W, pady=(5, 0))

            # Progress and control
            progress_frame = ttk.Frame(main_frame)
            progress_frame.pack(fill=tk.X, pady=(0, 10))

            self.progress = ttk.Progressbar(progress_frame, mode='determinate')
            self.progress.pack(fill=tk.X, pady=(0, 5))

            self.status_label = ttk.Label(progress_frame, text="Ready")
            self.status_label.pack(anchor=tk.W)

            # Control buttons
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X)

            ttk.Button(control_frame, text="Start Preprocessing",
                       command=self.start_preprocessing).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Close",
                       command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

        def add_files(self):
            """Add individual files"""
            files = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=[
                    ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.ibw"),
                    ("All files", "*.*")
                ]
            )

            for file_path in files:
                if file_path not in self.input_files:
                    self.input_files.append(file_path)
                    self.file_listbox.insert(tk.END, Path(file_path).name)

        def add_folder(self):
            """Add all image files from a folder"""
            folder_path = filedialog.askdirectory(title="Select Image Folder")

            if folder_path:
                supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.ibw']

                for file_path in Path(folder_path).iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        full_path = str(file_path)
                        if full_path not in self.input_files:
                            self.input_files.append(full_path)
                            self.file_listbox.insert(tk.END, file_path.name)

        def clear_files(self):
            """Clear file list"""
            self.input_files.clear()
            self.file_listbox.delete(0, tk.END)

        def select_output_folder(self):
            """Select output folder"""
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self.output_folder = folder
                self.output_label.config(text=f"Output: {folder}")

        def start_preprocessing(self):
            """Start batch preprocessing"""
            if not self.input_files:
                messagebox.showwarning("No Files", "Please add files to preprocess.")
                return

            if not self.output_folder:
                messagebox.showwarning("No Output", "Please select an output folder.")
                return

            try:
                total_files = len(self.input_files)
                self.progress['maximum'] = total_files

                streak_sdevs = self.streak_sdevs_var.get()
                flatten_order = self.flatten_order_var.get()

                print(f"Starting batch preprocessing of {total_files} files")
                print(f"Parameters: streak_sdevs={streak_sdevs}, flatten_order={flatten_order}")

                for i, file_path in enumerate(self.input_files):
                    self.status_label.config(text=f"Processing {Path(file_path).name}...")
                    self.root.update_idletasks()

                    try:
                        # Load image
                        im = LoadWave(file_path)
                        if im is None:
                            print(f"Failed to load {file_path}")
                            continue

                        # Apply preprocessing
                        if streak_sdevs > 0:
                            RemoveStreaks(im, sigma=streak_sdevs)

                        if flatten_order > 0:
                            Flatten(im, flatten_order)

                        # Save processed image to selected output folder
                        input_filename = Path(file_path).stem  # Get filename without extension
                        output_filename = f"{input_filename}_preprocessed.npy"
                        output_path = Path(self.output_folder) / output_filename

                        print(f"DEBUG Batch: Saving {output_filename} to {self.output_folder}")
                        print(f"DEBUG Batch: Full output path: {output_path}")
                        print(f"DEBUG Batch: Output folder exists: {Path(self.output_folder).exists()}")

                        # Ensure output directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        print(f"DEBUG Batch: Parent directory exists after mkdir: {output_path.parent.exists()}")

                        # Save as numpy array with proper error handling
                        try:
                            print(f"DEBUG Batch: About to save data of shape {im.data.shape} and type {type(im.data)}")
                            np.save(str(output_path), im.data)
                            print(f"DEBUG Batch: numpy save completed")

                            # Small delay for file system sync
                            import time
                            time.sleep(0.1)

                            if output_path.exists():
                                file_size = output_path.stat().st_size
                                print(f"SUCCESS: Saved {output_filename} ({file_size} bytes)")
                                print(f"DEBUG Batch: File exists at: {output_path.absolute()}")
                            else:
                                print(f"ERROR: Failed to save {output_filename}")
                                print(f"DEBUG Batch: File absolute path: {output_path.absolute()}")
                                print(
                                    f"DEBUG Batch: Parent directory contents: {list(output_path.parent.iterdir()) if output_path.parent.exists() else 'Parent does not exist'}")
                                raise IOError(f"File was not created: {output_path}")
                        except Exception as save_error:
                            print(f"ERROR saving {output_filename}: {save_error}")
                            import traceback
                            print(f"DEBUG Batch: Full traceback: {traceback.format_exc()}")
                            raise

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

                    self.progress['value'] = i + 1
                    self.root.update_idletasks()

                self.status_label.config(text=f"Complete! Processed {total_files} files.")
                messagebox.showinfo("Complete", f"Preprocessing complete! Processed {total_files} files.")

            except Exception as e:
                messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
                self.status_label.config(text="Error occurred")

    # Create and show the GUI
    try:
        gui = PreprocessingGUI()
        return gui
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open preprocessing interface: {str(e)}")
        return None


def GroupPreprocess(images_dict):
    """
    Group preprocessing for loaded images
    Apply preprocessing to all loaded images in memory
    """
    if not images_dict:
        messagebox.showwarning("No Images", "No images loaded for preprocessing.")
        return

    # Get preprocessing parameters
    result = get_preprocessing_params()
    if result is None:
        return

    streak_sdevs, flatten_order = result

    try:
        total_images = len(images_dict)
        processed = 0

        print(f"Starting group preprocessing of {total_images} images")
        print(f"Parameters: streak_sdevs={streak_sdevs}, flatten_order={flatten_order}")

        for image_name, wave in images_dict.items():
            print(f"Preprocessing {image_name}...")

            # Apply preprocessing
            if streak_sdevs > 0:
                RemoveStreaks(wave, sigma=streak_sdevs)

            if flatten_order > 0:
                Flatten(wave, flatten_order)

            processed += 1

        messagebox.showinfo("Complete", f"Group preprocessing complete! Processed {processed} images.")
        print(f"Group preprocessing complete: {processed} images processed")

    except Exception as e:
        messagebox.showerror("Error", f"Group preprocessing failed: {str(e)}")
        print(f"Group preprocessing error: {str(e)}")


def get_preprocessing_params():
    """Get preprocessing parameters from user"""
    # Create parameter dialog
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel()
    dialog.title("Preprocessing Parameters")
    dialog.geometry("600x300")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Preprocessing Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 20))

    # Parameters frame
    params_frame = ttk.Frame(main_frame)
    params_frame.pack(fill=tk.X, pady=10)

    # Streak removal
    streak_frame = ttk.Frame(params_frame)
    streak_frame.pack(fill=tk.X, pady=5)

    ttk.Label(streak_frame, text="Std. Deviations for streak removal (0 = disable):").pack(side=tk.LEFT)
    streak_sdevs_var = tk.DoubleVar(value=3)  # Igor Pro default
    ttk.Entry(streak_frame, textvariable=streak_sdevs_var, width=10).pack(side=tk.LEFT, padx=5)

    # Flattening parameters
    flatten_frame = ttk.Frame(params_frame)
    flatten_frame.pack(fill=tk.X, pady=5)

    ttk.Label(flatten_frame, text="Polynomial order for flattening (0 = disable):").pack(side=tk.LEFT)
    flatten_order_var = tk.IntVar(value=2)  # Igor Pro default
    ttk.Entry(flatten_frame, textvariable=flatten_order_var, width=10).pack(side=tk.LEFT, padx=5)

    def ok_clicked():
        result[0] = (streak_sdevs_var.get(), flatten_order_var.get())
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    dialog.wait_window()
    return result[0]


def SimpleContrastStretching(image):
    """
    Apply simple contrast stretching to image

    Parameters:
    image : Wave - Input image to process
    """
    print("Applying contrast stretching...")

    try:
        # Use skimage if available
        from skimage import exposure
        image.data = exposure.rescale_intensity(image.data)
        print("Applied histogram equalization using skimage")
    except ImportError:
        # Fallback: simple min-max stretching
        data_min = np.min(image.data)
        data_max = np.max(image.data)
        image.data = (image.data - data_min) / (data_max - data_min)
        print("Applied simple contrast stretching (skimage not available)")


def CreateMask(image, threshold_method='otsu', threshold_value=None):
    """
    Create binary mask from image

    Parameters:
    image : Wave - Input image
    threshold_method : str - Thresholding method
    threshold_value : float - Manual threshold value

    Returns:
    Wave - Binary mask
    """
    if threshold_value is not None:
        threshold = threshold_value
    elif threshold_method == 'otsu':
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(image.data)
        except ImportError:
            # Fallback
            threshold = np.mean(image.data) + np.std(image.data)
    elif threshold_method == 'mean':
        threshold = np.mean(image.data)
    else:
        threshold = 0.5 * (np.min(image.data) + np.max(image.data))

    mask_data = image.data > threshold
    mask = Wave(mask_data.astype(np.uint8), f"{image.name}_mask")

    print(f"Created mask using {threshold_method} threshold: {threshold:.6f}")
    return mask

