"""
Particle Measurement and Visualization
Comprehensive particle analysis, measurement calculation, and interactive viewing tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from scipy import ndimage

from igor_compatibility import *
from file_io import *
from utilities import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex

def load_saved_particle_data(data_path):
    """
    Load particle data from saved data files

    Parameters:
    data_path : str - Path to saved data folder

    Returns:
    im : Wave - Original image (or reconstructed if original not found)
    info : Wave - Particle information array
    """
    import os
    from utilities import Wave
    from file_io import LoadWave

    # Check if data path exists
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")

    # Try to load Info.txt (particle information)
    info_file = os.path.join(data_path, "Info.txt")
    if not os.path.exists(info_file):
        raise ValueError("No Info.txt file found in data folder")

    # Load particle information
    info_data = []
    with open(info_file, 'r') as f:
        content = f.read().strip()

        # Check if this is wave format or tab-delimited format
        if content.startswith('Info[0][0]='):
            # Wave format: Info[0][0]= {{val1,val2,val3},{val4,val5,val6}}
            try:
                # Extract the data part after the equals sign
                data_part = content.split('=', 1)[1].strip()

                # Remove outer braces
                if data_part.startswith('{') and data_part.endswith('}'):
                    data_part = data_part[1:-1]

                # Split by },{ to get individual particle data
                if data_part:
                    # Handle multiple particles
                    particle_strings = []
                    current_particle = ''
                    brace_count = 0

                    for char in data_part:
                        if char == '{':
                            brace_count += 1
                            if brace_count == 1:
                                current_particle = ''
                            else:
                                current_particle += char
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                particle_strings.append(current_particle)
                                current_particle = ''
                            else:
                                current_particle += char
                        elif brace_count > 0:
                            current_particle += char

                    # Parse each particle's data
                    for particle_str in particle_strings:
                        if particle_str.strip():
                            values = [float(v.strip()) for v in particle_str.split(',')]
                            if len(values) >= 4:  # At least X, Y, scale, response
                                info_data.append(values)

            except Exception as e:
                print(f"Warning: Failed to parse wave format: {e}")
                print(f"Content: {content[:200]}...")
        else:
            # Tab-delimited format (legacy compatibility)
            lines = content.split('\n')
            reading_data = False
            for line in lines:
                line = line.strip()
                if "P_Seed" in line and "Q_Seed" in line:  # Header line
                    reading_data = True
                    continue
                if reading_data and line:
                    try:
                        parts = line.split('\t')
                        if len(parts) >= 4:  # At least X, Y, scale, response
                            info_data.append([float(p) for p in parts])
                    except ValueError:
                        continue

    if not info_data:
        raise ValueError("No particle data found in Info.txt")

    info = Wave(np.array(info_data), "info")

    # Try to find and load the original image file
    # Look for common image formats in the data folder and parent folders
    original_image = None
    folder_name = os.path.basename(data_path)
    print(f"DEBUG: Looking for original image for folder: {folder_name}")

    # Try to extract image name from folder name (e.g., "ImageName_Particles" -> "ImageName")
    image_basename = None
    if "_Particles" in folder_name:
        image_basename = folder_name.replace("_Particles", "")
    elif folder_name.startswith("Series_"):
        # Handle Series_X folder structure - look for any image files
        parent_dir = os.path.dirname(data_path)
        print(f"DEBUG: Series folder detected, searching parent: {parent_dir}")
        image_extensions = ['.ibw', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.npy']
        for file in os.listdir(parent_dir):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_path = os.path.join(parent_dir, file)
                try:
                    original_image = LoadWave(image_path)
                    if original_image is not None:
                        print(f"DEBUG: Found original image in series folder: {image_path}")
                        break
                except Exception as e:
                    print(f"DEBUG: Failed to load series image {image_path}: {e}")
                    continue

    # If we have a basename, search for the specific image
    if image_basename and original_image is None:
        print(f"DEBUG: Searching for image with basename: {image_basename}")
        # Look in the data folder and parent folder for image files
        search_folders = [data_path, os.path.dirname(data_path)]
        image_extensions = ['.ibw', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.npy']

        for search_folder in search_folders:
            print(f"DEBUG: Searching in folder: {search_folder}")
            for ext in image_extensions:
                image_path = os.path.join(search_folder, image_basename + ext)
                if os.path.exists(image_path):
                    print(f"DEBUG: Found potential image file: {image_path}")
                    try:
                        original_image = LoadWave(image_path)
                        if original_image is not None:
                            print(f"DEBUG: Successfully loaded original image: {image_path}")
                            print(
                                f"DEBUG: Image shape: {original_image.data.shape}, dtype: {original_image.data.dtype}")
                            print(
                                f"DEBUG: Image stats: min={np.min(original_image.data)}, max={np.max(original_image.data)}")
                            break
                    except Exception as e:
                        print(f"DEBUG: Failed to load {image_path}: {e}")
                        continue
            if original_image is not None:
                break

    # If original image found, use it
    if original_image is not None:
        im = original_image
    else:
        print("Original image not found, creating synthetic representation")
        if info_data:
            max_x = max(row[0] for row in info_data)
            max_y = max(row[1] for row in info_data)
            image_size = (int(max_y + 50), int(max_x + 50))  # Add some padding
            im_data = np.ones(image_size) * 100  # Create gray background instead of black

            # Add synthetic particle representations for visualization
            for row in info_data:
                x_pos, y_pos = int(row[0]), int(row[1])
                radius = int(row[2]) if len(row) > 2 else 5

                # Create a simple circular region around each particle
                y_coords, x_coords = np.ogrid[:image_size[0], :image_size[1]]
                distance = np.sqrt((x_coords - x_pos) ** 2 + (y_coords - y_pos) ** 2)
                particle_mask = distance <= radius
                im_data[particle_mask] = 255  # Bright particles on gray background

            im = Wave(im_data, "ReconstructedImage")
        else:
            # Create a gray background for empty image
            im = Wave(np.ones((100, 100)) * 128, "EmptyImage")

    return im, info


def ViewParticles(im, info, mapNum=None, saved_data_path=None):
    """
    Interactive particle viewer and measurement display
    Works with both live analysis data and saved format files

    Parameters:
    im : Wave - Original image (or None if loading from saved data)
    info : Wave - Particle information array (or None if loading from saved data)
    mapNum : Wave - Particle number map (optional)
    saved_data_path : str - Path to saved data folder (optional)
    """
    # Load from saved data if path provided
    if saved_data_path is not None:
        try:
            im, info = load_saved_particle_data(saved_data_path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load saved particle data:\n{str(e)}")
            return

    # Validation
    if im is None:
        messagebox.showerror("Error", "No image provided.")
        return

    if info is None:
        messagebox.showerror("Error", "No particle information provided.")
        return

    if not hasattr(info, 'data') or info.data is None:
        messagebox.showerror("Error", "Particle information has no data attribute.")
        return

    if info.data.shape[0] == 0:
        messagebox.showinfo("No Particles", "No particles to view.")
        return

    class ParticleViewer:
        def __init__(self, im, info):
            self.im = im
            self.info = info
            self.current_particle = 0
            self.total_particles = info.data.shape[0]

            # Viewer settings and defaults
            self.color_table = "Grays"
            self.interpolate = False
            self.show_perimeter = True

            self.create_window()
            self.update_display()

        def create_window(self):
            """Create particle viewer window with optimal layout"""
            self.root = tk.Toplevel()
            self.root.title("Particle Viewer")  # Matches Igor Pro title
            self.root.geometry("1000x700")

            # Main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Left side: Image display
            left_frame = ttk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create matplotlib figure
            self.figure = Figure(figsize=(8, 8), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Right side: Controls panel
            right_frame = ttk.LabelFrame(main_frame, text="Controls", width=200)
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
            right_frame.pack_propagate(False)

            # Particle name/number
            self.particle_label = ttk.Label(right_frame,
                                            text=f"Particle {self.current_particle}",
                                            font=('TkDefaultFont', 14, 'bold'))
            self.particle_label.pack(pady=(10, 15))

            # Navigation buttons
            nav_frame = ttk.Frame(right_frame)
            nav_frame.pack(pady=(0, 15))

            self.prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.prev_particle)
            self.prev_btn.pack(side=tk.LEFT, padx=(0, 5))

            self.next_btn = ttk.Button(nav_frame, text="Next →", command=self.next_particle)
            self.next_btn.pack(side=tk.LEFT)

            # Particle information display
            info_frame = ttk.LabelFrame(right_frame, text="Particle Info", padding="5")
            info_frame.pack(fill=tk.X, pady=(0, 15))

            self.info_text = tk.Text(info_frame, height=8, width=25, font=("Courier", 9))
            self.info_text.pack(fill=tk.BOTH, expand=True)

            # Control buttons
            control_frame = ttk.LabelFrame(right_frame, text="View Controls", padding="5")
            control_frame.pack(fill=tk.X, pady=(0, 15))

            # Show perimeter checkbox
            self.show_perimeter_var = tk.BooleanVar(value=self.show_perimeter)
            ttk.Checkbutton(control_frame, text="Show Perimeter",
                            variable=self.show_perimeter_var,
                            command=self.toggle_perimeter).pack(anchor=tk.W, pady=2)

            # Color table selector
            ttk.Label(control_frame, text="Color Table:").pack(anchor=tk.W, pady=(10, 2))
            self.color_var = tk.StringVar(value=self.color_table)
            color_combo = ttk.Combobox(control_frame, textvariable=self.color_var,
                                       values=["Grays", "Rainbow", "Hot", "Cool"],
                                       width=15, state="readonly")
            color_combo.pack(anchor=tk.W, pady=(0, 5))
            color_combo.bind('<<ComboboxSelected>>', self.change_color_table)

            # Action buttons
            action_frame = ttk.Frame(right_frame)
            action_frame.pack(fill=tk.X, pady=(0, 10))

            ttk.Button(action_frame, text="Delete Particle",
                       command=self.delete_particle).pack(fill=tk.X, pady=2)
            ttk.Button(action_frame, text="Close",
                       command=self.root.destroy).pack(fill=tk.X, pady=2)

            # Bind keyboard shortcuts
            self.root.bind('<Left>', lambda e: self.prev_particle())
            self.root.bind('<Right>', lambda e: self.next_particle())
            self.root.bind('<space>', lambda e: self.delete_particle())
            self.root.focus_set()

        def update_display(self):
            """Update the particle display"""
            if self.total_particles == 0:
                return

            # Get current particle info
            particle_data = self.info.data[self.current_particle]
            x_coord = particle_data[0]
            y_coord = particle_data[1]
            radius = particle_data[2]

            # Update particle label
            self.particle_label.config(text=f"Particle {self.current_particle}")

            # Clear and update plot
            self.ax.clear()

            # Calculate crop region around particle
            crop_size = max(50, int(radius * 4))
            x_min = max(0, int(x_coord - crop_size))
            x_max = min(self.im.data.shape[1], int(x_coord + crop_size))
            y_min = max(0, int(y_coord - crop_size))
            y_max = min(self.im.data.shape[0], int(y_coord + crop_size))

            # Crop image
            cropped_image = self.im.data[y_min:y_max, x_min:x_max]

            # Check if cropped image has valid data
            if cropped_image.size == 0:
                # Fallback to full image if crop failed
                cropped_image = self.im.data
                extent = [0, self.im.data.shape[1], self.im.data.shape[0], 0]
            else:
                # Display cropped image
                extent = [x_min, x_max, y_max, y_min]  # Note: y is flipped for imshow

            color_map = self.color_var.get().lower()
            if color_map == "grays":
                color_map = "gray"
            elif color_map == "rainbow":
                color_map = "rainbow"
            elif color_map == "hot":
                color_map = "hot"
            elif color_map == "cool":
                color_map = "cool"

            # Ensure proper display with automatic scaling
            if np.all(cropped_image == 0):
                # Handle all-zero images by setting a small range
                self.ax.imshow(cropped_image, cmap=color_map, extent=extent, aspect='equal', vmin=0, vmax=1)
            else:
                # For real loaded images, use gentle contrast enhancement
                if hasattr(self.im, 'name') and ('Reconstructed' not in self.im.name):
                    # This is a real loaded image - use gentle contrast enhancement
                    print(f"DEBUG: Real image detected: {self.im.name}")
                    print(
                        f"DEBUG: Cropped image stats: min={np.min(cropped_image)}, max={np.max(cropped_image)}, mean={np.mean(cropped_image)}")

                    # Use less aggressive percentile range for better contrast
                    vmin, vmax = np.percentile(cropped_image, [1, 99])
                    if vmax > vmin:
                        # Ensure we don't compress the dynamic range too much
                        if (vmax - vmin) < 0.1 * (np.max(cropped_image) - np.min(cropped_image)):
                            # If percentile range is too narrow, use full range
                            self.ax.imshow(cropped_image, cmap=color_map, extent=extent, aspect='equal')
                        else:
                            self.ax.imshow(cropped_image, cmap=color_map, extent=extent, aspect='equal', vmin=vmin,
                                           vmax=vmax)
                    else:
                        self.ax.imshow(cropped_image, cmap=color_map, extent=extent, aspect='equal')
                else:
                    print(f"DEBUG: Synthetic image detected")
                    self.ax.imshow(cropped_image, cmap=color_map, extent=extent, aspect='equal')

            # Show particle perimeter if enabled
            if self.show_perimeter_var.get():
                # Use contrasting colors for better visibility
                edge_color = 'lime' if color_map == 'gray' else 'red'
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor=edge_color, linewidth=3)
                self.ax.add_patch(circle)

            # Mark particle center with contrasting color
            center_color = 'red' if color_map == 'gray' else 'yellow'
            self.ax.plot(x_coord, y_coord, marker='+', color=center_color,
                         markersize=12, markeredgewidth=3)

            self.ax.set_title(f"Particle {self.current_particle}")
            self.canvas.draw()

            # Update info text
            self.update_info_text()

            # Update button states
            self.prev_btn.config(state=tk.NORMAL if self.current_particle > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_particle < self.total_particles - 1 else tk.DISABLED)

        def update_info_text(self):
            """Update particle information text"""
            self.info_text.delete(1.0, tk.END)

            particle_data = self.info.data[self.current_particle]

            # ViewParticles information display format
            info_text = f"Particle {self.current_particle}\n"
            info_text += f"━━━━━━━━━━━━━━━━━━━━\n"

            # Detection parameters
            info_text += f"P_Seed (X): {particle_data[0]:.6f}\n"
            info_text += f"Q_Seed (Y): {particle_data[1]:.6f}\n"
            info_text += f"Scale: {particle_data[2]:.6f}\n"

            if len(particle_data) > 3:
                info_text += f"Response: {particle_data[3]:.8f}\n"

            # Measurements
            if len(particle_data) > 8:
                info_text += f"\nMeasurements:\n"
                info_text += f"━━━━━━━━━━━━━━━━━━━━\n"

                # Area - use appropriate units (m^2)
                area_val, area_units = format_scientific_notation(particle_data[8], "m^2")
                info_text += f"Area: {area_val} {area_units}\n"

            if len(particle_data) > 9:
                # Volume - often very small, use m^3 units
                vol_val, vol_units = format_scientific_notation(particle_data[9], "m^3")
                info_text += f"Volume: {vol_val} {vol_units}\n"

            if len(particle_data) > 10:
                # Height - usually normal range (nm)
                height_val, height_units = format_scientific_notation(particle_data[10], "nm")
                info_text += f"Height: {height_val} {height_units}\n"

            if len(particle_data) > 11:
                # X_Center - coordinate in meters
                x_val, x_units = format_scientific_notation(particle_data[11], "m")
                info_text += f"X_Center: {x_val} {x_units}\n"

            if len(particle_data) > 12:
                # Y_Center - coordinate in meters
                y_val, y_units = format_scientific_notation(particle_data[12], "m")
                info_text += f"Y_Center: {y_val} {y_units}\n"

            if len(particle_data) > 13:
                # AvgHeight - average intensity
                avg_val, avg_units = format_scientific_notation(particle_data[13], "")
                info_text += f"AvgHeight: {avg_val} {avg_units}\n"

            # Additional Igor Pro style information
            info_text += f"\nParticle #{self.current_particle + 1} of {self.total_particles}"

            self.info_text.insert(tk.END, info_text)

        def prev_particle(self):
            """Navigate to previous particle"""
            if self.current_particle > 0:
                self.current_particle -= 1
                self.update_display()

        def next_particle(self):
            """Navigate to next particle"""
            if self.current_particle < self.total_particles - 1:
                self.current_particle += 1
                self.update_display()

        def toggle_perimeter(self):
            """Toggle perimeter display"""
            self.show_perimeter = self.show_perimeter_var.get()
            self.update_display()

        def change_color_table(self, event=None):
            """Change color table"""
            self.color_table = self.color_var.get()
            self.update_display()

        def delete_particle(self):
            """Delete current particle"""
            if self.total_particles <= 1:
                messagebox.showwarning("Cannot Delete", "Cannot delete the last particle.")
                return

            result = messagebox.askyesno("Delete Particle",
                                         f"Delete particle {self.current_particle}?")
            if result:
                # Remove particle from info array
                mask = np.ones(self.total_particles, dtype=bool)
                mask[self.current_particle] = False
                self.info.data = self.info.data[mask]
                self.total_particles -= 1

                # Adjust current particle index
                if self.current_particle >= self.total_particles:
                    self.current_particle = self.total_particles - 1

                self.update_display()

    # Create and show the viewer
    viewer = ParticleViewer(im, info)
    return viewer


def MeasureParticles(im, info):
    """
    Measure particle properties from image

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information array

    Info array structure:
    Column 0: P_Seed (X coordinate)
    Column 1: Q_Seed (Y coordinate)
    Column 2: Scale (characteristic radius)
    Column 3: Response (blob strength)
    Column 4: unused
    Column 5: unused
    Column 6: unused
    Column 7: unused
    Column 8: Area (physical units)
    Column 9: Volume (integrated intensity)
    Column 10: Height (peak intensity)
    Column 11: X_Center (center of mass X)
    Column 12: Y_Center (center of mass Y)
    Column 13: AvgHeight (average intensity within particle)
    Column 14: unused (for future expansion)

    Returns:
    bool - Success status
    """
    if im is None or info is None:
        return False

    if info.data.shape[0] == 0:
        return True  # No particles to measure

    print(f"MeasureParticles: Measuring {info.data.shape[0]} particles...")

    # Ensure info array has 15 columns
    if info.data.shape[1] < 15:
        # Expand array to hold all measurements
        new_data = np.zeros((info.data.shape[0], 15))
        new_data[:, :info.data.shape[1]] = info.data
        info.data = new_data

    num_particles = info.data.shape[0]

    # Get image dimensions and scaling
    rows = im.data.shape[0]
    cols = im.data.shape[1]

    # Get scaling information
    x_scale = im.GetScale('x')
    y_scale = im.GetScale('y')

    x_delta = x_scale['delta']
    y_delta = y_scale['delta']
    x_offset = x_scale['offset']
    y_offset = y_scale['offset']

    # Physical area per pixel
    pixel_area = x_delta * y_delta

    for i in range(num_particles):
        # Extract particle parameters
        p_seed = info.data[i, 0]  # X coordinate
        q_seed = info.data[i, 1]  # Y coordinate
        scale = info.data[i, 2]  # Characteristic radius
        response = info.data[i, 3]  # Blob strength

        # Initialize measurements to zero
        area_pixels = 0
        volume = 0
        height = 0
        avg_height = 0
        x_center_sum = 0
        y_center_sum = 0
        total_intensity = 0

        # Calculate measurement region
        # Use characteristic radius for measurement region
        measurement_radius = scale

        # Bounds checking
        x_min = max(0, int(p_seed - measurement_radius))
        x_max = min(cols - 1, int(p_seed + measurement_radius))
        y_min = max(0, int(q_seed - measurement_radius))
        y_max = min(rows - 1, int(q_seed + measurement_radius))

        # Loop over measurement region
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                # Calculate distance from particle center
                dx = x - p_seed
                dy = y - q_seed
                distance = np.sqrt(dx * dx + dy * dy)

                # Check if pixel is within measurement radius
                if distance <= measurement_radius:
                    pixel_value = im.data[y, x]

                    # Area measurement
                    area_pixels += 1

                    # Volume measurement
                    volume += pixel_value

                    # Height measurement
                    if pixel_value > height:
                        height = pixel_value

                    # Center of mass calculation
                    x_center_sum += x * pixel_value
                    y_center_sum += y * pixel_value
                    total_intensity += pixel_value

        # Calculate derived measurements
        if area_pixels > 0:
            # Convert area to physical units
            area_physical = area_pixels * pixel_area

            # Calculate average height
            avg_height = volume / area_physical if area_physical > 0 else 0

            # Calculate center of mass coordinates
            if total_intensity > 0:
                x_center = x_center_sum / total_intensity
                y_center = y_center_sum / total_intensity

                # Convert to physical coordinates
                x_center_phys = x_center * x_delta + x_offset
                y_center_phys = y_center * y_delta + y_offset
            else:
                # Fallback to seed position if no intensity
                x_center_phys = p_seed * x_delta + x_offset
                y_center_phys = q_seed * y_delta + y_offset
        else:
            # No pixels found
            area_physical = 0
            avg_height = 0
            x_center_phys = p_seed * x_delta + x_offset
            y_center_phys = q_seed * y_delta + y_offset

        # Store measurements in info array
        info.data[i, 8] = area_physical  # Area
        info.data[i, 9] = volume  # Volume
        info.data[i, 10] = height  # Height
        info.data[i, 11] = x_center_phys  # X_Center
        info.data[i, 12] = y_center_phys  # Y_Center
        info.data[i, 13] = avg_height  # AvgHeight

    print(f"MeasureParticles: Completed measurement of {num_particles} particles")
    return True


def format_scientific_notation(value, units=""):
    """
    Format numbers like ViewParticles

    Displays:
    - Normal values: decimal notation
    - Very small values: coefficient with exponent in units
    - Wave notes: scientific notation
    """
    if abs(value) < 1e-20 or abs(value) > 1e20:
        # Extremely small/large: scientific notation
        return f"{value:.4e}", units
    elif abs(value) < 1e-3:
        # Small values: coefficient and exponent in units
        if value == 0:
            return "0", units
        exp = int(f"{value:.2e}".split('e')[1])
        coeff = value / (10 ** exp)
        if units:
            return f"{coeff:.3f}", f"{units} e{exp:+d}"
        else:
            return f"{coeff:.3f}e{exp:+d}", ""
    else:
        # Normal range: decimal notation
        return f"{value:.4f}", units


def format_export_number(value):
    """
    Format numbers like Igor export format

    Formatting for different number ranges:
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


def ExportResults(results_dict, file_path):
    """
    Export results in Igor Pro format

    Parameters:
    results_dict : dict - Dictionary of analysis results
    file_path : str - Output file path
    """
    if not results_dict:
        raise ValueError("No results to export")

    # Collect all particle data
    all_data = []

    for image_name, result in results_dict.items():
        if 'info' in result and result['info'].data.shape[0] > 0:
            info_data = result['info'].data
            num_particles = info_data.shape[0]

            # Add image name and all particle data columns
            for i in range(num_particles):
                row = [image_name] + list(info_data[i])
                all_data.append(row)

    if not all_data:
        raise ValueError("No particle data to export")

    # Create header
    header = [
        'Image',  # Image name
        'P_Seed',  # X coordinate (column 0)
        'Q_Seed',  # Y coordinate (column 1)
        'Scale',  # Characteristic radius (column 2)
        'Response',  # Blob strength (column 3)
        'Col4',  # Unused (column 4)
        'Col5',  # Unused (column 5)
        'Col6',  # Unused (column 6)
        'Col7',  # Unused (column 7)
        'Area',  # Area in physical units (column 8)
        'Volume',  # Integrated intensity (column 9)
        'Height',  # Peak intensity (column 10)
        'X_Center',  # Center of mass X (column 11)
        'Y_Center',  # Center of mass Y (column 12)
        'AvgHeight'  # Average intensity (column 13)
    ]

    # Add column 14 if present
    if len(all_data[0]) > 15:  # Image name + 15 data columns
        header.append('Col14')

    # Determine output format based on file extension
    if file_path.endswith('.txt'):
        # Igor Pro tab-delimited format
        with open(file_path, 'w') as txtfile:
            # Write header
            txtfile.write('\t'.join(header) + '\n')

            # Write data rows
            for row in all_data:
                # Format numbers with appropriate precision
                formatted_row = [row[0]]  # Image name

                for i, value in enumerate(row[1:], 1):
                    # Use standard number formatting for all values
                    formatted_row.append(format_export_number(float(value)))

                txtfile.write('\t'.join(formatted_row) + '\n')
    else:
        # CSV format for Excel compatibility
        import csv
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            # Format data with appropriate precision
            for row in all_data:
                formatted_row = [row[0]]  # Image name

                for i, value in enumerate(row[1:], 1):
                    # Use standard number formatting for all values
                    formatted_row.append(format_export_number(float(value)))

                writer.writerow(formatted_row)

    print(f"ExportResults: Exported {len(all_data)} particles from {len(results_dict)} images to {file_path}")
    print(f"ExportResults: Format matches Igor Pro HessianBlobs output exactly")


def CreateMeasurementWaves(info_wave):
    """
    Create individual measurement waves from info array

    This function extracts measurement data from the info wave and creates
    individual waves for each measurement type.

    Parameters:
    info_wave : Wave - Particle information array (15 columns)

    Returns:
    dict - Dictionary containing individual measurement waves:
        'Heights' : Wave - Peak intensities (column 10)
        'Areas' : Wave - Physical areas (column 8)
        'Volumes' : Wave - Integrated intensities (column 9)
        'AvgHeights' : Wave - Average intensities (column 13)
        'COM' : Wave - Center of mass coordinates (columns 11,12)
    """
    from utilities import Wave
    import numpy as np

    if info_wave is None or info_wave.data.shape[0] == 0:
        return {}

    num_particles = info_wave.data.shape[0]

    # Extract individual measurement arrays
    heights_data = info_wave.data[:, 10]  # Column 10: Height
    areas_data = info_wave.data[:, 8]  # Column 8: Area
    volumes_data = info_wave.data[:, 9]  # Column 9: Volume
    avgheights_data = info_wave.data[:, 13]  # Column 13: AvgHeight

    # Center of mass coordinates (2D wave)
    com_data = np.column_stack([
        info_wave.data[:, 11],  # Column 11: X_Center
        info_wave.data[:, 12]  # Column 12: Y_Center
    ])

    # Create individual waves
    measurement_waves = {
        'Heights': Wave(heights_data, 'Heights'),
        'Areas': Wave(areas_data, 'Areas'),
        'Volumes': Wave(volumes_data, 'Volumes'),
        'AvgHeights': Wave(avgheights_data, 'AvgHeights'),
        'COM': Wave(com_data, 'COM')
    }

    print(f"CreateMeasurementWaves: Created {len(measurement_waves)} measurement waves from {num_particles} particles")
    return measurement_waves


def CalculateParticleStatistics(info_wave):
    """
    Calculate statistical summary of particle measurements

    Parameters:
    info_wave : Wave - Particle information array

    Returns:
    dict:
        'num_particles' : int - Total number of particles
        'mean_area' : float - Mean area
        'std_area' : float - Standard deviation of area
        'mean_volume' : float - Mean volume
        'std_volume' : float - Standard deviation of volume
        'mean_height' : float - Mean height
        'std_height' : float - Standard deviation of height
        'mean_radius' : float - Mean radius (scale)
        'std_radius' : float - Standard deviation of radius
    """
    import numpy as np

    if info_wave is None or info_wave.data.shape[0] == 0:
        return {
            'num_particles': 0,
            'mean_area': 0, 'std_area': 0,
            'mean_volume': 0, 'std_volume': 0,
            'mean_height': 0, 'std_height': 0,
            'mean_radius': 0, 'std_radius': 0
        }

    num_particles = info_wave.data.shape[0]

    # Extract measurement data
    radii = info_wave.data[:, 2]  # Column 2: Scale (radius)
    areas = info_wave.data[:, 8]  # Column 8: Area
    volumes = info_wave.data[:, 9]  # Column 9: Volume
    heights = info_wave.data[:, 10]  # Column 10: Height

    # Calculate statistics
    stats = {
        'num_particles': num_particles,
        'mean_area': np.mean(areas) if num_particles > 0 else 0,
        'std_area': np.std(areas, ddof=1) if num_particles > 1 else 0,
        'mean_volume': np.mean(volumes) if num_particles > 0 else 0,
        'std_volume': np.std(volumes, ddof=1) if num_particles > 1 else 0,
        'mean_height': np.mean(heights) if num_particles > 0 else 0,
        'std_height': np.std(heights, ddof=1) if num_particles > 1 else 0,
        'mean_radius': np.mean(radii) if num_particles > 0 else 0,
        'std_radius': np.std(radii, ddof=1) if num_particles > 1 else 0
    }

    print(f"CalculateParticleStatistics: Computed statistics for {num_particles} particles")
    print(f"  Mean area: {stats['mean_area']:.4f} ± {stats['std_area']:.4f}")
    print(f"  Mean volume: {stats['mean_volume']:.6f} ± {stats['std_volume']:.6f}")
    print(f"  Mean height: {stats['mean_height']:.6f} ± {stats['std_height']:.6f}")
    print(f"  Mean radius: {stats['mean_radius']:.6f} ± {stats['std_radius']:.6f}")

    return stats


def ValidateParticleMeasurements(info_wave, im_wave=None):
    """
    Validate particle measurements for consistency

    Parameters:
    info_wave : Wave - Particle information array
    im_wave : Wave - Original image (optional, for bounds checking)

    Returns:
    dict - Validation results:
        'valid' : bool - Overall validation status
        'errors' : list - List of validation errors
        'warnings' : list - List of validation warnings
    """
    errors = []
    warnings = []

    if info_wave is None:
        errors.append("Info wave is None")
        return {'valid': False, 'errors': errors, 'warnings': warnings}

    if info_wave.data.shape[0] == 0:
        warnings.append("No particles to validate")
        return {'valid': True, 'errors': errors, 'warnings': warnings}

    num_particles = info_wave.data.shape[0]

    # Check array dimensions
    if info_wave.data.shape[1] < 15:
        warnings.append(f"Info array has {info_wave.data.shape[1]} columns, expected 15")

    # Validate individual particles
    for i in range(num_particles):
        particle_data = info_wave.data[i]

        # Check coordinates
        if particle_data[0] < 0 or particle_data[1] < 0:
            errors.append(f"Particle {i}: Negative coordinates ({particle_data[0]}, {particle_data[1]})")

        # Check scale/radius
        if particle_data[2] <= 0:
            errors.append(f"Particle {i}: Invalid scale/radius {particle_data[2]}")

        # Check measurements (if present)
        if len(particle_data) > 8:
            if particle_data[8] < 0:  # Area
                errors.append(f"Particle {i}: Negative area {particle_data[8]}")
            if particle_data[9] < 0:  # Volume
                errors.append(f"Particle {i}: Negative volume {particle_data[9]}")
            if particle_data[10] < 0:  # Height
                warnings.append(f"Particle {i}: Negative height {particle_data[10]}")

        # Check bounds against image (if provided)
        if im_wave is not None:
            if (particle_data[0] >= im_wave.data.shape[1] or
                    particle_data[1] >= im_wave.data.shape[0]):
                errors.append(f"Particle {i}: Coordinates outside image bounds")

    valid = len(errors) == 0

    print(f"ValidateParticleMeasurements: Validated {num_particles} particles")
    if errors:
        print(f"  Found {len(errors)} errors")
    if warnings:
        print(f"  Found {len(warnings)} warnings")

    return {'valid': valid, 'errors': errors, 'warnings': warnings}

