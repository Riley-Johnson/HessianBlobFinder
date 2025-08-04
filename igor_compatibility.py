"""
Data Structure Compatibility Module
Provides Wave data structures and mathematical functions for scientific computing
"""

import numpy as np
import warnings

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


class Wave:
    """
    Igor Pro Wave equivalent
    Maintains data, name, scaling, and notes like Igor Pro waves
    """

    def __init__(self, data, name="", note=""):
        self.data = np.asarray(data)
        self.name = name
        self.note = note

        # Initialize scaling for all dimensions
        self._scaling = {
            'x': {'offset': 0.0, 'delta': 1.0, 'units': ''},
            'y': {'offset': 0.0, 'delta': 1.0, 'units': ''},
            'z': {'offset': 0.0, 'delta': 1.0, 'units': ''},
            't': {'offset': 0.0, 'delta': 1.0, 'units': ''}
        }

    def SetScale(self, axis, offset, delta, units=""):
        """Set scaling for specified axis"""
        if axis.lower() in self._scaling:
            self._scaling[axis.lower()]['offset'] = offset
            self._scaling[axis.lower()]['delta'] = delta
            self._scaling[axis.lower()]['units'] = units

    def GetScale(self, axis):
        """Get scaling information for specified axis"""
        return self._scaling.get(axis.lower(), {'offset': 0.0, 'delta': 1.0, 'units': ''})

    def __repr__(self):
        return f"Wave('{self.name}', shape={self.data.shape}, dtype={self.data.dtype})"


def DimSize(wave, dimension):
    """
    Get size of specified dimension

    Parameters:
    wave : Wave - Input wave
    dimension : int - Dimension index (0=rows/y, 1=cols/x, 2=layers/z, 3=chunks/t)

    Returns:
    int - Size of dimension
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    if dimension < 0 or dimension >= len(wave.data.shape):
        return 0

    return wave.data.shape[dimension]


def DimOffset(wave, dimension):
    """
    Get offset of specified dimension

    Parameters:
    wave : Wave - Input wave
    dimension : int - Dimension index

    Returns:
    float - Offset value
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    axis_map = {0: 'y', 1: 'x', 2: 'z', 3: 't'}
    axis = axis_map.get(dimension, 'x')

    return wave.GetScale(axis)['offset']


def DimDelta(wave, dimension):
    """
    Get delta (spacing) of specified dimension

    Parameters:
    wave : Wave - Input wave
    dimension : int - Dimension index

    Returns:
    float - Delta value
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    axis_map = {0: 'y', 1: 'x', 2: 'z', 3: 't'}
    axis = axis_map.get(dimension, 'x')

    return wave.GetScale(axis)['delta']


def ScaleToIndex(wave, scale_value, dimension):
    """
    Convert scale value to index

    Parameters:
    wave : Wave - Input wave
    scale_value : float - Scale coordinate value
    dimension : int - Dimension index

    Returns:
    int - Corresponding index
    """
    offset = DimOffset(wave, dimension)
    delta = DimDelta(wave, dimension)

    if delta == 0:
        return 0

    index = int(round((scale_value - offset) / delta))

    # Clamp to valid range
    max_index = DimSize(wave, dimension) - 1
    return max(0, min(index, max_index))


def IndexToScale(wave, index, dimension):
    """
    Convert index to scale value

    Parameters:
    wave : Wave - Input wave
    index : int - Index value
    dimension : int - Dimension index

    Returns:
    float - Corresponding scale coordinate
    """
    offset = DimOffset(wave, dimension)
    delta = DimDelta(wave, dimension)

    return offset + index * delta


def Duplicate(source_wave, new_name):
    """
    Create a duplicate of a wave

    Parameters:
    source_wave : Wave - Source wave to duplicate
    new_name : str - Name for the new wave

    Returns:
    Wave - Duplicated wave
    """
    if not isinstance(source_wave, Wave):
        raise TypeError("Input must be a Wave object")

    # Create new wave with copied data
    new_data = source_wave.data.copy()
    new_wave = Wave(new_data, new_name, source_wave.note)

    # Copy scaling information
    for axis in ['x', 'y', 'z', 't']:
        scale_info = source_wave.GetScale(axis)
        new_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return new_wave


def Make(dimensions, wave_type="/O", name="temp"):
    """
    Create a new wave

    Parameters:
    dimensions : tuple or int - Dimensions of the wave
    wave_type : str - Wave type specification
    name : str - Wave name

    Returns:
    Wave - New wave filled with zeros
    """
    if isinstance(dimensions, int):
        dimensions = (dimensions,)
    elif isinstance(dimensions, (list, tuple)):
        dimensions = tuple(dimensions)
    else:
        raise TypeError("Dimensions must be int, list, or tuple")

    # Create array filled with zeros
    data = np.zeros(dimensions)

    return Wave(data, name)


def Concatenate(wave_list, axis=0, new_name="concatenated"):
    """
    Concatenate waves along specified axis

    Parameters:
    wave_list : list - List of Wave objects to concatenate
    axis : int - Axis along which to concatenate
    new_name : str - Name for result wave

    Returns:
    Wave - Concatenated wave
    """
    if not wave_list:
        raise ValueError("Wave list cannot be empty")

    # Extract data arrays
    data_arrays = [wave.data for wave in wave_list if isinstance(wave, Wave)]

    if not data_arrays:
        raise ValueError("No valid Wave objects in list")

    # Concatenate data
    concatenated_data = np.concatenate(data_arrays, axis=axis)

    # Create result wave
    result_wave = Wave(concatenated_data, new_name)

    # Copy scaling from first wave
    first_wave = wave_list[0]
    for dim_axis in ['x', 'y', 'z', 't']:
        scale_info = first_wave.GetScale(dim_axis)
        result_wave.SetScale(dim_axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result_wave


def MatrixOp(operation, *args, **kwargs):
    """
    Matrix operations

    Parameters:
    operation : str - Operation to perform
    args : various - Arguments for operation
    kwargs : various - Additional options

    Returns:
    Wave - Result of operation
    """
    # This is a simplified implementation
    # Full Igor Pro MatrixOp has many more operations

    if operation.lower() == "convolve":
        # Simple convolution
        if len(args) >= 2:
            wave1, kernel = args[0], args[1]
            if isinstance(wave1, Wave) and isinstance(kernel, Wave):
                from scipy import ndimage
                result_data = ndimage.convolve(wave1.data, kernel.data, mode='constant')
                return Wave(result_data, f"{wave1.name}_conv")

    elif operation.lower() == "fp32":
        # Convert to 32-bit float
        if len(args) >= 1 and isinstance(args[0], Wave):
            result_data = args[0].data.astype(np.float32)
            return Wave(result_data, f"{args[0].name}_fp32")

    # Default: return first argument if it's a wave
    if args and isinstance(args[0], Wave):
        return args[0]

    raise NotImplementedError(f"MatrixOp operation '{operation}' not implemented")


def FFT(wave, destination=None):
    """
    Fast Fourier Transform

    Parameters:
    wave : Wave - Input wave
    destination : Wave - Output wave

    Returns:
    Wave - FFT result
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    # Compute FFT
    fft_data = np.fft.fft2(wave.data).astype(np.complex128)

    # Create result wave
    if destination is None:
        result_wave = Wave(fft_data, f"{wave.name}_FFT")
    else:
        destination.data = fft_data
        result_wave = destination

    # Copy scaling
    for axis in ['x', 'y', 'z', 't']:
        scale_info = wave.GetScale(axis)
        result_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result_wave


def IFFT(wave, destination=None):
    """
    Inverse Fast Fourier Transform

    Parameters:
    wave : Wave - Input wave
    destination : Wave - Output wave

    Returns:
    Wave - IFFT result
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    # Compute IFFT and take real part
    ifft_data = np.fft.ifft2(wave.data).real

    # Create result wave
    if destination is None:
        result_wave = Wave(ifft_data, f"{wave.name}_IFFT")
    else:
        destination.data = ifft_data
        result_wave = destination

    # Copy scaling
    for axis in ['x', 'y', 'z', 't']:
        scale_info = wave.GetScale(axis)
        result_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result_wave


def KillWaves(*wave_names):
    """
    Delete waves
    In Python, this is mostly for Igor Pro compatibility

    Parameters:
    wave_names : str - Names of waves to delete
    """
    # This function exists for Igor Pro compatibility but obviously doesn't do anything
    pass


def SetScale(wave, axis, offset, delta, units=""):
    """
    Set scaling for wave dimension

    Parameters:
    wave : Wave - Target wave
    axis : str - Axis identifier ('x', 'y', 'z', 't')
    offset : float - Offset value
    delta : float - Delta (spacing) value
    units : str - Units string
    """
    if isinstance(wave, Wave):
        wave.SetScale(axis, offset, delta, units)


def Note(wave, note_text=""):
    """
    Set or get wave note

    Parameters:
    wave : Wave - Target wave
    note_text : str - Note text to set

    Returns:
    str - Current note
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    if note_text:
        wave.note = note_text
    else:
        return wave.note


def WaveExists(wave_name):
    """
    Check if wave exists
    Always returns True in Python since we pass Wave objects directly

    Parameters:
    wave_name : str - Name of wave to check

    Returns:
    bool - True if wave exists
    """
    return True


def Redimension(wave, *new_dimensions):
    """
    Change dimensions of wave

    Parameters:
    wave : Wave - Wave to redimension
    new_dimensions : int - New dimension sizes
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    if new_dimensions:
        # Reshape the data
        total_elements = np.prod(wave.data.shape)
        new_total = np.prod(new_dimensions)

        if new_total == total_elements:
            wave.data = wave.data.reshape(new_dimensions)
        else:
            # Create new array with different size
            new_data = np.zeros(new_dimensions, dtype=wave.data.dtype)
            # Copy as much data as possible
            flat_old = wave.data.flatten()
            flat_new = new_data.flatten()
            copy_size = min(len(flat_old), len(flat_new))
            flat_new[:copy_size] = flat_old[:copy_size]
            wave.data = new_data


def DeletePoints(wave, start_point, num_points, dimension=0):
    """
    Delete points from wave

    Parameters:
    wave : Wave - Wave to modify
    start_point : int - Starting point index
    num_points : int - Number of points to delete
    dimension : int - Dimension to delete from
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    # Use numpy delete
    wave.data = np.delete(wave.data,
                          range(start_point, start_point + num_points),
                          axis=dimension)


def InsertPoints(wave, start_point, num_points, dimension=0):
    """
    Insert points into wave

    Parameters:
    wave : Wave - Wave to modify
    start_point : int - Starting point index
    num_points : int - Number of points to insert
    dimension : int - Dimension to insert into
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    # Create zeros to insert
    insert_shape = list(wave.data.shape)
    insert_shape[dimension] = num_points
    zeros_to_insert = np.zeros(insert_shape, dtype=wave.data.dtype)

    # Insert the zeros
    wave.data = np.insert(wave.data, start_point, zeros_to_insert, axis=dimension)


def FindValue(wave, value, start_point=0):
    """
    Find value in wave

    Parameters:
    wave : Wave - Wave to search
    value : float - Value to find
    start_point : int - Starting point for search

    Returns:
    int - Index of found value (-1 if not found)
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    # Flatten array for searching
    flat_data = wave.data.flatten()

    # Search from start_point
    for i in range(start_point, len(flat_data)):
        if np.isclose(flat_data[i], value):
            return i

    return -1  # Not found


def WaveMax(wave):
    """
    Find maximum value in wave

    Parameters:
    wave : Wave - Input wave

    Returns:
    float - Maximum value
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    return np.max(wave.data)


def WaveMin(wave):
    """
    Find minimum value in wave

    Parameters:
    wave : Wave - Input wave

    Returns:
    float - Minimum value
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    return np.min(wave.data)


def Sum(wave):
    """
    Sum all elements in wave

    Parameters:
    wave : Wave - Input wave

    Returns:
    float - Sum of all elements
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    return np.sum(wave.data)


def Mean(wave):
    """
    Calculate mean of wave

    Parameters:
    wave : Wave - Input wave

    Returns:
    float - Mean value
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    return np.mean(wave.data)


def StdDev(wave):
    """
    Calculate standard deviation of wave

    Parameters:
    wave : Wave - Input wave

    Returns:
    float - Standard deviation
    """
    if not isinstance(wave, Wave):
        raise TypeError("Input must be a Wave object")

    return np.std(wave.data)


def Cmplx(real_part, imag_part):
    """
    Create complex number

    Parameters:
    real_part : float - Real part
    imag_part : float - Imaginary part

    Returns:
    complex - Complex number
    """
    return complex(real_part, imag_part)


def Real(complex_value):
    """
    Extract real part

    Parameters:
    complex_value : complex - Complex number

    Returns:
    float - Real part
    """
    return complex_value.real if isinstance(complex_value, complex) else float(complex_value)


def Imag(complex_value):
    """
    Extract imaginary part

    Parameters:
    complex_value : complex - Complex number

    Returns:
    float - Imaginary part
    """
    return complex_value.imag if isinstance(complex_value, complex) else 0.0


def Num2Str(number):
    """
    Convert number to string

    Parameters:
    number : float/int - Number to convert

    Returns:
    str - String representation
    """
    return str(number)


def Str2Num(string):
    """
    Convert string to number

    Parameters:
    string : str - String to convert

    Returns:
    float - Numeric value (NaN if conversion fails)
    """
    try:
        return float(string)
    except (ValueError, TypeError):
        return np.nan


def NameOfWave(wave):
    """
    Get name of wave

    Parameters:
    wave : Wave - Input wave

    Returns:
    str - Wave name
    """
    if isinstance(wave, Wave):
        return wave.name
    else:
        return ""


def Limit(value, min_val, max_val):
    """
    Limit value to range

    Parameters:
    value : float - Input value
    min_val : float - Minimum allowed value
    max_val : float - Maximum allowed value

    Returns:
    float - Limited value
    """
    return max(min_val, min(value, max_val))


def SelectNumber(condition, true_value, false_value):
    """
    Select number based on condition

    Parameters:
    condition : bool - Condition to test
    true_value : float - Value if condition is true
    false_value : float - Value if condition is false

    Returns:
    float - Selected value
    """
    return true_value if condition else false_value


def Round(value):
    """
    Round to nearest integer

    Parameters:
    value : float - Input value

    Returns:
    int - Rounded value
    """
    return int(np.round(value))


def Floor(value):
    """
    Floor function

    Parameters:
    value : float - Input value

    Returns:
    int - Floor value
    """
    return int(np.floor(value))


def Ceil(value):
    """
    Ceiling function

    Parameters:
    value : float - Input value

    Returns:
    int - Ceiling value
    """
    return int(np.ceil(value))


def Abs(value):
    """
    Absolute value

    Parameters:
    value : float - Input value

    Returns:
    float - Absolute value
    """
    return abs(value)


def Sign(value):
    """
    Sign function

    Parameters:
    value : float - Input value

    Returns:
    int - Sign (-1, 0, or 1)
    """
    return np.sign(value)