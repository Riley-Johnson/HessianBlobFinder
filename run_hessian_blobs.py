"""
Hessian Blob Particle Detection Suite - Main Runner
This is the main entry point for the Hessian Blob Detection Suite.
"""

import sys
import os
import warnings
import numpy as np
import platform
from pathlib import Path

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("Hessian Blob Particle Detection Suite")
    print("Python Port of Igor Pro Implementation")
    print("=" * 70)
    print()

def print_usage_instructions():
    """Print usage instructions"""
    print("USAGE INSTRUCTIONS:")
    print("=" * 50)
    print("1. Load image(s) using File menu or Load buttons")
    print("2. Run 'Hessian Blob Detection' from Analysis menu")
    print("3. Set parameters in the dialog (scale start, layers, etc.)")
    print("4. Use interactive threshold to select blob strength")
    print("5. View results and detected particles")
    print()
    print("SUPPORTED FILE FORMATS:")
    print("• Igor Binary Wave (.ibw) - requires 'igor' package")
    print("• TIFF (.tif, .tiff) - requires 'tifffile' or 'Pillow'")
    print("• PNG (.png) - requires 'Pillow'")
    print("• JPEG (.jpg, .jpeg) - requires 'Pillow'")
    print()
    print("INSTALL MISSING DEPENDENCIES:")
    print("pip install igor tifffile Pillow scikit-image")
    print("=" * 50)
    print()


def main():
    """Main entry point"""
    print_banner()
    print_usage_instructions()

    try:
        print("Launching Hessian Blob Detection Suite...")
        print()

        # Check if igor package is available
        try:
            import igor.binarywave as bw
            print("Igor package available - IBW files supported")
        except ImportError:
            print("WARNING: Igor package not found - IBW files may not load properly")
            print("   Try: pip install igor")
        print()

        # Import and run the main GUI
        from main_gui import main as gui_main
        gui_main()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to launch application: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check that all required files are present")
        print("2. Verify all dependencies are installed")
        print("3. Try running with --debug for more information")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()