import os
import sys
import numpy as np
import zipfile
import ast
import struct
from pathlib import Path

def read_npy_header(file_like):
    """
    Reads the header of a .npy file from a file-like object.
    Returns a dictionary containing 'shape', 'fortran_order', 'descr'.
    """
    # Read magic string
    magic = file_like.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError("Invalid .npy file: missing magic string")

    # Read version
    major, minor = file_like.read(2)
    
    # Read header length
    if major == 1:
        header_len_bytes = file_like.read(2)
        header_len = struct.unpack('<H', header_len_bytes)[0]
    elif major == 2:
        header_len_bytes = file_like.read(4)
        header_len = struct.unpack('<I', header_len_bytes)[0]
    else:
        raise ValueError(f"Unsupported .npy version: {major}.{minor}")

    # Read header
    header_bytes = file_like.read(header_len)
    header_str = header_bytes.decode('ascii')
    
    # Parse header dictionary safely
    header = ast.literal_eval(header_str)
    return header

def analyze_directory(base_path):
    base_path = Path(base_path)
    print(f"Analyzing directory: {base_path}")
    
    # 1. Analyze PNGs
    pngs_dir = base_path / "pngs"
    if pngs_dir.exists() and pngs_dir.is_dir():
        png_count = sum(1 for _ in pngs_dir.rglob("*.png"))
        print(f"\n[PNGs Directory]")
        print(f"  Path: {pngs_dir}")
        print(f"  Total PNG files: {png_count}")
        
        # Optional: Count per class if organized in subfolders
        subdirs = [d for d in pngs_dir.iterdir() if d.is_dir()]
        if subdirs:
            print("  Subdirectories (potential classes):")
            for d in sorted(subdirs):
                count = sum(1 for _ in d.glob("*.png"))
                print(f"    {d.name}: {count} images")
    else:
        print(f"\n[PNGs Directory] Not found or not a directory: {pngs_dir}")

    # 2. Analyze .npz file
    npz_path = base_path / "split_0.npz"
    if npz_path.exists():
        print(f"\n[NPZ File Analysis]")
        print(f"  Path: {npz_path}")
        file_size_gb = npz_path.stat().st_size / (1024**3)
        print(f"  File Size: {file_size_gb:.2f} GB")
        
        try:
            with zipfile.ZipFile(npz_path, 'r') as z:
                print("\n  Contents:")
                print(f"  {'Name':<20} | {'Shape':<20} | {'Dtype':<10} | {'Size (MB)':<10}")
                print("-" * 70)
                
                for filename in z.namelist():
                    if not filename.endswith('.npy'):
                        print(f"  {filename:<20} | {'[Not .npy]':<20} | {'-':<10} | {z.getinfo(filename).file_size / 1024**2:.2f}")
                        continue
                        
                    with z.open(filename) as f:
                        try:
                            header = read_npy_header(f)
                            shape = str(header['shape'])
                            dtype = header['descr']
                            size_mb = z.getinfo(filename).file_size / (1024**2)
                            print(f"  {filename:<20} | {shape:<20} | {dtype:<10} | {size_mb:.2f}")
                            
                            # If small enough, maybe load it to show some stats?
                            # For now, just showing metadata is safer for 53GB file.
                            
                        except Exception as e:
                            print(f"  {filename:<20} | Error reading header: {e}")

                # Try to load small arrays to print content
                print("\n  Small arrays content:")
                data = np.load(npz_path, allow_pickle=True) # This is lazy
                for key in data.files:
                    # Heuristic: if file size is small (< 100MB), print some info
                    # We can check size from zip info
                    info = z.getinfo(key + ".npy")
                    if info.file_size < 100 * 1024 * 1024: # 100 MB
                        try:
                            arr = data[key]
                            print(f"    {key}: {arr}")
                            if hasattr(arr, 'min') and not arr.dtype == object:
                                print(f"      Min: {arr.min()}, Max: {arr.max()}")
                            
                            if key == 'labels':
                                unique, counts = np.unique(arr, return_counts=True)
                                print("      Class counts:")
                                for u, c in zip(unique, counts):
                                    print(f"        Class {u}: {c}")
                                    
                        except Exception as e:
                            print(f"    Could not load {key}: {e}")

        except Exception as e:
            print(f"Error analyzing .npz file: {e}")
    else:
        print(f"\n[NPZ File] Not found: {npz_path}")

if __name__ == "__main__":
    target_dir = "/media/mpascual/PortableSSD/medsyn/synthetic_samples/DistDiff"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    analyze_directory(target_dir)
