import argparse
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import sys
import os
import zipfile
import struct
import ast

# Global configuration for target class counts
TARGET_COUNTS = {
    0: 10000,
    1: 10000,
    2: 10000,
    3: 10000,
    4: 10000,
    5: 10000,
    6: 10000,
    7: 10000,
    8: 10000
}

TARGET_SIZE = (64, 64)

def resize_image(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

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
    major_bytes = file_like.read(1)
    minor_bytes = file_like.read(1)
    major = struct.unpack('B', major_bytes)[0]
    # minor = struct.unpack('B', minor_bytes)[0]
    
    # Read header length
    if major == 1:
        header_len_bytes = file_like.read(2)
        header_len = struct.unpack('<H', header_len_bytes)[0]
    elif major == 2:
        header_len_bytes = file_like.read(4)
        header_len = struct.unpack('<I', header_len_bytes)[0]
    else:
        raise ValueError(f"Unsupported .npy version: {major}")

    # Read header
    header_bytes = file_like.read(header_len)
    header_str = header_bytes.decode('ascii')
    
    # Parse header dictionary safely
    header = ast.literal_eval(header_str)
    return header

def main():
    parser = argparse.ArgumentParser(description="Postprocess DistDiff generated samples: Downscale and Subsample.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input .npz file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed .npz file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--block_size", type=int, default=1000, help="Number of images to process at once")
    parser.add_argument("--split", type=str, default="train", help="Split prefix for keys (e.g. 'train', 'val'). Default: 'train'")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # Safety check: Prevent overwriting input
    if input_path.resolve() == output_path.resolve():
        print(f"Error: Output path cannot be the same as input path: {input_path}")
        sys.exit(1)
        
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if output_path.exists():
        print(f"Warning: Output file already exists: {output_path}")
        # Check if running in non-interactive mode or force overwrite flag (not implemented), 
        # but for safety, we'll ask or abort if not interactive.
        # Since this is likely run in a terminal, input() works.
        try:
            response = input("Do you want to overwrite it? (y/n): ")
            if response.lower() != 'y':
                print("Aborting.")
                sys.exit(0)
        except EOFError:
            print("Non-interactive mode detected. Aborting to prevent overwrite.")
            sys.exit(1)

    print(f"Processing file: {input_path}")
    
    # 1. Load labels and metadata first (small files)
    print("Loading metadata...")
    with zipfile.ZipFile(input_path, 'r') as z:
        # Load labels
        with z.open('labels.npy') as f:
            labels = np.load(f)
        
        # Load is_synth if exists
        is_synth = None
        if 'is_synth.npy' in z.namelist():
            with z.open('is_synth.npy') as f:
                is_synth = np.load(f)
                
        # Load class_names if exists
        class_names = None
        if 'class_names.npy' in z.namelist():
            with z.open('class_names.npy') as f:
                class_names = np.load(f, allow_pickle=True)

        # Prepare to read images
        # We need to know the shape and dtype of images without loading data
        with z.open('images.npy') as f:
            header = read_npy_header(f)
            images_shape = header['shape']
            images_dtype = np.dtype(header['descr'])
            is_fortran = header['fortran_order']
            
            if is_fortran:
                print("Warning: Fortran order arrays detected. This script assumes C-order for reshaping.")

    print(f"Original images shape: {images_shape}")
    print(f"Original labels shape: {labels.shape}")
    
    # 2. Index classes and select samples
    print("Indexing classes...")
    # We need to read labels into memory to index them efficiently. 
    # Labels are int64, 112k * 8 bytes ~ 1MB. Safe to load.
    
    class_indices = {}
    unique_classes = np.unique(labels)
    
    for cls in unique_classes:
        class_indices[cls] = np.where(labels == cls)[0]
        print(f"  Class {cls}: {len(class_indices[cls])} available")

    # Select indices
    selected_indices = []
    np.random.seed(args.seed)
    
    print("\nSelecting samples...")
    for cls, target_count in TARGET_COUNTS.items():
        if cls not in class_indices:
            print(f"  Warning: Class {cls} not found in data. Skipping.")
            continue
            
        available = len(class_indices[cls])
        if available < target_count:
            print(f"  Warning: Class {cls} has {available} samples, requested {target_count}. Taking all.")
            count_to_take = available
        else:
            count_to_take = target_count
            
        # Randomly select indices
        indices = np.random.choice(class_indices[cls], count_to_take, replace=False)
        selected_indices.extend(indices)
        print(f"  Class {cls}: Selected {len(indices)} samples")
        
    selected_indices = np.array(selected_indices)
    # Sort indices to optimize disk read access pattern (sequential reads are faster)
    selected_indices.sort()
    
    total_selected = len(selected_indices)
    print(f"\nTotal samples selected: {total_selected}")
    
    # 3. Process images by blocks
    total_images = images_shape[0]
    img_height = images_shape[1]
    img_width = images_shape[2]
    img_channels = images_shape[3]
    
    # Calculate bytes per image
    item_size = img_height * img_width * img_channels * images_dtype.itemsize
    
    # Pre-allocate output arrays
    out_images = np.zeros((total_selected, *TARGET_SIZE, 3), dtype=np.uint8)
    out_labels = labels[selected_indices]
    out_is_synth = is_synth[selected_indices] if is_synth is not None else None
    
    print(f"Processing images in blocks of {args.block_size}...")
    
    with zipfile.ZipFile(input_path, 'r') as z:
        with z.open('images.npy') as f:
            # Skip header again
            read_npy_header(f) # Advances file pointer past header
            
            # We need to map global indices to selected indices
            # Create a map: global_idx -> list of positions in out_images
            # Since selected_indices is sorted, we can just iterate through it.
            
            current_selected_ptr = 0
            
            for start_idx in tqdm(range(0, total_images, args.block_size)):
                end_idx = min(start_idx + args.block_size, total_images)
                current_block_size = end_idx - start_idx
                
                # Check if any selected indices are in this block
                # selected_indices is sorted.
                # Find range in selected_indices that falls into [start_idx, end_idx)
                
                block_indices_in_selected = []
                block_indices_relative = []
                
                while current_selected_ptr < total_selected:
                    sel_idx = selected_indices[current_selected_ptr]
                    if sel_idx < start_idx:
                        # Should not happen if sorted and logic is correct
                        current_selected_ptr += 1
                        continue
                    if sel_idx >= end_idx:
                        break
                    
                    block_indices_in_selected.append(current_selected_ptr)
                    block_indices_relative.append(sel_idx - start_idx)
                    current_selected_ptr += 1
                
                bytes_to_read = current_block_size * item_size
                raw_bytes = f.read(bytes_to_read)
                
                if len(raw_bytes) != bytes_to_read:
                    print(f"Warning: Unexpected end of file at index {start_idx}. Expected {bytes_to_read} bytes, got {len(raw_bytes)}.")
                    break

                if len(block_indices_in_selected) == 0:
                    continue
                
                # Convert raw bytes to numpy array for this block
                block_data = np.frombuffer(raw_bytes, dtype=images_dtype)
                block_data = block_data.reshape((current_block_size, img_height, img_width, img_channels))
                
                # Process needed images
                for i, rel_idx in enumerate(block_indices_relative):
                    out_idx = block_indices_in_selected[i]
                    
                    img = block_data[rel_idx]
                    img_resized = resize_image(img, TARGET_SIZE)
                    out_images[out_idx] = img_resized
            
    # 4. Save
    print(f"\nSaving to {output_path}...")
    
    # Match structure of reference file: keys prefixed with split name (e.g., 'train_images')
    # and no class_names metadata.
    save_dict = {
        f'{args.split}_images': out_images,
        f'{args.split}_labels': out_labels,
    }
    if out_is_synth is not None:
        save_dict[f'{args.split}_is_synth'] = out_is_synth
        
    # Note: class_names is excluded to match the reference file structure
        
    np.savez_compressed(output_path, **save_dict)
    print("Done!")

if __name__ == "__main__":
    main()
