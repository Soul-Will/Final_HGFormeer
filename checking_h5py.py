import h5py
from pathlib import Path
   
   # Check a file
hdf5_files = list(Path('data/processed/volumes_ssl/').glob('**/*.h5'))
print(f"Found {len(hdf5_files)} HDF5 files")
   
   # Inspect one
with h5py.File(hdf5_files[0], 'r') as f:
    print(f"Keys: {list(f.keys())}")
    print(f"Shape: {f['volume'].shape}")
    print(f"Compression: {f['volume'].compression}")
    print(f"Size on disk: {hdf5_files[0].stat().st_size / 1e6:.1f} MB")