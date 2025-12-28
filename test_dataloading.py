from utils.data_loader import VolumeDataset3D
   
dataset = VolumeDataset3D(
    data_dir='data/processed/volumes_ssl/',
    patch_size=(5, 128, 128),
    num_patches_per_epoch=10
)
   
# Load a patch
patch, label = dataset[0]
print(f"Patch shape: {patch.shape}, Label: {label}")



# from utils.data_loader import SELMA3DDataset
# ds = SELMA3DDataset('data/processed/volume_demo/train', patch_size=(8,256,256), samples_per_volume=2)
# img, mask, meta = ds[0]
# print(img.shape, mask.shape, meta.get('global_threshold_norm'))
