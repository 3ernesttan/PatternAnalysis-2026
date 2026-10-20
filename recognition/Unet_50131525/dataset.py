from statistics import mean

import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
from torch.masked import std
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataset import _T_co

data_path = "/home/ernest_tan/HipMRI_data"

class HipMRIDataset(Dataset):
    def __init__(self,data_path,split="train"):
        self.image_dir = os.path.join(data_path, f"keras_slices_{split}")
        self.mask_dir = os.path.join(data_path,f"keras_slices_seg_{split}")

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #get the MRI files
        image_path = self.image_files[idx]

        #find its segmentation mask
        filename = os.path.basename(image_path)
        mask_filename = filename.replace("case_","seg_", 1)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        #load MRI and mask using nibabel
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        #normalize MRI intensity using zscore
        image = (image - image.mean()) / (image.std() + 1e-8)

        #Conversion of image into pytorch
        image =torch.tensor(image,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.long)

        #Add dimension to the image
        image = image.unsqueeze(0)

        return image,mask

dataset = HipMRIDataset(data_path, split="train")

image, mask = dataset[0]

print("Dataset size:", len(dataset))
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask labels:", torch.unique(mask))

loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, masks = next(iter(loader))

print("Batch images:", images.shape)
print("Batch masks:", masks.shape)

