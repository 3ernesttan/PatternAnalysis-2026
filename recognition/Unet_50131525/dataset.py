import nibabel as nib
import os,glob,torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

class HipMRIDataset(Dataset):
    def __init__(self,data_path,split="train"):
        self.image_dir = os.path.join(data_path, "semantic_MRs")
        self.mask_dir = os.path.join(data_path,"semantic_labels_only")

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*_LFOV.nii.gz")))



    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #get the MRI files
        image_path = self.image_files[idx]

        #find its segmentation mask
        filename = os.path.basename(image_path)
        mask_filename = filename.replace("_LFOV","_SEMANTIC")
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
        image = (F.interpolate(image.unsqueeze(0).unsqueeze(0),
                              size = (128,128,64),
                              mode = "trilinear",
                              align_corners=False
                               ).squeeze(0))

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(128, 128,64),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        return image,mask


if __name__ == "__main__":
    data_path = "/home/groups/comp3710/HipMRI_Study_open"

    dataset = HipMRIDataset(data_path)

    image, mask = dataset[0]

    print("Dataset size:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask labels:", torch.unique(mask))