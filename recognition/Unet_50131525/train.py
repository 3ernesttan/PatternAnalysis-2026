import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet2D
from dataset import HipMRIDataset
from pathlib import Path
import torch.nn as nn

def main():
    data_dir = Path("/home/ernest_tan/HipMRI_data")

    #training paths
    train_img_dir = data_dir/"keras_slices_train"
    train_mask_dir = data_dir/"keras_slices_seg_train"

    #validation paths
    val_img_dir = data_dir/"keras_slices_validate"
    val_mask_dir = data_dir/"keras_slices_seg_validate"

    #initialize datasets
    train_dataset = HipMRIDataset(train_img_dir, train_mask_dir)
    val_dataset = HipMRIDataset(val_img_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=4,shuffle=False)
    print(f"Train loader batches:{len(train_loader)}\n"
          f"Data loader batches:{len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet2D(in_channels=1,num_classes=4).to(device)

    #define loss function
    criterion = nn.CrossEntropyLoss()

    #define optimizer
    optimizer = optim.AdamW(model.parameters(),lr = 0.0001)

    #Setting the number of epochs
    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss =0.0
        for images,masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")


