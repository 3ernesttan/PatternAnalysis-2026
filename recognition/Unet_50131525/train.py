import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet3d
from dataset import HipMRIDataset
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm

def dice_score(prediction,masks,class_id):
    epsilon = 1e-6
    pred_mask = (prediction == class_id)
    true_mask = (masks == class_id)

    #calculate the intersection,prediction and target
    intersection = torch.sum(pred_mask*true_mask)
    pred_total = torch.sum(pred_mask)
    target = torch.sum(true_mask)

    dice = ((2.0 * intersection + epsilon) / (pred_total+target+epsilon))
    return dice

def main():
    data_dir = Path("/home/groups/comp3710/HipMRI_Study_open")

    #initialize datasets
    train_dataset = HipMRIDataset(data_dir, split="train")
    val_dataset = HipMRIDataset(data_dir, split="validate")

    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
    print(f"Train loader batches:{len(train_loader)}\n"
          f"Data loader batches:{len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Check SLURM GPU allocation.")

    model = UNet3d(in_channels=1,num_classes=6).to(device)

    #define loss function
    criterion = nn.CrossEntropyLoss()

    #define optimizer
    optimizer = optim.AdamW(model.parameters(),lr = 0.0001)

    #Setting the number of epochs
    epochs = 5
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss =0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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

        model.eval()
        val_loss = 0.0

        intersections = [0.0]*6
        pred_totals = [0.0]*6
        target_totals =[0.0]*6

        with torch.no_grad():
            for images,masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs,masks)

                val_loss += loss.item()

                #convert logits into predicted class labels
                preds = torch.argmax(outputs,dim=1)

                for class_id in range(6):
                    pred_mask = (preds == class_id)
                    true_mask = (masks == class_id)

                    intersection = torch.sum(pred_mask * true_mask)
                    pred_total = torch.sum(pred_mask)
                    target = torch.sum(true_mask)

                    intersections[class_id] += intersection.item()
                    pred_totals[class_id] += pred_total.item()
                    target_totals[class_id] += target.item()

            for class_id in range(6):
                denominator = pred_totals[class_id] + target_totals[class_id]

                if denominator > 0:
                    dice = 2*intersections[class_id]/denominator
                    print(f"Class {class_id}: Dice = {dice:.4f}")
                else:
                    print(f"Class {class_id}: Dice = N/A (absent)")

        avg_val_loss = val_loss/len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")

        #Saving best model
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),"best_UNet3d.pth")

if __name__ == "__main__":
    main()