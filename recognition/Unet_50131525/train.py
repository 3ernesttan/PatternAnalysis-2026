import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet2D
from dataset import HipMRIDataset
from pathlib import Path
import torch.nn as nn

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
    data_dir = Path("/home/ernest_tan/HipMRI_data")

    #training paths
    train_img_dir = data_dir/"keras_slices_train"
    train_mask_dir = data_dir/"keras_slices_seg_train"

    #validation paths
    val_img_dir = data_dir/"keras_slices_validate"
    val_mask_dir = data_dir/"keras_slices_seg_validate"

    #initialize datasets
    train_dataset = HipMRIDataset(data_dir, split="train")
    val_dataset = HipMRIDataset(data_dir, split="validate")

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
    epochs = 1
    best_val_loss = float("inf")
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

        model.eval()
        val_loss = 0.0
        dice_totals = [0.0]*4
        with torch.no_grad():
            for images,masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs,masks)

                val_loss += loss.item()

                #convert logits into predicted class labels
                preds = torch.argmax(outputs,dim=1)

                background_dice = dice_score(preds,masks,0)
                dice1 = dice_score(preds,masks,1)
                dice2 = dice_score(preds,masks,2)
                dice3 = dice_score(preds,masks,3)

                #accumulate dice scores
                dice_totals[0] += background_dice.item()
                dice_totals[1] += dice1.item()
                dice_totals[2] += dice2.item()
                dice_totals[3] += dice3.item()

            avg_dice = [total / len(val_loader) for total in dice_totals]
            for class_id, score in enumerate(avg_dice):
                print(f"Class {class_id}: Dice = {score:.4f}")

        avg_val_loss = val_loss/len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")

        #Saving best model
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),"best_unet2d.pth")

if __name__ == "__main__":
    main()