import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from modules import UNet3d
from dataset import HipMRIDataset

def main():
    data_dir = Path("/home/groups/comp3710/HipMRI_Study_open")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load the test dataset
    test_dataset = HipMRIDataset(data_dir,split='test')

    #create a data loader
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    #initialise the same architecture used during training
    model = UNet3d(in_channels=1, num_classes=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    aux_params = sum(p.numel() for p in model.aux2.parameters())
    aux_params += sum(p.numel() for p in model.aux3.parameters())

    print(f"Intermediate parameters: {total_params - aux_params:,}")
    print(f"Advanced parameters: {total_params:,}")
    print(f"Additional parameters: {aux_params:,}")

    checkpoint = torch.load("best_unet_class5.pth",map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint,strict=False)
    model.eval()

    num_classes = 6
    intersections = torch.zeros(num_classes,dtype=torch.float64)
    pred_counts = torch.zeros(num_classes,dtype=torch.float64)
    true_counts = torch.zeros(num_classes,dtype=torch.float64)

    with (torch.no_grad()):
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs,dim=1)

            if batch_idx == 0:
                mri = images[0,0].cpu().numpy()
                true_mask = masks[0].cpu().numpy()
                pred_mask = predictions[0].cpu().numpy()

                z = (true_mask == 5).sum(axis=(1,0)).argmax()

                mri = mri[:,:,z]
                true_mask = true_mask[:,:,z]
                pred_mask = pred_mask[:,:,z]

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(mri, cmap="gray")
                axes[0].set_title("Original MRI")

                axes[1].imshow(true_mask, cmap="tab10", vmin=0, vmax=5,interpolation="nearest")
                axes[1].set_title("Ground Truth")

                axes[2].imshow(pred_mask,cmap="tab10",vmin=0,vmax=5,interpolation="nearest")
                axes[2].set_title("U-Net Prediction")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig("intermediate_predictions.png",dpi=150)
                plt.close()

            for class_id in range(num_classes):
                pred_mask = (predictions == class_id)
                true_mask = (masks == class_id)

                intersection = torch.sum(pred_mask * true_mask)
                pred_total = torch.sum(pred_mask)
                target = torch.sum(true_mask)

                intersections[class_id] += intersection.item()
                pred_counts[class_id] += pred_total.item()
                true_counts[class_id] += target.item()

    dice = (2 * intersections + 1e-6)/(pred_counts + true_counts + 1e-6)
    union = pred_counts + true_counts - intersections
    iou = (intersections + 1e-6) / (union + 1e-6)

    for class_id, score in enumerate(dice):
        print(f"Class {class_id}: Dice = {score.item():.4f},Iou = {iou[class_id]:.4f}")

    print("Test images:", len(test_dataset))
    print("Test batches:", len(test_loader))

if __name__ == "__main__":
    main()