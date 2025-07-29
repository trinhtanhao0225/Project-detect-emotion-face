import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from torchvision.transforms.v2 import RandomAffine, ColorJitter
from tqdm import tqdm

from genderDataset import GenderDataset  


def get_args():
    parser = argparse.ArgumentParser(description="CarColors Training Script")
    parser.add_argument("--data_path", "-d", default=r"C:\Users\Public\Documents\Project_detect_emotion\Gender", help="Data path")
    parser.add_argument("--num_epochs", "-n", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--log_folder", "-f", type=str, default="tensorboard_gender", help="TensorBoard log folder")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models_gender", help="Checkpoint folder")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Resume training from checkpoint")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    train_transform = Compose([
        RandomAffine(degrees=(-5, 5), translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & Dataloaders
    train_dataset = GenderDataset(args.data_path, "Training", transform=train_transform)
    val_dataset = GenderDataset(args.data_path, "Validation", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, len(train_dataset.categories))
    model.to(device)

    # Optimizer & Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    # Resume if available
    start_epoch = 0
    best_acc = 0.0

    if args.saved_checkpoint and os.path.exists(args.saved_checkpoint):
        checkpoint = torch.load(args.saved_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        print(f"Resumed training from epoch {start_epoch} with best acc {best_acc:.4f}")

    # Create folders if needed
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.checkpoint_folder, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_folder)

    for epoch in range(start_epoch, args.num_epochs):
        # ---------- Training ----------
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{args.num_epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=np.mean(train_losses))

        avg_train_loss = np.mean(train_losses)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # ---------- Validation ----------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        print(f"[Val] Epoch {epoch+1}: Accuracy = {val_acc:.4f}")

        # ---------- Save Checkpoints ----------
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            print(f"✨ New best accuracy: {best_acc:.4f} — saving best model")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_folder, "last.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.checkpoint_folder, "best.pt"))


if __name__ == "__main__":
    args = get_args()
    train(args)
