import torch
import torchvision.transforms.v2 as transforms
from PIL import Image as Img
from PIL.Image import Image as PILImage
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def transform_image(transform , image):
    out = transform(image)
    out_np = out.numpy().transpose(1, 2, 0)
    return out_np

def plot_image(plt, img, t_img):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(t_img)
    plt.show()
    
def to_rgb(image: PILImage) -> PILImage:
    if image.mode != "RGB":
        return image.convert("RGB")
    
    return image

def save_checkpoint(state, filename):
    torch.save(state, filename)

def calculate_accuracy(correct, labels):
    return  100 * (correct / labels)

# validate an epoch
def validate_one_epoch(model, loss_func, data_loader, device):
    running_vloss = 0
    avg_vloss = 0
    model.eval()
    with torch.no_grad():
        for i , (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_func(pred, y)
            running_vloss += loss.item()
        
    avg_vloss = running_vloss / len(data_loader)
    
    return avg_vloss


class HGDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        
        if self.transform:
            image = self.transform(image)

        return image, item['label']