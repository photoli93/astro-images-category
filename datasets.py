import numpy as np
from pathlib import Path
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE

# # SKIP IMAGES DURING EPOCHS METHOD
# class SpaceImageDataset(Dataset):
#     """Custom PyTorch Dataset for space images"""
    
#     # Call the initializer/constructor to set up everything the object needs
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = Path(root_dir)
#         self.transform = transform
#         self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
#         # Load all image paths and labels
#         self.samples = []
#         for cls in self.classes:
#             cls_path = self.root_dir / cls
#             for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
#                 for img_path in cls_path.glob(ext):
#                     self.samples.append((img_path, self.class_to_idx[cls]))
    
#     # Number of samples (Needed for batching)
#     def __len__(self):
#         return len(self.samples)
    
#     # Called when PyTorch wants an image
#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
        
#         # Load image and converts BGR to RGB
#         image = cv2.imread(str(img_path))
        
#         if image is None:
#                 print(f"Skipping unreadable image: {img_path}")
#                 return self.__getitem__((idx + 1) % len(self.samples))  # move to next one
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, label
    
# IMPROVEMENT IDEA TO TRY
# DELETE IMAGES BEFORE EPOCHS
# Scan folders and remove all corrupted images (instead of skipped images maybe better?)
class SpaceImageDataset(Dataset):
    """Custom PyTorch Dataset for space images with corrupt file handling."""

    def __init__(self, root_dir, transform=None, log_file="skipped_images.txt"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        self.skipped = []

        # Create or clear the log file
        self.log_path = self.root_dir / log_file
        with open(self.log_path, "w") as f:
            f.write("List of skipped/corrupt images:\n")

        # Scan folders
        for cls in self.classes:
            cls_path = self.root_dir / cls
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in cls_path.glob(ext):
                    image = cv2.imread(str(img_path))
                    if image is None:
                        # Log the bad image
                        self.skipped.append(str(img_path))
                        with open(self.log_path, "a") as f:
                            f.write(f"{img_path}\n")
                        continue
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"Dataset created from {self.root_dir}")
        print(f"   -> {len(self.samples)} valid images")
        if self.skipped:
            print(f"{len(self.skipped)} images skipped (see {self.log_path})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Define transforms
train_transforms = transforms.Compose([
    # ToPILImage() because of cv2.imread() defined earlier (returned a NumPy array but need a PIL Image)
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # Normalize for faster training, better converging
    # Pre-trained models like ResNet, VGG or EfficientNet were trained on ImageNet and these number are
    # the channel-wise mean and std of ImageNet images
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Prevent memorization of specific features
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
])

val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])