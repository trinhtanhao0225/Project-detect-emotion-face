import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, root, status, transform=None):
        self.root = root
        self.status = status
        self.transform = transform
        self.categories= ['angry','disgust','fear','happy','neutral','sad','surprise']
        self.all_images = []
        self.all_labels = []

        data_dir = os.path.join(root, status)
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.all_images.append(os.path.join(class_path, img_name))
                    self.all_labels.append(class_name)  

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, item):
        image_path = self.all_images[item]
        label =self.categories.index(self.all_labels[item])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Test
if __name__ == "__main__":
    transform = Compose([
        Resize((64, 64)),
        ToTensor()
    ])
    dataset = EmotionDataset(root=r'C:\Users\Public\Documents\Project_detect_emotion\ClassificationEmotion', status='train', transform=transform)
    image, label = dataset[100]
    print(image.shape, label)
