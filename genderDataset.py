import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset

class GenderDataset(Dataset):
    def __init__(self,root,status,transform=None):
        self.root=root
        self.status=status
        self.transform=transform
        self.categories= ['female','male']
        self.all_images=[]
        self.all_labels=[]
        data_files=os.path.join(root,status)

        for class_name in os.listdir(data_files):
            class_path=os.path.join(data_files,class_name)
            for image in os.listdir(class_path):
                image_path=os.path.join(class_path,image)
                self.all_images.append(image_path)
                self.all_labels.append(class_name)
        
    def __len__(self):
        return len(self.all_labels)
    def __getitem__(self,item):
        image_path = self.all_images[item]
        label =self.categories.index(self.all_labels[item])

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image,label
if __name__=='__main__':
    transform=Compose([
        ToTensor()
    ])
    dataset=GenderDataset(r'C:\Users\Public\Documents\Project_detect_emotion\Gender','Training',transform=transform)
    image,label=dataset[150]
    print(image,label)