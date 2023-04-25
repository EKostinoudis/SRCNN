from torch.utils.data import Dataset
import torchvision.transforms as T
from os import listdir
from os.path import join
from PIL import Image

SIZE = 1000

class ImageDataset(Dataset):
    def __init__(self, path, factor):
        super().__init__()
        self.path = path
        self.files = [name for name in listdir(path) if name.endswith('.png')]
        self.transform = T.Compose([T.CenterCrop(SIZE), T.Resize(SIZE//factor), T.ToTensor()])
        self.transform_target = T.Compose([T.CenterCrop(SIZE), T.ToTensor()])
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(join(self.path, self.files[index])).convert('RGB')
        target = self.transform_target(image)
        return self.transform(image), target
