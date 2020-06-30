from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

dataset = ImageFolder("Data")
trainData, testData, trainLabel, testLabel = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=0)

transforms = transforms.Compose(
    [transforms.Resize(200, 200),
     transforms.ToTensor,
     transforms.Normalize([0.5] * 3, [0.5] * 3)
     ]
)

class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def checkChannel(self,dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if Image.open(dataset[index][0].getbands() == {'R', 'G', 'B'}):
                datasetRGB.append(dataset[index])
        return datasetRGB

    def getResizedImage(self, item):
        image = Image.open(self.dataset[item][0])
        _, _, width, height = image.getbbox()
        factor = (0,0,width, width) if width > height else (0,0,height,height)
        return image.crop(factor)

    def __getitem__(self, item):
        image = self.getResizedImage(item)
        if transforms is not None:
            return image, self.dataset[item][1]
        return self.transform(image),self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


imageLoader = ImageLoader(trainData, transforms)

dataLoader = DataLoader(imageLoader, batch_size = 10, shuffle = True)

data = iter(dataLoader)

d = next(data)