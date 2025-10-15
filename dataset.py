from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from continuous_dilation import RandomGrayscaleDilation


class MNISTNormalizedThicknessDataset(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        thickness = img.sum()
        
        normalize_transform = transforms.Normalize([0.5],[0.5])
        img = normalize_transform(img)
        
        return img, thickness


def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4, dilation=True, seed=12345):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    *( [RandomGrayscaleDilation(kernel_bounds=(1, 3), seed=seed)] if dilation else [] ),
                                    ])

    train_dataset=MNISTNormalizedThicknessDataset(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNISTNormalizedThicknessDataset(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)