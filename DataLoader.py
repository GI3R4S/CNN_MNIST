import torch
from torchvision import datasets, transforms

batch_size = 60


class DataLoader():
    @staticmethod
    def get_training_data_loader():
        train_dataset = datasets.MNIST(root='mnist_data',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        return train_loader

    @staticmethod
    def get_testing_data_loader():
        test_dataset = datasets.MNIST(root='mnist_data',
                                      train=False,
                                      transform=transforms.ToTensor())

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        return test_loader
