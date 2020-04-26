from __future__ import print_function

import os
from glob import glob

import numpy as np
import scikitplot as skplt
import torch.nn.functional as F
from matplotlib import pyplot
from torch import nn, optim
from tqdm.autonotebook import tqdm

from DataLoader import DataLoader
import  time

device = 'cuda'

current_epoch = 0
number_of_epochs = 1000
learning_rate = 0.0001
momentum = 0.5

confusion_matrix_pred_training_set = []
confusion_matrix_expected_training_set = []
confusion_matrix_pred_testing_set = []
confusion_matrix_expected_testing_set = []
loss_list_training = []
loss_list_testing = []


class OwnArchitecture(nn.Module):

    def __init__(self):
        super(OwnArchitecture, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


model = OwnArchitecture()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
model = model.to(device)


def train(epoch):
    train_loader = DataLoader.get_training_data_loader()
    model.train()
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader))

    for iteration, (input_data, expected_output_data) in progress:
        input_data, expected_output_data = input_data.to(device), expected_output_data.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        loss = F.cross_entropy(output, expected_output_data)
        loss.backward()
        optimizer.step()


def validation_testing_set():
    test_loader = DataLoader.get_testing_data_loader()
    model.eval()
    test_loss = 0
    correct = 0
    file_number = 0
    for input_data, expected_output_data in test_loader:
        input_data, expected_output_data = input_data.to(device), expected_output_data.to(device)
        output = model(input_data)
        test_loss += F.cross_entropy(output, expected_output_data, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        if current_epoch == (number_of_epochs - 1):
            global confusion_matrix_pred_testing_set, confusion_matrix_expected_testing_set
            confusion_matrix_pred_testing_set = confusion_matrix_pred_testing_set + pred.data.squeeze().tolist()
            confusion_matrix_expected_testing_set = confusion_matrix_expected_testing_set + expected_output_data.squeeze().tolist()

        predicted = pred.data.squeeze().tolist()
        expected = expected_output_data.squeeze().tolist()

        if current_epoch == number_of_epochs - 1:
            for i in range(len(predicted)):
                if predicted[i] != expected[i]:
                    img = np.transpose(input_data[i].cpu().detach().numpy(), (1, 2, 0)).squeeze()
                    file_name = "img_" + file_number.__str__() + "_" + expected[i].__str__() + "_detected_as_" + \
                                predicted[i].__str__() + ".png"
                    file_number += 1
                    pyplot.imsave("./own_architecture/testing_set/" + file_name, img)

        correct += pred.eq(expected_output_data.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    global loss_list_testing
    loss_list_testing.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def validation_training_set():
    train_loader = DataLoader.get_training_data_loader()
    model.eval()
    test_loss = 0
    correct = 0
    file_number = 0

    for input_data, expected_output_data in train_loader:
        input_data, expected_output_data = input_data.to(device), expected_output_data.to(device)
        output = model(input_data)
        test_loss += F.cross_entropy(output, expected_output_data, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        if current_epoch == (number_of_epochs - 1):
            global confusion_matrix_pred_training_set, confusion_matrix_expected_training_set
            confusion_matrix_pred_training_set = confusion_matrix_pred_training_set + pred.data.squeeze().tolist()
            confusion_matrix_expected_training_set = confusion_matrix_expected_training_set + expected_output_data.squeeze().tolist()

        predicted = pred.data.squeeze().tolist()
        expected = expected_output_data.squeeze().tolist()

        if current_epoch == number_of_epochs - 1:
            for i in range(len(predicted)):
                if predicted[i] != expected[i]:
                    img = np.transpose(input_data[i].cpu().detach().numpy(), (1, 2, 0)).squeeze()
                    file_name = "img_" + file_number.__str__() + "_" + expected[i].__str__() + "_detected_as_" + \
                                predicted[i].__str__() + ".png"
                    file_number += 1
                    pyplot.imsave("./own_architecture/training_set/" + file_name, img)

        correct += pred.eq(expected_output_data.data.view_as(pred)).cpu().sum()

    test_loss /= len(train_loader.dataset)
    global loss_list_training
    loss_list_training.append(test_loss)
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


if not os.path.exists("./own_architecture/"):
    os.makedirs("./own_architecture/")
    os.makedirs("./own_architecture/testing_set")
    os.makedirs("./own_architecture/training_set")
else:
    if not os.path.exists("./own_architecture/testing_set"):
        os.makedirs("./own_architecture/testing_set")
    else:
        files = glob("./own_architecture/testing_set/*")
        for f in files:
            os.remove(f)
    if not os.path.exists("./own_architecture/training_set"):
        os.makedirs("./own_architecture/training_set")
    else:
        files = glob("./own_architecture/training_set/*")
        for f in files:
            os.remove(f)

start = time.time()
for current_epoch in range(0, number_of_epochs):
    print("Epoch " + (current_epoch + 1).__str__() + "/" + number_of_epochs.__str__())
    train(current_epoch)
    validation_training_set()
    validation_testing_set()
end = time.time()



skplt.metrics.plot_confusion_matrix(confusion_matrix_expected_training_set, confusion_matrix_pred_training_set
                                    , title="Macierz pomyłek zestawu treningowego")
pyplot.savefig("./own_architecture/confussion_matrix_training.png")
pyplot.show()

skplt.metrics.plot_confusion_matrix(confusion_matrix_expected_testing_set, confusion_matrix_pred_testing_set
                                    , title="Macierz pomyłek zestawu testowego")
pyplot.savefig("./own_architecture/confussion_matrix_testing.png")
pyplot.show()

pyplot.plot(loss_list_training, label="Training data loss")
pyplot.plot(loss_list_testing, label="Testing data loss")
pyplot.ylabel('Loss value in epoch')
pyplot.xlabel('Number of epochs')
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=2, mode="expand", borderaxespad=0.)
pyplot.savefig("./own_architecture/loss_figure.png")
pyplot.show()
print("Training process took: " + (end - start).__str__())