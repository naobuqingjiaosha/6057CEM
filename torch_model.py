import os
import time
import torchvision.models as models
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import optim, save
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch import nn
from torchvision import transforms
from torch.optim import lr_scheduler
import logging
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
my_transforms = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),  # type of data
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Standardization
    ]
)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
def log(nr):
    logging.warning(': ' + nr)


def time_sj():
    times = time.time()
    local_time = time.localtime(times)
    return str(time.strftime("%Y-%m-%d %H:%M:%S  ", local_time))


def my_transferResnet():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    model.fc = nn.Sequential(
        nn.Linear(512, 5, bias=True),
    )
    for p in model.parameters():
        p.requires_grad = False
    for layer in [model.layer4.parameters(), model.fc.parameters()]:
        for p in layer:
            p.requires_grad = True
    
    return model


def reasoning():
    print(f'{time_sj()}Start validating the test set')
    model.eval()
    classes = data_class
    n_classes = len(classes)
    target_num = torch.zeros((1, n_classes))
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))
    #y_true is the onehot representation of the real label,
    #y_scores is the confidence of the predicted label, and there is a print below
    y_true,y_scores=[],[]
        
    with torch.no_grad():
        for step, (image, label) in enumerate(test_loader):
            # get results
            image = image.to(device)
            
            label = label.to(device)
            outputs = model(image)
            
            scores, predicted = outputs.max(1)
            
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            score_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), scores.cpu().view(-1, 1))
            y_scores.extend(score_mask.cpu().numpy())
#             print('pre_mask:',pre_mask)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(outputs.size()).scatter_(1, label.data.cpu().view(-1, 1), 1.)
            y_true.extend(tar_mask.cpu().numpy())
#             print('tar_mask:',pre_mask)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)
            
        print(y_scores)
        print(y_true)
        #Plot PR curves for each category
        #calculate
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        precision_dict, recall_dict = dict(), dict()
        for i in range(len(classes)):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true[:, i], y_scores[:,i])
    
        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
        print(f'Test Accuracy is {accuracy.item()}')
        recall, precision, F1 = recall.numpy().tolist(), precision.numpy().tolist(), F1.numpy().tolist()
        
        plt.figure()
        for i in range(len(classes)):
            plt.plot(recall_dict[i], precision_dict[i], label='Precision-recall curve of class {0}(area={1:0.2f})'.format(classes[i],precision[0][i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig('./PR-curve.png')
        plt.show()
        
        for i in range(len(classes)):
            print(f'class: {classes[i]}, '
                  f'recall: {round(recall[0][i], 2)}, '
                  f'precision: {round(precision[0][i], 2)}, '
                  f'F1: {round(F1[0][i], 2)}')


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 64, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5, 5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64 * (150//8) * (150//8), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

    #Here are two different models, choose one to run each time
    def forward(self, image):
        #x = self.conv1(image)
        x = self.conv2(image)
        #image_viwed = x.view(-1,  8 * 8 * 64)
        image_viwed = x.view(-1, 64 * (150//8) * (150//8))
        #out = self.fc1(image_viwed)
        out = self.fc2(image_viwed)
        return out


def loader_data():
    print(f'{time_sj()}start loading data')
    dataset_data = datasets.ImageFolder(file_path, transform=my_transforms)
    data_class = dataset_data.classes
    print(dataset_data.classes)  # Returns the paths and categories of images from all folders
    train_size = int(len(dataset_data) * proportion)
    #train_size = int(0.8 * len(dataset_data))
    test_size = int(len(dataset_data)-train_size)
    #train_size = int((1 - (verification + proportion)) * len(dataset_data))
    #validation_size = int(verification * len(dataset_data))
    #test_size = int(len(dataset_data) - (train_size + validation_size))
    print(f'{time_sj()}The current total number of images is:{len(dataset_data)}')
    print(f'{time_sj()}The number of pictures in the training set is:{train_size} The number of test sets is:{test_size}')
    train_set, test_set = random_split(dataset_data, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    #val_loader = DataLoader(val_set, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, drop_last=True)
    print(f'{time_sj()}data loaded')
    return train_loader, test_loader, data_class


def init_model(model_opt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) != 'cuda':
        print('Recommended GPU operation, current CPU operation')
    if model_opt==0:
        model = My_model().to(device)
    else:
        model = my_transferResnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return device, model, optimizer, loss_function, exp_lr_scheduler


# if os.path.exists("./tor_test_models.pkl"):
#     # print('start loading')
#     model.load_state_dict(torch.load("./tor_test_models.pkl"))

class EarlyStopping:
    def __init__(self, patience=5):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            log(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                f'Best results observed at epoch {self.best_epoch}, best model saved as tor_test_models.pkl.\n'
                f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                f'i.e. `python=30` or use `patience=0` to disable EarlyStopping.')
        return stop

def train(epoch):
    train_total_loss = []  # Used to save each loss and finally calculate the average
    model.train()  # model training
    train_succeed = []  # Calculation accuracy
    for (images, labels) in train_loader:
        optimizer.zero_grad()  # Gradient set to 0
        output = model(images.to(device))  # forward propagation
        result = output.max(dim=1).indices
        train_succeed.append(result.eq(labels.to(device)).float().mean().item())  # percentage of accuracy
        # train_correct = (result == labels.to(device)).sum()  # correct prediction
        loss = loss_function(output, labels.to(device))  # calculate loss
        train_total_loss.append(loss.item())  # Add each loss to the list and finally calculate the average
        loss.backward()  # backpropagation
        optimizer.step()  # optimizer update
    plt_train_acc.append(np.mean(train_succeed))
    plt_train_loss.append(np.mean(train_total_loss))
    exp_lr_scheduler.step()
    #save(model.state_dict(), './tor_test_models.pkl')
    #save(optimizer.state_dict(), './tor_test_optimizer.pkl')
    # if os.path.exists("./tor_test_models.pkl"):
    #     # print('start loading')
    #     model.load_state_dict(torch.load("./tor_test_models.pkl"))

    succeed = []  # Calculation accuracy
    total_loss = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
        #for image, label in test_loader:
            # get results
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)  # The returned data format is Tensor
            result = output.max(dim=1).indices  # Returns the maximum label index
            succeed.append(result.eq(labels).float().mean().item())  # percentage of accuracy
            # Calculate the loss by the result
            loss = loss_function(output, labels)
            total_loss.append(loss.item())  # average loss
        plt_test_acc.append(np.mean(succeed))
        plt_test_loss.append(np.mean(total_loss))
    print(time_sj() + 'current number of{}epoch, total of {}epoch,Accuracy is{},Loss is{},test set accuracy{},test set loss{}'.format(epoch + 1, epoch_total,
                                                                                    np.around(np.mean(train_succeed),
                                                                                              2),
                                                                                    np.around(np.mean(train_total_loss),
                                                                                              2),
                                                                                    np.around(np.mean(succeed), 2),
                                                                                    np.around(np.mean(total_loss), 2)))
    stop = stopper(epoch=epoch + 1, fitness=np.mean(total_loss))  # early stop check
    return stop


def plt_loss(train_acc, test_acc, train_loss, test_loss):
    x_train = [i for i in range(len(train_acc))]
    x_test = [i for i in range(len(test_acc))]
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x_train, train_acc, label='train_acc')
    plt.plot(x_test, test_acc, label='test_acc')
    plt.plot(x_train, train_loss, label='train_loss')
    plt.plot(x_test, test_loss, label='test_loss')
    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("acc", fontsize=15)
    plt.legend()
    plt.grid()
    plt.show()

def plt_compare(train, test, classification, path):
    x = [i for i in range(len(train))]
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x, train, label='train_' + classification)
    plt.plot(x, test, label='test_' + classification)
    plt.xlabel("epochs", fontsize=15)
    plt.ylabel(classification, fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    #tr = ""
    file_path = r'C:\Users\92912\OneDrive - Coventry University\神经网络\flower\Dataset'  # Set image path
    epoch_total = 100  # Set the total number of training rounds
    proportion = 0.8  # Ratio of training set
    patience = 10  # Set early stop epochs default 5
    if patience:
        stopper, stop = EarlyStopping(patience=patience), False
    train_loader, test_loader, data_class = loader_data()
    device, model, optimizer, loss_function, exp_lr_scheduler = init_model(model_opt=1)
    plt_train_acc, plt_test_acc = [], []
    plt_train_loss, plt_test_loss = [], []
    print(f'{time_sj()}start training')
    for i in range(epoch_total):
        if train(i): break
    time.sleep(1)
    print(f'{time_sj()}finished training')
    reasoning()
    #plt_loss(plt_train_acc, plt_test_acc, plt_train_loss, plt_test_loss)
    plt_compare(plt_train_acc, plt_test_acc, 'acc', 'acc.png')
    plt_compare(plt_train_loss, plt_test_loss, 'loss', 'loss.png')
    print(f'{time_sj()}The program has been fully executed')