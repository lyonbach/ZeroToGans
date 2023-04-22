from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tarfile

from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

import matplotlib.pyplot as plt



def show_example(image, label, dataset):

    print(f"Label: {dataset.classes[label]}")
    plt.imshow(image.permute(1, 2, 0))


def show_batch(data_loader):

    for images, _ in data_loader:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

def apply_kernel(image, kernel):

    ri, ci = image.shape
    rk, ck = kernel.shape
    ro, co = ri - rk + 1, ci - ck + 1
    
    output = torch.zeros((ro, co))
    
    for i in range(ro):
        for j in range(co):
            output[i, j] = torch.sum(image[i:i+rk, j:j+ck] * kernel)
    
    return output

def test_apply_kernel():

    sample_image = torch.tensor([
        [3, 3, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [3, 1, 2, 2, 3],
        [2, 0, 0, 2, 2],
        [2, 0, 0, 0, 1],
    ], dtype=torch.float32)

    sample_kernel = torch.tensor([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2],
    ], dtype=torch.float32)

    print(apply_kernel(sample_image, sample_kernel))

def accuracy(output, labels):

    _, predictions = torch.max(output, dim=1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))


@torch.no_grad()
def evaluate(model, validation_data_loader):

    model.eval()
    outputs = [model.validation_step(batch) for batch in validation_data_loader]
    return model.validaiton_epoch_end(outputs)

def fit(epochs, lr, model, training_data_loader, validation_data_loader, optimization_function=torch.optim.SGD):

    history = []
    optimizer = optimization_function(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        training_losses = []
        for batch in training_data_loader:
            loss = model.training_step(batch)
            training_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        result = evaluate(model, validation_data_loader)
        result["train_loss"] = torch.stack(training_losses).mean().item()
        history.append(result)
        model.epoch_end(epoch, result)
    return history


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch
        output = self(images)

        return F.cross_entropy(output, labels)

    def validation_step(self, batch):

        images, labels = batch
        output = self(images)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validaiton_epoch_end(self, output):

        batch_losses = [x["val_loss"] for x in output]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in output]
        epoch_acc = torch.stack(batch_accs).mean()

        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, result):

        print(f"Epoch[{epoch + 1}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, acc_loss:{result['val_acc']:.4f}")


class Cifar10CnnModel(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # output: 32 x 32 x 32
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # MaxpPool actually reduces the image size. Output becomes 64 x 16 x 16.
            # 64 channels does not make any sense in terms of visualization but it holds some values.
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Once again the image size is being reduced and the output becomes 64 x 8 x 8-
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output is now 256 x 4 x 4.

            nn.Flatten(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # And finally we convert the output to match 10 labels.
        )

    def forward(self, xb):

        return self.network(xb)



if __name__ == "__main__":

    random_seed = 42
    validation_size = 5000 # 10% of training set.
    batch_size = 128
    torch.manual_seed(random_seed)


    target_folder = Path("../../data/CIFAR10")
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, target_folder)

    # with tarfile.open(target_folder / "cifar10.tgz", "r:gz") as tar:
    #     tar.extractall(path=target_folder)

    dataset = ImageFolder(target_folder / "cifar10"/"train", transform=ToTensor())
    training_size = len(dataset) - validation_size
    training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    training_data_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_data_loader = DataLoader(validation_dataset, 2*batch_size, num_workers=4, pin_memory=True)

    # show_batch(training_data_loader)
    # test_apply_kernel()

    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = .001

    model = Cifar10CnnModel()
    evaluation = evaluate(model, validation_data_loader)
    print(evaluation)

    history = fit(num_epochs, lr, model, training_data_loader, validation_data_loader, opt_func)
    print(history)

    evaluation = evaluate(model, validation_data_loader)
    print(evaluation)
    