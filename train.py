"""
Okay we have a dataset! let's try training it.
First we need to partition the data
"""
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

# let's use pytorch! way easier.
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# custom stuff I wrote
import utils
from models import Model


if __name__ == "__main__":

    # hyperparameters. change at your leisure
    num_epochs = 1000
    batch_size = 6

    # load the train data
    train_data_path = Path("data", "train_full.csv")
    train_inputs, train_labels = utils.load_model_data(train_data_path)

    custom_dataset = utils.DLoader(train_inputs, train_labels)
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, optimizer
    train_start_time = time.perf_counter()
    model = Model()
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_history = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        epoch_loss = 0.0

        for batch_inputs, batch_labels in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch_inputs)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())

        print(f"Epoch: {epoch}: Loss: {epoch_loss}")
        loss_history.append(epoch_loss)
    print('Finished Training')
    print(f"Training took: {round(( time.perf_counter() - train_start_time ) / 60, 2)} minutes")

    # save the model with unique name (unix timestamp it) in a training folder
    # but first make the trainings folder if it doesn't exit
    if not os.path.isdir("./trainings"):
        os.mkdir("./trainings")

    model_name = int(time.time())
    train_dir = Path("trainings", f"{model_name}_{num_epochs}_{batch_size}")
    os.mkdir(train_dir)

    save_path = Path(train_dir, f"{model_name}_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    plt.clf()
    plt.plot(list(range(len(loss_history))), loss_history)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    save_path = Path(train_dir, f"{model_name}_train_loss.png")
    plt.savefig(save_path)
