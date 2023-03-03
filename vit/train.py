import os
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from config import args
from dataloader import getDataLoader
from utils import remove_dir_and_create_dir, create_model, model_parallel


def main(args):
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # weight_dir = args.summary_dir + "/weights"
    # remove_dir_and_create_dir(weight_dir)

    # Record Logs
    log_dir = args.summary_dir + "/logs"
    remove_dir_and_create_dir(log_dir)
    writer = SummaryWriter(log_dir)

    # Define the multiprocess
    number_of_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process.'.format(number_of_workers))

    # Get dataset
    train_loader, train_dataset = getDataLoader(data_dir=args.dataset_train_dir,
                                                batch_size=args.batch_size,
                                                num_workers=number_of_workers,
                                                train_or_test=True)
    number_of_train = len(train_dataset)
    test_loader, test_dataset = getDataLoader(data_dir=args.dataset_test_dir,
                                              batch_size=args.batch_size,
                                              num_workers=number_of_workers,
                                              train_or_test=False)
    number_of_test = len(test_dataset)
    print('Using {} images for training.'.format(number_of_train))
    print('Using {} images for testing.'.format(number_of_test))

    # Create model
    model = create_model(args)
    model = model_parallel(args, model)
    model.to(device)

    # Define loss function
    loss_function = nn.CrossEntropyLoss()

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Start training
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        # Train model
        model.train()
        # Define train accuracy and loss
        train_accuracy = 0.0
        train_loss = []

        # Display a visible process bar
        train_bar = tqdm(train_loader)
        for data in train_bar:
            # Display train process
            train_bar.set_description("epoch {}".format(epoch))
            # Get images and labels
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # Empty the gradient
            optimizer.zero_grad()
            # Input the data into model
            outputs = model(images)
            prediction = torch.max(outputs, dim=1)[1]
            # Calculate the loss
            loss = loss_function(outputs, labels)
            # Get gradient automatically
            loss.backward()
            # Optimize the model parameters
            optimizer.step()
            scheduler.step()
            # Print statistics
            train_loss.append(loss.item())
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))
            train_accuracy += torch.eq(labels, prediction).sum()
            # Clear batch variables from memory
            del images, labels

        # Test model
        model.eval()
        # Define test accuracy and loss
        test_accuracy = 0.0
        test_loss = []
        with torch.no_grad():
            for data in test_loader:
                # Get images and labels
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # Input the data into model
                outputs = model(images)
                # Calculate the loss
                loss = loss_function(outputs, labels)
                prediction = torch.max(outputs, dim=1)[1]
                # Print statistics
                test_loss.append(loss.item())
                test_accuracy += torch.eq(labels, prediction).sum()
                # Clear batch variables from memory
                del images, labels

        # Statistics of data
        train_accuracy = train_accuracy / number_of_train
        test_accuracy = test_accuracy / number_of_test
        print("=> train_loss: {:.4f}, train_accuracy: {:.4f}, test_loss: {:.4f}, test_accuracy: {:.4f}".
              format(np.mean(train_loss), train_accuracy, np.mean(test_loss), test_accuracy))

        # Record data by TensorBoard
        writer.add_scalar("train_loss", np.mean(train_loss), epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)
        writer.add_scalar("test_loss", np.mean(test_loss), epoch)
        writer.add_scalar("test_accuracy", test_accuracy, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # Judge whether this epoch has the best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # torch.save(model.state_dict(), "{}/epoch={}_test_accuracy={:.4f}.pth".format(weights_dir, epoch, val_accurate))


if __name__ == '__main__':
    main(args)
