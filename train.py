import argparse
import os
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image

from model import Generator, Discriminator
from utils import init_weights, save_checkpoints
import data_loader


def train(dataloader):
    """ Train the model on `num_steps` batches
    Args:
        dataloader : (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        num_steps : (int) # of batches to train on, each of size args.batch_size
    """

    # Define Generator, Discriminator
    G = Generator(out_channel=ch).to(device) # MNIST channel: 1, CIFAR-10 channel: 3
    D = Discriminator(in_channel=ch).to(device)

    # adversarial loss
    loss_fn = nn.BCELoss()

    # Initialize weights
    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # -----Training----- #
    for epoch in range(epochs):
        # For each batch in the dataloader

        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            D.zero_grad()
            # Format batch
            real_cpu = data[0].to(device) # load image batch size
            b_size = real_cpu.size(0) # batch size
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device, requires_grad=False) # real batch

            # Forward pass **real batch** through D
            output = D(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss_fn(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            ## Train with **all-fake** batch
            # Generate noise batch of latent vectors
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            # Generate fake image batch with G
            fake = G(noise)
            label.fill_(fake_label) # fake batch

            # Classify all fake batch with D
            output = D(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss_fn(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = D(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss_fn(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizer_G.step()

            # Save fake images generated by Generator
            batches_done = epoch * len(dataloader) + i
            if batches_done % 400 == 0:
                save_image(fake.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        print(f"[Epoch {epoch + 1}/{epochs}] [D loss: {errD.item():.4f}] [G loss: {errG.item():.4f}]")

        # Save Generator model's parameters
        save_checkpoints(
            {'epoch': i + 1,
             'state_dict': G.state_dict(),
             'optim_dict': optimizer_G.state_dict()},
            checkpoint='./ckpt/',
            is_G=True
        )

        # Save Discriminator model's parameters
        save_checkpoints(
            {'epoch': i + 1,
             'state_dict': D.state_dict(),
             'optim_dict': optimizer_D.state_dict()},
            checkpoint='./ckpt/',
            is_G=False
        )


if __name__ == '__main__':

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/',
                        help="Directory containing the dataset")
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument('--b1', type=float, default=0.5,
                        help="Momentum decay rate")
    parser.add_argument('--b2', type=float, default=0.9,
                        help="Adaptive term decay rate")
    parser.add_argument('--latent_dim', type=int, default=100,
                        help="Dimensionality of the latent space")
    parser.add_argument('--epoch', type=int, default=50,
                        help="Total training epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--img_ch', type=int, default=1,
                        help="image channel size(MNIST: 1, CIFAR-10: 3)")
    parser.add_argument('--gpu', action='store_true', default='False',
                        help="GPU available")

    # Load the parameters from parser
    args = parser.parse_args()

    lr = args.lr
    b1 = args.b1
    b2 = args.b2
    latent_dim = args.latent_dim
    epochs = args.epoch
    batch_size = args.batch_size
    ch = args.img_ch

    logging.info("Loading the training dataset...")

    # fetch train dataloader
    train_dataloader = data_loader.train_data_loader()

    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s).".format(epochs))
    train(train_dataloader)