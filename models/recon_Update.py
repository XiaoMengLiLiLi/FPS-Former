#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
from torch.nn import functional as F
from data import transforms
from tqdm import tqdm
from models_vit.dc_layer import DataConsistencyInKspace

DC_layer = DataConsistencyInKspace()

def train_recon(net, data_loader, optimizer, epoch, args, writer):
    net.train()
    # train and update
    epoch_loss = []
    batch_loss = []
    iter_data_time = time.time()
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        input, target, mean, std, fname, slice, mask, masked_kspace = batch
        target = target.to(args.device)
        output = net(input.to(args.device))

        output = transforms.complex_abs(output)
        target = transforms.complex_abs(target)

        # from matplotlib import pyplot as plt
        # input = transforms.complex_abs(input)
        # plt.imshow(input[0,:,:].cpu(),cmap='gray')
        # plt.show()
        # plt.imshow(output[0,:,:].cpu().detach().numpy(),cmap='gray')
        # plt.show()
        # plt.imshow(target[0,:,:].cpu(),cmap='gray')
        # plt.show()

        loss = F.l1_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.verbose and batch_idx % 10 == 0:
            print('Update Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_idx * len(input), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.detach().item()))
            t_comp = (time.time() - iter_data_time)
            iter_data_time = time.time()
            print('itr time: ',t_comp)
            print('lr: ',optimizer.param_groups[0]['lr'])
        batch_loss.append(loss.detach().item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        writer.add_scalar('TrainLoss/L1/'+ args.train_dataset, sum(epoch_loss) / len(epoch_loss), epoch)

    return sum(epoch_loss) / len(epoch_loss)

