#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from collections import defaultdict
import numpy as np
from utils import evaluate
from tqdm import tqdm
from data import transforms
from models_vit.dc_layer import DataConsistencyInKspace

DC_layer = DataConsistencyInKspace()

def evaluate_recon(net_g, data_loader, args, writer = None, epoch = None):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            input, target, mean, std, fname, slice, mask, masked_kspace = batch
            target = target.to(args.device)

            output = net_g(input.to(args.device))

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = transforms.complex_abs(output) * std + mean
            target = transforms.complex_abs(target) * std + mean

            # from matplotlib import pyplot as plt
            # input = transforms.complex_abs(input)
            # plt.imshow(input[0,:,:].cpu(),cmap='gray')
            # plt.show()
            # plt.imshow(output[0,:,:].cpu().detach().numpy(),cmap='gray')
            # plt.show()
            # plt.imshow(target[0,:,:].cpu(),cmap='gray')
            # plt.show()

            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])

        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        print('No. Volume: ', len(outputs))
        if writer != None:
            writer.add_scalar('Dev_Loss/NMSE', metrics['nmse'], epoch)
            writer.add_scalar('Dev_Loss/SSIM', metrics['ssim'], epoch)
            writer.add_scalar('Dev_Loss/PSNR', metrics['psnr'], epoch)
    return metrics['val_loss'], metrics['nmse'], metrics['ssim'], metrics['psnr']

def test_recon_save(net_g, data_loader, args):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        input_plt = []
        output_plt = []
        target_plt = []
        error_plt = []
        itr_time = []
        for idx, batch in enumerate(tqdm(data_loader)):

            input, target, mean, std, fname, slice, mask, masked_kspace = batch
            input = input.to(args.device)
            target = target.to(args.device)
            iter_data_time = time.time()
            output = net_g(input.to(args.device))
            t_comp = (time.time() - iter_data_time)
            itr_time.append(t_comp)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = transforms.complex_abs(output) * std + mean
            target = transforms.complex_abs(target) * std + mean
            input = transforms.complex_abs(input) * std + mean

            input_plt.append(input)
            output_plt.append(output)
            target_plt.append(target)
            output_1 = (output - output.min()) / (output.max() - output.min())
            target_1 = (target - target.min()) / (target.max() - target.min())
            error_plt.append(torch.sub(target_1, output_1))

            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'input': (input).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
        metrics= dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        outputs_save = {}
        targets_save = {}
        input_save = {}
        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            input = np.stack([tgt for _, tgt in sorted(inputs[fname])])
            outputs_save[fname] = output
            targets_save[fname] = target
            input_save[fname] = input
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics_avg = {metric: np.mean(values) for metric, values in metrics.items()}
        metrics_std = {metric: np.std(values) for metric, values in metrics.items()}
        avg_time = np.mean(itr_time)

    return metrics_avg,metrics_std,avg_time,output_plt,target_plt,input_plt,error_plt