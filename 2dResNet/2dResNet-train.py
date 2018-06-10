import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, sys
import numpy as np
import time

from global_params import rootPath, savedPath, device
from load_moments_dataset import Moments
from Frame2dResNet50 import Frame2dResNet50
from utils import save_training_state, load_training_state

trainset = Moments(subset='training', use_frames=4)
valset = Moments(subset='validation', use_frames=4)
print("Number of training videos:", len(trainset))
print("Number of validation videos:", len(valset))

trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=12)
valset_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=12)

def test(model, test_size = 2000, print_to = sys.stdout):
    model.eval()
    num_correct1 = 0
    num_correct5 = 0
    num_samples = 0
    torch.manual_seed(123)
    with torch.no_grad():
        for data, target, _ in valset_loader:
            data_g = data.to(device)
            scores = model(data_g).data.cpu().numpy()
            preds = scores.argsort(axis=1)[:,-5:]
            target_np = target.numpy()
            batch_sz = preds.shape[0]
            num_correct1 += (preds[:,-1] == target_np).sum()
            num_correct5 += sum([1 if target_np[i] in preds[i] else 0 for i in range(batch_sz)])
            num_samples += batch_sz
            if (debug) :
                break
            if (num_samples % 400 == 0) :
                print('Tested [{}/{} ({:.2%})]'.format(
                    num_samples, len(valset), num_samples / len(valset)), flush=True)
    acc1 = 1.0*num_correct1/num_samples
    acc5 = 1.0*num_correct5/num_samples
    print('\tValidation set accuracy: top-1 {}/{} ({:.2%}), top-5 {}/{} ({:.2%})'.format(
        num_correct1, num_samples, acc1, num_correct5, num_samples, acc5), flush=True, file=print_to)


def train_save(epoch, model, optimizer, param_val, print_to = sys.stdout, epoc_start = 0):
    """
    @pre: the return of @a model is the score of each category, which we will use cross-entropy loss
    """
    print(param_val)
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoc_start, epoc_start+epoch):
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            scores = model(data)
            loss = F.cross_entropy(scores, target)
            loss.backward()
            optimizer.step()
            if iteration % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()), flush=True, file=print_to)
            if batch_idx == 800 :
                test(model, print_to=print_to)
                model.train()
            iteration += 1
        save_training_state(os.path.join(savedPath, '2dResNet-'+param_val+'-%d.pth'%(ep+1)), model, optimizer)
        test(model, print_to=print_to)
        model.train()
        t1 = time.time()
        print("Epoch %d done, takes %fs"%(ep+1, t1-t0), flush=True)
        print("Epoch %d done, takes %fs"%(ep+1, t1-t0), flush=True, file=print_to)

def finetuneparam(pre_imgnet, nllr, use_pre_ours=0, epocs=3) :
    torch.manual_seed(123)
    param_val = 'p%dlr%d'%(pre_imgnet, nllr)
    if (use_pre_ours == 0) :
        model = Frame2dResNet50(use_pretrain=pre_imgnet).to(device)
        optimizer = optim.Adam(model.parameters(), lr=10**(-nllr))
        torch.manual_seed(123)
    else :
        model = Frame2dResNet50().to(device)
        optimizer = optim.Adam(model.parameters(), lr=10**(-nllr))
        load_training_state(os.path.join(savedPath, '2dResNet-'+param_val+'-%d.pth'%(use_pre_ours)), model, optimizer)
        torch.manual_seed(123)
    log_file = open('log-train/log-p%dlr%d.txt'%(pre_imgnet, nllr), 'a')
    train_save(epocs, model, optimizer, param_val, print_to=log_file, epoc_start=use_pre_ours)
    log_file.close()

if __name__ == '__main__':
    finetuneparam(4, 5, use_pre_ours=0, epocs=6)