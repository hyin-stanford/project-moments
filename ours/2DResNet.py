import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchvision.models as models
import time

from Moments import Moments

rootPath = "../data"   ## subject to change
resultPath = os.path.join(rootPath, 'results')

trainset = Moments(subset='training')
valset = Moments(subset='validation')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")
print("Running on device", device)        

print("Number of training videos:", len(trainset))
print("Number of validation videos:", len(valset))

trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
valset_loader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=8)


class FrameResNet50(nn.Module) :
    def __init__(self, use_pretrain=-1, num_classes=200) :
        super().__init__()
        self.frame_model = models.resnet50(num_classes=num_classes)
        if (use_pretrain >= 0) :
            self.loadPretrainedParam(use_pretrain)
        
    def forward(self, x) :
        B, C, T, H, W = x.shape
        if self.training :
            return self.frame_model(x[:,:,T//2,:,:])
        else :
            logits = self.frame_model(x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W))
            return logits.view(B, T, -1).mean(dim=1)
    
    def loadPretrainedParam(self, n_levels) :
        assert(n_levels <= 4)
        resnet_imgnet_checkpoint = torch.load(os.path.join(rootPath, 'models/resnet50-19c8e357.pth'))
        states_to_load = {}
        for name, param in resnet_imgnet_checkpoint.items() :
            if name.startswith('fc') :
                continue
            if name.startswith('layer') :
                if int(name[5]) <= n_levels :
                    states_to_load[name]=param
            else :
                states_to_load[name]=param
        model_state = self.frame_model.state_dict()
        model_state.update(states_to_load)
        self.frame_model.load_state_dict(model_state)


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def test(test_size = 2000):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in valset_loader:
            data, target = data.to(device), target.to(device)
            scores = model(data)
            _, preds = scores.max(1)
            num_correct += (preds == target).sum()
            num_samples += preds.size(0)
            # if (num_samples%100 == 0) :
            #     print("Number of test sample examined:", str(num_samples))
            if (num_samples >= test_size) :
                break

    acc = 100.0 * num_correct / num_samples
    print('\nTest set accuracy: {}/{} ({:.2f}%)\n'.format(num_correct, num_samples, acc))


def train_save(epoch, model, optimizer):
    """
    @pre: the result of @a model is the score of each category, which we will use cross-entropy loss
    """
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoch):
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # print("iteration =", iteration)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            scores = model(data)
            loss = F.cross_entropy(scores, target)
            loss.backward()
            optimizer.step()
            if iteration % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            if batch_idx == 800 :
                test()
            iteration += 1
        save_checkpoint(os.path.join(resultPath, '2d_resnet-%i.pth'%ep+1), model, optimizer)
        test()
        t1 = time.time()
        print("Epoch %d done, takes %fs\n"%(ep+1, t1-t0))
    
    # save the final model
    save_checkpoint(os.path.join(resultPath, '2d_resnet-final.pth'), model, optimizer)

torch.manual_seed(123)
model = FrameResNet50(use_pretrain=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_save(5, model, optimizer)
