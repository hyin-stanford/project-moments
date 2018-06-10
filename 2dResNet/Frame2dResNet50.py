import torch
import torch.nn as nn
import os
import torchvision.models as models

from global_params import pretrainedPath

class Frame2dResNet50(nn.Module) :
    def __init__(self, use_pretrain=-1, num_classes=200) :
        super().__init__()
        self.frame_model = models.resnet50(num_classes=num_classes) ## back to 50
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
        resnet_imgnet_checkpoint = torch.load(os.path.join(pretrainedPath, 'resnet50-19c8e357.pth'))
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