import os
import torch
import torchvision.models as models

from global_params import savedPath, idx2label
from Frame2dResNet50 import Frame2dResNet50

def load_frame_model(model_name) :
    frame_model = models.resnet50(num_classes=200)
    frame_model_checkpoint = torch.load(os.path.join(savedPath, '2dResNet-frame_model','fm-'+model_name))
    frame_model.load_state_dict(frame_model_checkpoint)
    return frame_model, idx2label

def save_frame_model(model_name) :
    modelPathFileName = os.path.join(savedPath, model_name)
    model = Frame2dResNet50()
    model_checkpoint = torch.load(modelPathFileName)
    model.load_state_dict(model_checkpoint['state_dict'])
    print('model loaded from %s' % modelPathFileName)    
    torch.save(model.frame_model.state_dict(), os.path.join(savedPath, '2dResNet-frame_model','fm-'+model_name))

  
    
    
def save_training_state(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_training_state(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
    
    
    
if __name__ == '__main__':
    model_name = "2dResNet-p4lr5-1.pth"
    save_frame_model(model_name)

    frame_model, idxlabel = load_frame_model(model_name)
    print(frame_model)
    print(idxlabel)

