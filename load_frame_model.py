import torch
import os
import torchvision.models as models

def load_frame_model(model_name) :
	frame_model_checkpoint = torch.load(model_name)
	frame_model = models.resnet50(num_classes=200)
	frame_model.load_state_dict(frame_model_checkpoint)
	return frame_model


model_name = 'fm-2d_resnet-3.pth'
model = load_frame_model(model_name)
print(model)