import torch
import torchvision.transforms as trn
from torch.utils.data import Dataset
import glob
import os
from PIL import Image


class Moments(Dataset) :
    """
    A customized data loader for Moments-In-Time dataset.
    """    
    def __init__(self, subset='validation', use_frames=16) :
        super().__init__()
        root = os.path.join('../data/Moments_in_Time_Mini/jpg', subset) ## subject to change
        self.use_frames = use_frames
        
        self.filenames = []
        for video_path in glob.glob(os.path.join(root, "*/*")) :
            label = video_path.split('/')[-2]
            self.filenames.append((video_path, label))
        self.len = len(self.filenames)
        
        self.tf = trn.Compose([trn.Resize((224, 224)), 
                               trn.ToTensor(), 
                               trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ## subject to change
                              ])
    
    def __getitem__(self, index) :
        video_path, label = self.filenames[index]
        tot_frames = len(os.listdir(video_path)) - 1
        video = []
        for i in range(1, tot_frames+1, (tot_frames-1)//(self.use_frames-1)) :
            img = Image.open(os.path.join(video_path, 'image_{:05d}.jpg'.format(i))).convert('RGB')
            video.append(self.tf(img))
        return torch.stack(video, dim=1)

    def __len__(self) :
        return self.len
