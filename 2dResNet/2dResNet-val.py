import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, sys
import numpy as np
import time

from global_params import savedPath, idx2label, device
from load_moments_dataset import Moments
from Frame2dResNet50 import Frame2dResNet50

valset = Moments(subset='validation', use_frames=4)
valset_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=12)

def testModel(modelName, print_to = sys.stdout) :
    t0 = time.time()
    model = Frame2dResNet50().to(device)
    modelPathFileName = os.path.join(savedPath, modelName)
    state = torch.load(modelPathFileName)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % modelPathFileName)
    
    
    torch.manual_seed(123)
    print('video,label,pred1,prob1,pred2,prob2,pred3,prob3,pred4,prob4,pred5,prob5,correct1,correct5', 
          flush=True, file=print_to)
    
    model.eval()
    num_correct1 = 0
    num_correct5 = 0
    num_samples = 0
    torch.manual_seed(123)
    with torch.no_grad():
        for data, target, video_path in valset_loader:
            data_g = data.to(device)
            scores = model(data_g)
            probs = F.softmax(scores, dim=1).data.cpu().numpy()
            preds = probs.argsort(axis=1)[:,-1:-6:-1]
            target_np = target.numpy()
            
            batch_sz = preds.shape[0]
            num_correct1 += (preds[:,0] == target_np).sum()
            num_correct5 += sum([1 if target_np[i] in preds[i] else 0 for i in range(batch_sz)])
            num_samples += batch_sz

            for i in range(batch_sz) :
                video_info = [video_path[i]]
                video_info.append(idx2label[target_np[i]])
                correct_top_5 = 0
                for k in range(5) :
                    video_info.append(idx2label[preds[i,k]])
                    video_info.append(str(probs[i, preds[i,k]]))
                    if (preds[i, k] == target_np[i]) :
                        correct_top_5 += 1
                correct_top_1 = 1 if preds[i,0]==target_np[i] else 0
                video_info.append(str(correct_top_1))
                video_info.append(str(correct_top_5))
                print(','.join(video_info), flush=True, file=print_to)
                
            if (num_samples % 400 == 0) :
                print('Tested [{}/{} ({:.2%})]'.format(
                    num_samples, len(valset), num_samples / len(valset)), flush=True)
    acc1 = 1.0*num_correct1/num_samples
    acc5 = 1.0*num_correct5/num_samples
    print('Validation set accuracy: top-1 {}/{} ({:.2%}), top-5 {}/{} ({:.2%})'.format(
        num_correct1, num_samples, acc1, num_correct5, num_samples, acc5), flush=True)
    print('total,%d,,,,,,,,,,,%d,%d'%(num_samples, num_correct1, num_correct5), flush=True, file=print_to)
    t1 = time.time()
    print('Validation takes %fs'%(t1-t0))

if __name__ == '__main__':
    for i in range(1, 6) :
        modelName = "2dResNet-p4lr4-%d.pth"%i
        output_FName = 'val/val-'+modelName[:-4]+'.csv'
        FOut = open(output_FName, 'w')
        print('Outputing result to '+ output_FName)
        testModel(modelName, FOut)
        FOut.close()
