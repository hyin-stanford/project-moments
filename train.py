import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies_5= AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, video_id) in enumerate(data_loader):
        # print (inputs.size())
        # print ('Training data has been loaded')
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc, acc_5 = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        accuracies_5.update(acc_5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.tolist(),
            'acc': accuracies.val,
            'acc_5': accuracies_5.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Training, Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Acc_5 {acc_5.val:.3f} ({acc_5.avg:.3f})\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies,
                  acc_5= accuracies_5))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.tolist(),
        'acc': accuracies.avg,
        'acc_5':accuracies_5.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
