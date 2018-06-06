import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies_5= AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, video_id) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        print ('This line has been executed')
        # print (inputs.size())

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        # inputs= Variable(inputs)
        # print (inputs.view(-1,3,inputs.size(3), inputs.size(4)).size())
        inputs = Variable(inputs.view(-1,3,inputs.size(3), inputs.size(4)), volatile=True)
        # print (inputs.size())
        targets = Variable(targets, volatile=True)
        outputs = model(inputs).mean(dim=0, keepdim= True)
        print (outputs.size())
        # outputs= outputs.view(-1,16)
        # print (outputs.size())
        # print (targets.size())
        
        loss = criterion(outputs, targets)
        acc, acc_5 = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        accuracies_5.update(acc_5, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if opt.debug:
          if acc_5==0:
            print (video_id)
            print (targets)
        print('Validation, Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'
              'Acc_5 {acc_5.val:.3f} ({acc_5.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies, 
                  acc_5= accuracies_5))



    logger.log({'epoch': epoch, 'loss': losses.avg.tolist(), 'acc': accuracies.avg, 'acc_5':accuracies_5.avg})

    return losses.avg
