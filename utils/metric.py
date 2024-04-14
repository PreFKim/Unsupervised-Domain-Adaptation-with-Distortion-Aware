import torch

def accuracy(pred, label):
    with torch.no_grad():

        b,c,h,w = pred.shape

        _,pred = torch.max(pred, dim=1)
        accuracy = torch.sum(pred==label)/(b*h*w)
    return accuracy

def miou(pred,label):
    with torch.no_grad():
        _,pred = torch.max(pred, dim=1)
        intersection = torch.sum(((pred<12) & (pred==label)))
        union = torch.sum((pred<12)|(label<12))


        iou = intersection / (union+1e-9)

    return iou
