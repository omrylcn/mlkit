import torch



def accuracy_fn(y_pred, y_true):
    return (torch.argmax(y_pred, dim=1) == y_true).float().mean()