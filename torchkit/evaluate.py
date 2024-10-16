import torch

def evaluate(model, dataloader, loss_fn, device, metrics):
    model.eval()
    total_loss = 0
    total_metrics = {name: 0 for name in metrics.keys()}
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            
            for name, metric_fn in metrics.items():
                total_metrics[name] += metric_fn(y_pred, y).item()
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {name: value / len(dataloader) for name, value in total_metrics.items()}
    
    return avg_loss, avg_metrics