from tqdm import tqdm
from typing import Dict, List, Any, Callable

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast, GradScaler


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    scaler: GradScaler,
    scheduler: Any = None,
    metrics: Dict[str, Callable] = None,
    gradient_clip_val: float = None,
) -> Dict[str, List[float]]:
    """
    Train and test a PyTorch model for multiple epochs with advanced features.

    This function performs training and testing for a specified number of epochs,
    incorporating mixed precision training, learning rate scheduling, custom metric
    computation, and gradient clipping.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained and tested.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader containing the training data.
    test_dataloader : torch.utils.data.DataLoader
        DataLoader containing the test data.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model parameters.
    loss_fn : torch.nn.Module
        Loss function to be optimized.
    epochs : int
        Number of epochs to train for.
    device : torch.device
        Device to perform computations on (e.g., 'cuda' or 'cpu').
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    scheduler : Any, optional
        Learning rate scheduler (default is None).
    metrics : Dict[str, Callable], optional
        Dictionary of metric functions to compute during training and testing (default is None).
    gradient_clip_val : float, optional
        Maximum norm of the gradients for gradient clipping (default is None).

    Returns
    -------
    Dict[str, List[float]]
        A dictionary containing lists of training and testing metrics for each epoch.
        Keys include 'train_loss', 'train_metrics', 'test_loss', 'test_metrics'.

    Notes
    -----
    - Uses mixed precision training with autocast and GradScaler.
    - Supports dynamic learning rate adjustment if a scheduler is provided.
    - Allows computation of multiple custom metrics for both training and testing.
    - Implements optional gradient clipping to prevent exploding gradients.
    - Displays a progress bar for epochs and prints metrics after each epoch.

    Examples
    --------
    >>> model = MyModel()
    >>> train_dataloader = DataLoader(train_dataset, batch_size=32)
    >>> test_dataloader = DataLoader(test_dataset, batch_size=32)
    >>> loss_fn = nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> scaler = GradScaler()
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=10)
    >>> metrics = {"accuracy": accuracy_fn}
    >>> results = train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
    ...                 epochs=10, device=device, scaler=scaler, scheduler=scheduler,
    ...                 metrics=metrics, gradient_clip_val=1.0)
    >>> print(f"Final train loss: {results['train_loss'][-1]:.4f}, accuracy: {results['train_metrics']['accuracy'][-1]:.4f}")
    >>> print(f"Final test loss: {results['test_loss'][-1]:.4f}, accuracy: {results['test_metrics']['accuracy'][-1]:.4f}")
    """
    results = {
        "train_loss": [],
        "train_metrics": {name: [] for name in (metrics.keys() if metrics else [])},
        "test_loss": [],
        "test_metrics": {name: [] for name in (metrics.keys() if metrics else [])},
    }

    epoch_pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_pbar:
        # Training step
        train_stats = train_step(
            model, train_dataloader, loss_fn, optimizer, device, scaler, scheduler, metrics, gradient_clip_val
        )

        # Testing step
        test_stats = test_step(model, test_dataloader, loss_fn, device, metrics)

        # Update results
        results["train_loss"].append(train_stats["loss"])
        results["test_loss"].append(test_stats["loss"])
        for name in metrics.keys() if metrics else []:
            results["train_metrics"][name].append(train_stats[name])
            results["test_metrics"][name].append(test_stats[name])

        
        epoch_pbar.set_postfix({
            "Train Loss": f"{results['train_loss'][-1]:.4f}",
            "Val Loss": f"{results['test_loss'][-1]:.4f}",
            "Train Acc": f"{results['train_metrics']['accuracy'][-1]:.4f}",
            "Val Acc": f"{results['test_metrics']['accuracy'][-1]:.4f}"
        })

        

    return results


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    scheduler: Any = None,
    metrics: Dict[str, Callable] = None,
    gradient_clip_val: float = None,
) -> Dict[str, float]:
    """
    Train a PyTorch model for a single epoch with advanced features.

    This function performs a full training epoch, including mixed precision training,
    learning rate scheduling, custom metric computation, and gradient clipping.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the training data.
    loss_fn : torch.nn.Module
        Loss function to be optimized.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model parameters.
    device : torch.device
        Device to perform computations on (e.g., 'cuda' or 'cpu').
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    scheduler : Any, optional
        Learning rate scheduler (default is None).
    metrics : Dict[str, Callable], optional
        Dictionary of metric functions to compute during training (default is None).
    gradient_clip_val : float, optional
        Maximum norm of the gradients for gradient clipping (default is None).

    Returns
    -------
    Dict[str, float]
        A dictionary containing average loss and metrics for the epoch.
        Keys include 'loss' and any additional metrics specified.

    Notes
    -----
    - Uses mixed precision training with autocast and GradScaler.
    - Supports dynamic learning rate adjustment if a scheduler is provided.
    - Allows computation of multiple custom metrics.
    - Implements optional gradient clipping to prevent exploding gradients.
    - Displays a progress bar with live updates of loss and metrics.

    Examples
    --------
    >>> model = MyModel()
    >>> dataloader = DataLoader(dataset, batch_size=32)
    >>> loss_fn = nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> scaler = GradScaler()
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=10)
    >>> metrics = {"accuracy": accuracy_fn}
    >>> train_stats = train_step(model, dataloader, loss_fn, optimizer, device, scaler,
    ...                          scheduler=scheduler, metrics=metrics, gradient_clip_val=1.0)
    >>> print(f"Training loss: {train_stats['loss']:.4f}, Accuracy: {train_stats['accuracy']:.4f}")
    """
    model.train()

    train_stats = {"loss": 0.0}
    if metrics:
        train_stats.update({name: 0.0 for name in metrics.keys()})

    num_batches = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Training")
    for batch, (X, y) in pbar:
        X, y = X.to(device), y.to(device)

        with autocast():
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        scaler.scale(loss).backward()

        if gradient_clip_val:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        train_stats["loss"] += loss.item()

        if metrics:
            with torch.no_grad():
                for name, metric_fn in metrics.items():
                    train_stats[name] += metric_fn(y_pred, y).item()

        pbar.set_postfix({k: f"{v / (batch + 1):.4f}" for k, v in train_stats.items()})

    train_stats = {k: v / num_batches for k, v in train_stats.items()}
    return train_stats


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    metrics: Dict[str, Callable] = None,
) -> Dict[str, float]:
    """
    Test a PyTorch model for a single epoch.

    This function performs a full testing epoch, including custom metric computation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be tested.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the test data.
    loss_fn : torch.nn.Module
        Loss function to be evaluated.
    device : torch.device
        Device to perform computations on (e.g., 'cuda' or 'cpu').
    metrics : Dict[str, Callable], optional
        Dictionary of metric functions to compute during testing (default is None).

    Returns
    -------
    Dict[str, float]
        A dictionary containing average loss and metrics for the epoch.
    """
    model.eval()
    test_stats = {"loss": 0.0}
    if metrics:
        test_stats.update({name: 0.0 for name in metrics.keys()})

    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_stats["loss"] += loss.item()

            if metrics:
                for name, metric_fn in metrics.items():
                    test_stats[name] += metric_fn(y_pred, y).item()

    test_stats = {k: v / num_batches for k, v in test_stats.items()}
    return test_stats
