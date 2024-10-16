import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_data_loaders(data_path, batch_size):
    """
    Creates data loaders for training and validation.

    Parameters:
    ----------
    data_path : str
    Path to the dataset.
    batch_size : int
    Number of samples in each batch.

    Returns:
    -------
    train_loader : DataLoader
    Data loader for training.
    val_loader : DataLoader
    Data loader for validation.
    classes : list
    List of classes in the dataset.

    Notes:
    -----
    This function creates data loaders for training and validation using the given dataset and batch size.
    It uses the `ImageFolder` class from PyTorch to load the dataset and the `DataLoader` class to create the data loaders.
    The `ImageFolder` class is used to load the dataset and the `DataLoader` class is used to create the data loaders.
    The `DataLoader` class is used to create the data loaders.
    The `DataLoader` class is used to create the data loaders.

    Examples:
    --------
    >>> get_data_loaders('data/food_data', 32)
    (DataLoader, DataLoader, list)
    """

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=data_path, transform=data_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, dataset.classes
