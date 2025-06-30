import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from strictfire import StrictFire
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lizardist.distributed.communicator import Communicator
from lizardist.distributed.sync import Synchronizer
from lizardist.engine.model import ConvNet
from lizardist.utils.logger import get_logger


def train(rank: int, world_size: int, epochs: int = 10, batch_size: int = 32) -> None:
    # init communication
    comm = Communicator()
    sync = Synchronizer(comm)

    logger = get_logger(rank)

    # init model
    model = ConvNet()

    # broadcast initial weights from rank 0
    if rank == 0:
        logger.info("Broadcasting initial weights to all processes")
    sync.broadcast_parameters(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)

    # shard dataset
    dataset_size = len(dataset)
    shard_size = dataset_size // world_size
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size

    # handle last shard if not evenly divisible
    if rank == world_size - 1:
        end_idx = dataset_size

    dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Sync gradients
            sync.sync_gradients(model, use_bucketing=False)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 10 == 0 and rank == 0:
                logger.info(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        if rank == 0:
            avg_loss = running_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            logger.info(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    pprint.pprint(comm.get_bucket_stats())
    comm.finalize()


if __name__ == "__main__":
    from lizardist.distributed.communicator import Communicator

    comm = Communicator()
    StrictFire(lambda **kwargs: train(comm.get_rank(), comm.get_world_size(), **kwargs))
