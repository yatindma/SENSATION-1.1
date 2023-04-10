import torch
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, train_loader, val_loader, criterion, scheduler, num_epochs, device):
    """
    We are not using the val_loaded here, but you can use it to validate your model
    """

    # model to device
    model.to(device)
    # Set up the learning rate , optimizer and scheduler
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)  # Adjust these parameters as needed

    start_time = time.time()
    total_batches = len(train_loader) * num_epochs

    # Set up the progress bar
    pbar = tqdm(total=total_batches, desc="Training progress")

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update the progress bar
            remaining_batches = total_batches - (i + 1 + len(train_loader) * epoch)
            elapsed_time = time.time() - start_time
            remaining_time = remaining_batches * (elapsed_time / (i + 1 + len(train_loader) * epoch))
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs} (Batch {i + 1}/{len(train_loader)}) ({remaining_time:.1f}s left)")
            pbar.update(1)

        epoch_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, train loss: {epoch_loss:.4f},"
              f" time: {epoch_time:.2f}s, lr: {scheduler.get_last_lr()[0]:.6f}")  # Print the current learning rate
        scheduler.step()

    # Close the progress bar
    pbar.close()

    return model
