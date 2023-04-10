import torch
from Train.dataset import load_sidewalk_dataset
from train import train_model
import fastseg as fs


dataset_path = "segments/sidewalk-semantic"
train_ds, val_ds, num_labels = load_sidewalk_dataset(dataset_path)

# Load the Fast-SCNN model
model = fs.MobileV3Large(num_classes=num_labels)

# Set up training hyper-parameters
batch_size = 50
num_epochs = 1

# Use the DataLoader to load your datasets
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Set up the criterion (loss function)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# Check if GPU is available
device = torch.device("mps")

# Train the model
model = train_model(model=model, train_loader=train_loader, val_loader=None,
                    criterion=criterion, scheduler=None, num_epochs=num_epochs, device=device)

# Save the trained model
torch.save(model.state_dict(), "../Model/fast_scnn_model.pth")
