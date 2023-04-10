import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from datasets import load_dataset
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms
import os

# Set the API key as an environment variable
os.environ["HUGGINGFACE_API_KEY"] = "xyz"


def remap_labels(label, id2label):
    remapped_label = torch.zeros_like(label)
    for label_id, label_name in id2label.items():
        remapped_label[label == label_id] = label_id
    return remapped_label


class SidewalkDataset(Dataset):
    def __init__(self, dataset, id2label):
        self.dataset = dataset
        self.id2label = id2label
        self.img_transform = transforms.Compose([
            Resize((512, 512)),
            ToTensor()
        ])
        self.label_transform = transforms.Compose([
            Resize((512, 512), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.img_transform(item['pixel_values'])
        label = torch.tensor(np.array(self.label_transform(item['label'])), dtype=torch.long)
        label = remap_labels(label, self.id2label)
        return image, label


def load_sidewalk_dataset(dataset_path):
    dataset = load_dataset(dataset_path)
    train_val_split = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    filename = "id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id=dataset_path, repo_type="dataset", filename=filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    num_labels = len(id2label)

    train_ds = SidewalkDataset(train_dataset, id2label)
    val_ds = SidewalkDataset(val_dataset, id2label)

    return train_ds, val_ds, num_labels
