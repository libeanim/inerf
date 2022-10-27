import os
import json
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

class PoseDataset(Dataset):

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
    
        with open(os.path.join(self.root, 'transforms.json'), 'r') as f:
            data = json.load(f)
        
        self.camera_angle_x = data['camera_angle_x']

        frames = data['frames']
        self.samples = []
        for frame in frames:
            filepath = os.path.join(root, f"{frame['file_path']}.png")
            transform_matrix = torch.Tensor(frame['transform_matrix'])
            self.samples.append((filepath, transform_matrix))

    def loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class InMemoryDataset(Dataset):

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __getitem__(self, index: Any) -> Tuple[torch.Tensor, int]:
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def create_dataset_file(dataset: Dataset, path: str):
        from tqdm import tqdm
        res = {}
        for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Loading images"):
            res[idx] = data
        with open(path, 'wb') as f:
            pickle.dump(res, f)
