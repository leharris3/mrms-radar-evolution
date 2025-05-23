import torch
from torch.utils.data import Dataset
from typing import Tuple


class TornadoReflectivityDataset(Dataset):
    """
    Toy version – returns random reflectivity cubes.

    Shapes (match the real dataset):
        source : [15, 224, 224]   # 15 frames before t₀
        target : [ 3, 224, 224]   #  3 frames after t₀
    Values are sampled from U(0, 1], so they are already “normalized”.
    """

    def __init__(
        self,
        split: str = "train",
        num_events: int = 10,  # total synthetic events
        train_split: float = 0.85,
        img_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
    ):
        super().__init__()
        assert split in {"train", "val"}, f"Invalid split '{split}'"
        self.split = split
        self.H, self.W = img_size

        # reproducible randomness
        g = torch.Generator().manual_seed(seed)

        # work out how many samples belong to the requested split
        n_train = int(num_events * train_split)
        self.length = n_train if split == "train" else num_events - n_train

        # pre-create the random tensors once; cheapest & simplest
        self.sources = torch.rand((self.length, 15, self.H, self.W), generator=g)
        self.targets = torch.rand((self.length, 3, self.H, self.W), generator=g)

    # ------------------------------------------------------------------ #
    # PyTorch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return {
            "source": self.sources[idx][0, :, :],  # tensor, shape [224, 224]
            "target": self.targets[idx][0, :, :],  # tensor, shape [224, 224]
        }


# ---------------------------------------------------------------------- #
# Quick smoke-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    train_ds = TornadoReflectivityDataset(split="train")
    print("Train samples:", len(train_ds))
    sample = train_ds[0]
    print(
        "source shape:",
        sample["source"].shape,
        "| min/max:",
        sample["source"].min().item(),
        sample["source"].max().item(),
    )
