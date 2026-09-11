import numpy as np
import torch
from torch.utils.data import Dataset

class EpisodeFrames(Dataset):
    def __init__(self, dataset_dir, manifest, split):
        chunks = []

        for record in manifest["episodes"]:
            if record["split"] != split:
                continue

            with np.load(dataset_dir / record["file"]) as episode:
                frames = episode["observations"]
                assert frames.dtype == np.uint8
                assert frames.shape[1:] == (64, 64, 3)
                chunks.append(frames.copy())

        if not chunks:
            raise ValueError(f"No episodes for split: {split}")

        self.frames = np.concatenate(chunks, axis=0)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        # Saved images are already preprocessed.
        return (
            torch.from_numpy(self.frames[index])
            .permute(2, 0, 1)
            .float()
            / 255.0
        )


train_data = EpisodeFrames(dataset_dir, manifest, "train")
val_data = EpisodeFrames(dataset_dir, manifest, "validation")

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(0),
)
val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

print("Training frames:", len(train_data))
print("Validation frames:", len(val_data))
