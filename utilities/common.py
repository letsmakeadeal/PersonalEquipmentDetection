from typing import Optional

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


def collate_fn(batch):
    batch_keys = batch[0].keys()
    batch_dict = dict(zip(batch_keys, [None for _ in range(len(batch_keys))]))
    for key in batch_keys:
        if key == 'bboxes':
            batch_dict[key] = [torch.stack([torch.Tensor(inner_sample) for inner_sample in sample[key]])
                               for sample in batch]
        else:
            if type(batch[0][key]).__name__ == 'Tensor':
                batch_dict[key] = \
                    torch.stack([sample[key] for sample in batch])
            else:
                batch_dict[key] = [sample[key] for sample in batch]

    return batch_dict


def seed_everything_deterministic(seed):
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_callback(callbacks) -> Optional[ModelCheckpoint]:
    try:
        return next((c for c in callbacks if type(c) == ModelCheckpoint))
    except StopIteration:
        return None
