
from torch.utils.data import Dataset
from typing import List


class DatasetAggregator(Dataset):
    def __init__(self, datasets: List[Dataset]):
        super(DatasetAggregator, self).__init__()
        self.datasets = datasets
        datasets_to_idxs = [list(zip([idx for _ in range(len(dataset))], range(len(dataset))))
                            for idx, dataset in enumerate(datasets)]
        datasets_to_idxs = [inner_item for item in datasets_to_idxs for inner_item in item]

        print(f'Number of samples in DatasetAggregator {len(datasets_to_idxs)}')
        self.mixed_datasets = datasets_to_idxs

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        return self.datasets[self.mixed_datasets[idx][0]].__getitem__(self.mixed_datasets[idx][1])

