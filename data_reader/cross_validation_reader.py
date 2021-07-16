import torch
from random import seed
import numpy as np
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Dataset

from utils.utils import get_graph_diameter

rnd_state = np.random.RandomState(seed(1))

def split_ids(ids, folds=10):
    """
    Function that returns train, test and validation splits ids
    """
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]

    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    valid_ids = []
    train_ids = []

    for fold in range(folds):
        valid_fold = []
        while len(valid_fold) < stride:
            id = random.choice(ids)
            if id not in test_ids[fold] and id not in valid_fold:
               valid_fold.append(id)

        valid_ids.append(np.asarray(valid_fold))
        train_ids.append(np.array([e for e in ids if e not in test_ids[fold] and e not in valid_ids[fold]]))
        assert len(train_ids[fold]) + len(test_ids[fold]) + len(valid_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]) + list(valid_ids[fold]))) == n, 'invalid splits'


    return train_ids, test_ids, valid_ids


def getcross_validation_split(dataset_path='~/storage/Dataset/MUTAG', dataset_name='MUTAG', n_folds=2, batch_size=1, use_node_attr=True): # edited: now it's using node attributes

    dataset = TUDataset(root=dataset_path, name=dataset_name, pre_transform=get_graph_diameter, use_node_attr=use_node_attr)
    train_ids, test_ids, valid_ids = split_ids(rnd_state.permutation(len(dataset)), folds=n_folds)
    splits=[]

    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:
            # print(torch.from_numpy((train_ids if split.find('train') >= 0 else test_ids)[fold_id]))
            # print("---")
            gdata = dataset[torch.tensor(split[fold_id], dtype=torch.long)]

            loader = DataLoader(gdata,
                                batch_size=batch_size,
                                shuffle=False, # True to have the data reshuffled at every epoch, hinders incremental learning
                                pin_memory=True,
                                #persistent_workers=True,
                                num_workers=0)#, TODO LINUX=Set this to 4; https://github.com/pytorch/pytorch/issues/12831 and https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
            loaders.append(loader)
        splits.append(loaders)
        # print("---")

    return splits #0-train, 1-test, 2-valid

class MyDataset(Dataset):
    def __init__(self, root, name, pre_transform, use_node_attr):
        self.TUDataset = TUDataset(root=root, name=name, pre_transform=pre_transform, use_node_attr=use_node_attr)

    def __getitem__(self, index):
        data, target  = self.TUDataset[index]
        return data, target, index

    def __len__(self):
        return len(self.TUDataset)

if __name__ == '__main__':
    cv_splits= getcross_validation_split(n_folds=10)

    split_1=cv_splits[2]
    train=split_1[0]
    test=split_1[1]
    valid=split_1[2]
    for batch in train:
        print(batch.diameter)
    print(len(train))
    print(len(test))
    print(len(valid))