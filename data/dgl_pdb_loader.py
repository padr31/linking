from itertools import accumulate

import numpy as np
import torch
from dgl.data.utils import Subset
from dgllife.data import PDBBind
from dgllife.utils import (RandomSplitter, ScaffoldSplitter,
                           SingleTaskStratifiedSplitter)

"""Load the dataset.
    Parameters
    ----------
    args = {
        subset: 'core',
        load_binding_pocket: True,
        split: 'random',
        frac_train: 0.8,
        frac_val: 0.1,
        frac_test: 0.1,
    }
    Returns
    -------
    dataset
        Full dataset.
    train_set
        Train subset of the dataset.
    val_set
        Validation subset of the dataset.
    """


def load_dataset(args):
    dataset = PDBBind(
        subset=args["subset"],
        load_binding_pocket=args["load_binding_pocket"],
        zero_padding=True,
    )

    # No validation set is used and frac_val = 0.
    if args["split"] == "random":
        train_set, _, test_set = RandomSplitter.train_val_test_split(
            dataset,
            frac_train=args["frac_train"],
            frac_val=args["frac_val"],
            frac_test=args["frac_test"],
        )

    elif args["split"] == "scaffold":
        train_set, _, test_set = ScaffoldSplitter.train_val_test_split(
            dataset,
            mols=dataset.ligand_mols,
            sanitize=False,
            frac_train=args["frac_train"],
            frac_val=args["frac_val"],
            frac_test=args["frac_test"],
        )

    elif args["split"] == "stratified":
        train_set, _, test_set = SingleTaskStratifiedSplitter.train_val_test_split(
            dataset,
            labels=dataset.labels,
            task_id=0,
            frac_train=args["frac_train"],
            frac_val=args["frac_val"],
            frac_test=args["frac_test"],
        )

    elif args["split"] == "temporal":
        years = dataset.df["release_year"].values.astype(np.float32)
        indices = np.argsort(years).tolist()
        frac_list = np.array([args["frac_train"], args["frac_val"], args["frac_test"]])
        num_data = len(dataset)
        lengths = (num_data * frac_list).astype(int)
        lengths[-1] = num_data - np.sum(lengths[:-1])
        train_set, val_set, test_set = [
            Subset(dataset, list(indices[offset - length : offset]))
            for offset, length in zip(accumulate(lengths), lengths)
        ]

    else:
        raise ValueError(
            "Expect the splitting method "
            'to be "random" or "scaffold", got {}'.format(args["split"])
        )

    train_labels = torch.stack([train_set.dataset.labels[i] for i in train_set.indices])
    train_set.labels_mean = train_labels.mean(dim=0)
    train_set.labels_std = train_labels.std(dim=0)

    return dataset, train_set, test_set


args = {
    "subset": "core",
    "load_binding_pocket": True,
    "split": "random",
    "frac_train": 0.8,
    "frac_val": 0.1,
    "frac_test": 0.1,
}

x = load_dataset(args)
print()
