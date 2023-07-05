"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import os
import csv
import torch
import numpy as np


class DataManagerFile(object):
    """
    A data loader, it can load data, split in train/test, shuffle, normalize, etc.
    NOTE: the 0-th regime should always be the observational one
    """

    def __init__(
        self,
        data,
        masks,
        regimes,
        train_samples=0.8,
        test_samples=None,
        train=True,
        normalize=False,
        mean=None,
        std=None,
        random_seed=42,
        intervention=False,
        intervention_knowledge="known",
        dcd=False,
        regimes_to_ignore=None,
    ):
        """
        :param str file_path: Path to the data and the DAG
        :param int i_dataset: Exemplar to use (usually in [1,10])
        :param float/int train_samples: default=0.8. If float, specifies the proportion of
            data used for training and the rest is used for testing. If an integer, specifies
            the exact number of examples to use for training.
        :param int test_samples: default=None. Specifies the number of examples to use for testing.
            The default value uses all examples that are not used for training.
        :param int random_seed: Random seed to use for data set shuffling and splitting
        :param boolean intervention: If True, use interventional data with interventional targets
        :param str intervention_knowledge: Determine if the intervention target are known or unknown
        :param boolean dcd: If True, use the baseline DCD that use interventional data, but
            with a loss that doesn't take it into account (intervention should be set to False)
        :param list regimes_to_ignore: Regimes that are ignored during training
        """
        self.random = np.random.RandomState(random_seed)
        self.dcd = dcd
        self.intervention = intervention
        if intervention_knowledge == "known":
            self.interv_known = True
        elif intervention_knowledge == "unknown":
            self.interv_known = False
        else:
            raise ValueError(
                "intervention_knowledge should either be 'known' \
                             or 'unknown'"
            )

        # index of all regimes, even if not used in the regimes_to_ignore case
        self.all_regimes = np.unique(regimes)

        # Remove some regimes
        if regimes_to_ignore is not None and self.intervention:
            for regime_to_ignore in regimes_to_ignore:
                if regime_to_ignore not in self.all_regimes:
                    raise ValueError(
                        f"Regime {regime_to_ignore} is not in the possible regimes: {self.all_regimes}"
                    )
                to_keep = np.array(regimes) != regime_to_ignore
                data = data[to_keep]
                masks = [mask for i, mask in enumerate(masks) if to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(regimes) if to_keep[i]]
                )

        # Determine train/test partitioning
        if isinstance(train_samples, float):
            train_samples = int(data.shape[0] * train_samples)
        if test_samples is None:
            test_samples = data.shape[0] - train_samples
        assert train_samples + test_samples <= data.shape[0], (
            "The number of examples to load must be "
            + "smaller than the total size of the dataset"
        )

        # Shuffle and filter examples
        shuffle_idx = np.arange(data.shape[0])
        self.random.shuffle(shuffle_idx)
        data = data[shuffle_idx[: train_samples + test_samples]]
        if intervention:
            masks = [masks[i] for i in shuffle_idx[: train_samples + test_samples]]
        regimes = regimes[shuffle_idx[: train_samples + test_samples]]

        # Train/test split
        if not train:
            if train_samples == data.shape[0]:  # i.e. no test set
                self.dataset = None
                self.masks = None
                self.regimes = None
            else:
                self.dataset = torch.as_tensor(
                    data[train_samples : train_samples + test_samples]
                ).type(torch.Tensor)
                if intervention:
                    self.masks = masks[train_samples : train_samples + test_samples]
                self.regimes = regimes[train_samples : train_samples + test_samples]
        else:
            self.dataset = torch.as_tensor(data[:train_samples]).type(torch.Tensor)
            if intervention:
                self.masks = masks[:train_samples]
            self.regimes = regimes[:train_samples]

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if self.mean is None or self.std is None:
                self.mean = torch.mean(self.dataset, 0, keepdim=True)
                self.std = torch.std(self.dataset, 0, keepdim=True)
            self.dataset = (self.dataset - self.mean) / self.std

        self.num_regimes = np.unique(self.regimes).shape[0]
        self.num_samples = self.dataset.size(0)
        self.dim = self.dataset.size(1)

    def convert_masks(self, idxs):
        """
        Convert mask index to mask vectors
        :param np.ndarray idxs: indices of mask to convert
        :return: masks
        Example:
            if self.masks[i] = [1,4]
               self.dim = 10 then
            masks[i] = [1,0,1,1,0,1,1,1,1,1]
        """
        masks_list = [self.masks[i] for i in idxs]

        masks = torch.ones((idxs.shape[0], self.dim))
        for i, m in enumerate(masks_list):
            for j in m:
                masks[i, j] = 0

        return masks

    def sample(self, batch_size):
        """
        Sample without replacement `batch_size` examples from the data and
        return the corresponding masks and regimes
        :param int batch_size: number of samples to sample
        :return: samples, masks, regimes
        """
        sample_idxs = self.random.choice(
            np.arange(int(self.num_samples)), size=(int(batch_size),), replace=False
        )
        samples = self.dataset[torch.as_tensor(sample_idxs).long()]
        if self.intervention:
            masks = self.convert_masks(sample_idxs)
            regimes = torch.as_tensor(self.regimes).long()
            regimes = regimes[torch.as_tensor(sample_idxs).long()]
        else:
            masks = torch.ones_like(samples)
            regimes = None
        return samples, masks, regimes
