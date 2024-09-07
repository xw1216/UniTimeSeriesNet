import enum
import math

import numpy as np


class SleepType(enum.Enum):
    WAKE = 0
    NREM = 1
    REM = 2


def calc_class_loss_weight(cfg, label):
    n_cls = cfg.dataset.n_class
    labels_count = np.zeros(shape=(n_cls,))

    for c in range(n_cls):
        labels_count[c] = (label == c).sum()

    total = np.sum(labels_count)

    factor = 1 / n_cls
    loss_class_weight = np.array(cfg.model.loss_class_weight)
    mu = np.full(shape=(n_cls,), fill_value=factor) * loss_class_weight
    weight = np.log(mu * total / labels_count.astype(np.float32))
    weight[weight < 1.0] = 1.0
    weight = np.round(weight, 2)

    # mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]
    return weight.tolist()


if __name__ == "__main__":
    print(len(SleepType))
