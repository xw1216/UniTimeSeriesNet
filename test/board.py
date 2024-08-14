import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=os.path.join(os.pardir, 'runs'))

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

