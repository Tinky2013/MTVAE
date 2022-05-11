import pandas as pd
import numpy as np
import torch

dt = pd.read_csv('data/usernet_train.csv')

print(torch.tensor(np.array(dt)))
print(torch.tensor(np.array(dt)).shape)