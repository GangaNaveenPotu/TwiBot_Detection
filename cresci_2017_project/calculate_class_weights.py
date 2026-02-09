
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
import torch

df = pd.read_csv('D:/Capstone/cresci-2017/processed_data.csv')
labels = df['label'].values

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print(f"Class weights: {class_weights_tensor.tolist()}")
