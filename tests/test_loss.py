"""
测试结果，对于分类：mask是多余的，因为BCE Loss本身就会判断preds/labels是否在0-1内，如果有nan/inf就会报错。
"""
# Test the function
import torch, numpy as np, sys, os
sys.path.append('/dev_data/wlh/stock-baselines')
from basicts.losses.losses import masked_binary_cross_entropy

sys.path.append(os.path.abspath(__file__ + "/../.."))

#preds = torch.tensor([[0.9, 0.4, 0.7], [0.2, 0.8, np.nan]], dtype=torch.float32)
#labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, np.nan]], dtype=torch.float32)

#preds = torch.tensor([[0.9, 0.4, 0.7], [0.2, 0.8, 0.5]], dtype=torch.float32)  # 注意这里替换了 np.nan 为 0.5
#labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)  # 注意这里替换了 np.nan 为 1.0

preds = torch.tensor([[0.9, 0.4, 0.7], [0.2, 0.8, 0.5]], dtype=torch.float32)  # 注意这里替换了 np.nan 为 0.5
labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)  # 注意这里替换了 np.nan 为 1.0


loss = masked_binary_cross_entropy(preds, labels)
print(f"Masked Binary Cross Entropy Loss: {loss.item()}")