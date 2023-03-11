import torch

def MinMaxScaling(feature: torch.Tensor, eps=1.0):
    # Calculate the min and max value of the last dimension
    f_min = torch.min(feature, dim=-1, keepdim=True).values
    f_max = torch.max(feature, dim=-1, keepdim=True).values
    # norm = (X - minX) / (maxX - minX)
    feature = (feature - f_min) / (f_max - f_min + eps)
    return feature