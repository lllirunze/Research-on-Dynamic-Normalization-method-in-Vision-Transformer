import torch

def MinMaxScaling(feature: torch.Tensor, eps=1.0):
    # Calculate the min and max value of the last dimension
    f_min = torch.min(feature, dim=-1, keepdim=True).values
    f_max = torch.max(feature, dim=-1, keepdim=True).values
    # norm = (X - minX) / (maxX - minX)
    feature = (feature - f_min) / (f_max - f_min + eps)
    return feature

def ZScoreScaling(feature: torch.Tensor, eps=1.0):
    # Calculate the average value and variance of the last dimension
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
    mean = var_mean[1]
    var = var_mean[0]
    # norm = (X - meanX) / (varX)
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    return feature
