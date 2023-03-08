## Research on Dynamic Normalization method in Vision Transformer

### Abstract

The dynamic feature normalization method of Transformer model is studied from the aspect of model regularization. It is proposed to use feature normalization instead of traditional layer normalization to achieve explicit Token value normalization and accelerate model convergence. Combined with the idea of parameter reorganization, a dynamic learnable feature normalization is proposed to improve the flexibility and computational efficiency of feature normalization.

### Keywords

- Transformer
- Feature Normalization
- Parameter Reorganization

### Requirements

```commandline
pip install -r requirements.txt
```

### Progress

#### March 2, 2023

ViT has reappeared at this stage, but the code can not run because the memory of GPU is too small.

The error content is as follows.

```commandline
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 58.00 MiB (GPU 0; 4.00 GiB total capacity; 3.34 GiB already allocated; 0 bytes free; 3.41 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

#### March 5, 2023

'CUDA out of memory' has been solved by reducing batch size from 16 to 4, but there is another serious problem.

The error content is as follows.

```commandline
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
```

According to CSDN, it is because cudNN and CUDA version does not match.

I need to change cudNN or CUDA version so that I can run these code.

#### March 6, 2023

There is a ridiculous bug: number of classes = 5.

Actually, the number of classes of CIFAR-10 is 10.

After debugging, I can run ViT, and the result is as follows.

```commandline
epoch 0: 100%|██████████| 12500/12500 [1:48:44<00:00,  1.92it/s, loss=2.3856]
=> train_loss: 1.9306, train_accuracy: 0.2730, test_loss: 1.7810, test_accuracy: 0.3433
```

The GPU of my computer(ThinkPad X1 Extreme) is RTX 1650-Ti, so the speed of code running is extremely slow.
Specifically, each epoch needs to run for about one hour.

I need to find one or more better GPU in order to produce results faster than before.

#### March 8, 2023

##### About Layer Normalization:

When input $X \in \mathbb{R}^{B \times C}$ is a batch of embeddings,
    where $B$ is the batch size and $C$ is the number of features.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

When input $X \in \mathbb{R}^{L \times B \times C}$ is a batch of a sequence of embeddings (which is applied in **ViT**),
    where $B$ is the batch size, $C$ is the number of channels, $L$ is the length of the sequence.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations,
    where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width.
    This is not a widely used scenario.
    $\gamma \in \mathbb{R}^{C \times H \times W}$ and $\beta \in \mathbb{R}^{C \times H \times W}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C, H, W}{\mathbb{E}}[X]}{\sqrt{\underset{C, H, W}{Var}[X] + \epsilon}}
    + \beta$$


### Reference

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [Feature normalization and likelihood-based similarity measures for image retrieval](http://www.cs.bilkent.edu.tr/~saksoy/papers/prletters01_likelihood.pdf)
