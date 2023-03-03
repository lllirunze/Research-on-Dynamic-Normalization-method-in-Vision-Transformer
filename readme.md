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

ViT has reappeared at this stage, but the code can not run because the memory of GPU is too small.

The error content is as follows.

```commandline
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 58.00 MiB (GPU 0; 4.00 GiB total capacity; 3.34 GiB already allocated; 0 bytes free; 3.41 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

As a result, I need to ask Prof. Wang for help.

### Reference

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [Feature normalization and likelihood-based similarity measures for image retrieval](http://www.cs.bilkent.edu.tr/~saksoy/papers/prletters01_likelihood.pdf)
