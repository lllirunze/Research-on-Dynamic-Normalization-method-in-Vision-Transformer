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

When input $X \in \mathbb{R}^{L \times B \times C}$ is a batch of a sequence of embeddings (which is applied in **ViT**), where $B$ is the batch size, $C$ is the number of channels, $L$ is the length of the sequence.
    
$\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    
$$\text{LN}(X) = \gamma \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}} + \beta$$

##### About Feature Normalization:

In the algorithm based on gradient descent, the feature normalization method is used to unify the dimension of the feature, which can improve the model convergence speed and the final model accuracy.

###### Min-Max Scaling

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

###### Z-Score Scaling

$$X_{norm} = \frac{X - \mu}{\sigma}$$

where $\mu = \frac{1}{N} \sum_{i=1}^{N} X_{i}$ , $\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N}(X_{i}- \mu)^{2}}$ .

#### March 11, 2023

Finish Min-Max Scaling of Pytorch, whose test has not been carried out yet.

#### March 12, 2023

Finish Z-Score Scaling of Pytorch, whose test hasn't been carried out yet.

#### March 18, 2023

I have reported to Prof. Wang on the progress of Graduation Project and inquired about dynamic learnable normalization function.

Prof. Wang gave me some suggestions and provided me with two papers on normalization function and one paper on parameter reorganization.

Some devices for deep learning will be provided in the future.

#### March 19, 2023 Unified Normalization:

Unified Normalization(UN) can speed up the inference by being fused with other linear operation and achieve comparable performance on par with LN.

UN consists of **a unified framework** for leveraging offline methods, **a tailored fluctuation smoothing strategy** to mitigate the fluctuations and **an adaptive outlier filtration strategy** for stabilizing training.

##### 1 Unified Framework

###### Forward Propagation:

$$ Z_{t} = \frac{X_{t}- \hat{\mu_{t}}}{\sqrt{\hat{\sigma_{t}}^{2} + \epsilon}} $$

$$ Y_{t} = \gamma \cdot Z_{t} + \beta $$

Let $ Z_{t} $ denote the normalized alternative to input $ X_{t} $ at iteration $ t $. The training statistics for normalizing are marked as $ \hat{\mu_{t}} $ and $ \hat{\sigma_{t}}^{2} $, given by

$$ \hat{\mu_{t}} = \Theta_{\mu}(\mu_{t}, \cdots, \mu_{t-M+1}) $$

$$ \hat{\sigma_{t}}^{2} = \Theta_{\sigma^{2}}(\sigma_{t}^{2}, \cdots, \sigma_{t-M+1}^{2}) $$

###### Backward Propagation:

The gradients of loss $ L $ pass as:

$$ \frac{\partial L}{\partial Z_{t}} = \gamma \cdot \frac{\partial L}{\partial Y_{t}} $$

$$ \frac{\partial L}{\partial X_{t}} = \frac{1}{\sqrt{\hat{\sigma_{t}}^{2} + \epsilon}} (\frac{\partial L}{\partial Z_{t}} - \psi_{\hat{\mu_{t}}} - Z_{t} \cdot \psi_{\hat{\sigma_t}^{2}}) $$

Giving gradients $ \frac{\partial L}{\partial Y_{t}} $, $ \psi_{\hat{\mu_{t}}} $ and $ \psi_{\hat{\sigma_t}^{2}} $ indicate the gradient statistics that used for estimating $ \frac{\partial L}{\partial X_{t}} $. In this framework, estimated gradients are gained from averaging functions $ \Theta_{g_{\mu}} $ and $ \Theta_{g_{\sigma^{2}}} $,

$$ \psi_{\hat{\mu_{t}}} = \Theta_{g_{\mu}}(g_{\hat{\mu_{t}}}, \cdots, g_{\hat{\mu}_{t-M+1}}) $$

$$ \psi_{\hat{\sigma_{t}}^{2}} = \Theta_{g_{\sigma^{2}}}(g_{\hat{\sigma_{t}}^{2}}, \cdots, g_{\hat{\sigma}_{t-M+1}^{2}}) $$

The gradients passed from $\hat{\mu_{t}}$ and $\hat{\sigma_{t}}^{2}$ are denoted as $g_{\hat{\mu_t}}$ and $g_{\hat{\sigma_{t}}^{2}}$.

##### 2 Fluctuation Smoothing

We turn ot adopt geometric mean(GM) with less sensitivity to outliers instead of arithmetic mean(AM) to gain a better representation of activation statistics in a skewed distribution. 

The averaging functions are defined as:

$$\hat{\mu_{t}} = 0,\quad \hat{\sigma_{t}}^{2} = \sqrt[M]{\prod_{i=0}^{M-1} \sigma_{t-i}^{2}}$$

$$\psi_{\hat{\mu_{t}}} = 0, \quad \psi_{\hat{\sigma}_{t}^{2}} = \alpha \psi_{\hat{\sigma}_{t-1}^{2}} + (1- \alpha)\frac{1}{M} \sum_{i=0}^{M-1} g_{\hat{\sigma}_{t-1}^{2}}$$

##### 3 Outlier Filtration

The main goal of outlier filtration is to decide when to apply the moving average strategies.

To identify outliers, we set an adaptive threshold for outlier filtration with the $AM - GM$ *inequality*.

Let $\Omega_{t} = (\sigma_{t}^{2}, \sigma_{t-1}^{2}, \cdots, \sigma_{t-M+1}^{2})$ denote the $M$ recent activation statistics recorded in forward propagation at iteration $t$, where $M>1$, then we have

$$E(\Omega_{t}) - \Pi(\Omega_{t}) \le M \cdot V(\Omega_{t}^{\frac{1}{2}})$$

where $\Omega_{t}^{\frac{1}{2}} = (\sigma_{t}, \sigma_{t-1}, \cdots, \sigma_{t-M+1})$ and $V(\cdot)$, $E(\cdot)$, $\Pi(\cdot)$ are operators that calculate the variance, arithmetic mean, and geometric mean for input respectively.

Once the mini-batch is deemed to contain extremely large outliers and all the moving average strategies will be dropped in a specific normalization layer.

If $E(\Omega_{t}) - \Pi(\Omega_{t}) > M \cdot V(\Omega_{t}^{\frac{1}{2}})$,

$$\hat{\sigma}_{t}^{2} = \sigma_{t}^{2}, \quad \psi_{\hat{\sigma}_{t}^{2}} = g_{\hat{\sigma}_{t}^{2}}$$

else,

$$\hat{\sigma_{t}}^{2} = \sqrt[M]{\prod_{i=0}^{M-1} \sigma_{t-i}^{2}},\quad \psi_{\hat{\sigma}_{t}^{2}} = \alpha \psi_{\hat{\sigma}_{t-1}^{2}} + (1- \alpha)\frac{1}{M} \sum_{i=0}^{M-1} g_{\hat{\sigma}_{t-1}^{2}}$$
### Reference

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [Feature normalization and likelihood-based similarity measures for image retrieval](http://www.cs.bilkent.edu.tr/~saksoy/papers/prletters01_likelihood.pdf)
- [Unified Normalization for Accelerating and Stabilizing Transformers](https://arxiv.org/abs/2208.01313)
- [Dynamic Token Normalization Improves Vision Transformers](https://arxiv.org/abs/2112.02624)
- [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_RepMLPNet_Hierarchical_Vision_MLP_With_Re-Parameterized_Locality_CVPR_2022_paper.html)
