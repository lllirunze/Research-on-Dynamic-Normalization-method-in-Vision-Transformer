# Research on Dynamic Normalization method in Vision Transformer

---

## Abstract

The dynamic feature normalization method of Transformer model is studied from the aspect of model regularization. It is proposed to use feature normalization instead of traditional layer normalization to achieve explicit Token value normalization and accelerate model convergence. Combined with the idea of parameter reorganization, a dynamic learnable feature normalization is proposed to improve the flexibility and computational efficiency of feature normalization.

---

## Keywords

- Vision Transformer
- Learnable Dynamic Feature Normalization
- Parameter Reorganization

---

## Requirements

```commandline
pip install -r requirements.txt
```

---

## Train model

#### CIFAR-10

```commandline
cd vit
python train.py --num_classes 10 --epochs 300 --batch_size 128 --lr 0.01 --dataset_train_dir "./data/CIFAR10" --dataset_test_dir "./data/CIFAR10" --summary_dir "./summary/vit_base_patch16_224_cifar10" --model 'vit_base_patch16_224_cifar10'
```

#### CIFAR-100

```commandline
cd vit
python train.py --num_classes 100 --epochs 300 --batch_size 128 --lr 0.01 --dataset_train_dir "./data/CIFAR100" --dataset_test_dir "./data/CIFAR100" --summary_dir "./summary/vit_base_patch16_224_cifar100" --model 'vit_base_patch16_224_cifar100'
```

summary_dir can be '/home/sdf/lrz/summary/vit_base_patch16_224_cifar100'

#### ImageNet-1k

```commandline
TODO: Unknown
```

You can modify the config of your command such as epochs, batch size, etc.

---

## Train model with pre-trained weights

The example is as follows.

```commandline
python train.py --num_classes 100 --epochs 300 --batch_size 128 --lr 0.01 --dataset_train_dir "./data/CIFAR100" --dataset_test_dir "./data/CIFAR100" --summary_dir "./summary/vit_base_patch16_224_cifar100" --weights '/home/sdf/lrz/summary/vit_base_patch16_224_cifar100/weights/xxx.pth' --model 'vit_base_patch16_224_cifar100'
```

## Update

### March 26, 2023

After 300 epochs, we got the accuracy of ViT model including training accuracy: 86.99% and test accuracy: 40.00%.

Actually, the accuracy of base model isn't high enough but the accuracy cannot improve with the increase of epoch.

### March 23, 2023: Store weights and bias

The parameters such as weights and bias can be stored in the .pth files. As a result, we can use them to continue our experiment although previous ones are forced termination for the sake of some strange conditions.

### March 22, 2023: First Attempt

I have successfully implemented on lab's machine, which has 3 GPUs.

Unfortunately, I didn't report the 30-epoch-experiment of CIFAR-10 because I accidentally deleted the data.

However, I have reported the 50-epoch-experiment of CIFAR-100. The data is as follows.

```commandline
epoch 49: 100%|███████| 391/391 [04:44<00:00,  1.37it/s, loss=2.0035]
=> train_loss: 2.0278, train_accuracy: 0.4650, test_loss: 2.5127, test_accuracy: 0.3715

```

What's more, the 50-epoch pre-train weights and bias has been stored in the machine. In the future, we can use these existing weights and bias to train more epochs in order to spare time.

### March 21, 2023: Dynamic Token Normalization

Dynamic Token Normalization(DTN) is performed both within each token(intra-token) and across different tokens(inter-token). DTN has several merits.

- DTN is built on a unified formulation and thus can represent various existing normalization methods.
- It learns to normalize tokens in both intra-token and inter-token manners,  enabling Transformers to capture both the global contextual information and the local positional context.
- By simply replacing LN layers, DTN can be readily plugged into
various vision transformers.

### March 19, 2023: Unified Normalization

Unified Normalization(UN) can speed up the inference by being fused with other linear operation and achieve comparable performance on par with LN.

UN consists of **a unified framework** for leveraging offline methods, **a tailored fluctuation smoothing strategy** to mitigate the fluctuations and **an adaptive outlier filtration strategy** for stabilizing training.

### March 18, 2023: Consult

I have reported to Prof. Wang on the progress of Graduation Project and inquired about dynamic learnable normalization function.

Prof. Wang gave me some suggestions and provided me with two papers on normalization function and one paper on parameter reorganization.

Some devices for deep learning will be provided in the future.

### March 8, 2023

##### About Layer Normalization

When input $X \in \mathbb{R}^{L \times B \times C}$ is a batch of a sequence of embeddings (which is applied in **ViT**), where $B$ is the batch size, $C$ is the number of channels, $L$ is the length of the sequence.

$$\text{LN}(X) = \gamma \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}} + \beta$$

where $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.

##### About Feature Normalization

In the algorithm based on gradient descent, the feature normalization method is used to unify the dimension of the feature, which can improve the model convergence speed and the final model accuracy.

###### Min-Max Scaling

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

###### Z-Score Scaling

$$X_{norm} = \frac{X - \mu}{\sigma}$$

where $\mu = \frac{1}{N} \sum_{i=1}^{N} X_{i}$ , $\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N}(X_{i}- \mu)^{2}}$.

### March 6, 2023: Debug

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

### March 5, 2023

'CUDA out of memory' has been solved by reducing batch size from 16 to 4, but there is another serious problem.

The error content is as follows.

```commandline
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
```

According to CSDN, it is because cudNN and CUDA version does not match.

I need to change cudNN or CUDA version so that I can run these code.

### March 2, 2023

ViT has reappeared at this stage, but the code can not run because the memory of GPU is too small.

The error content is as follows.

```commandline
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 58.00 MiB (GPU 0; 4.00 GiB total capacity; 3.34 GiB already allocated; 0 bytes free; 3.41 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

---

## Reference

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [Feature normalization and likelihood-based similarity measures for image retrieval](http://www.cs.bilkent.edu.tr/~saksoy/papers/prletters01_likelihood.pdf)
- [Unified Normalization for Accelerating and Stabilizing Transformers](https://arxiv.org/abs/2208.01313)
- [Dynamic Token Normalization Improves Vision Transformers](https://arxiv.org/abs/2112.02624)
- [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_RepMLPNet_Hierarchical_Vision_MLP_With_Re-Parameterized_Locality_CVPR_2022_paper.html)
