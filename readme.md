# Research on Dynamic Normalization method in Vision Transformer

---

## Abstract

This article studies the normalization methods of Vision Transformer and proposes a dynamic learnable normalization method (DTN) to replace the conventional layer normalization, achieving token feature normalization and accelerating the convergence speed of the model. In order to achieve dynamic and learnable effects, this article introduces dynamic learnable normalization parameters, which can enable DTN to use LN and IN simultaneously in the same calculation formula. This parameter can play different roles in network training for different attention heads in DTN, thereby improving the performance of the Vision Transformer and applying it to various tasks in the field of computer vision. DTN can focus on different aspects of the input sequence through multiple attention heads, which can provide the model with different interpretations of the input sequence from different perspectives, thereby facilitating the interpretability of the model and ultimately enabling it to better capture the information of the input sequence.

This article applies the proposed method to different Vision Transformer and conducts image classification experiments with other normalization methods on CIFAR-10 and CIFAR-100 datasets and compares and analyzes the experimental results. Experiments have shown that DTN has better experimental results. Compared with the same type of normalization method, your dynamic normalization method based on the Vision Transformer can better achieve model convergence ability and improve model accuracy.

---

## Keywords

- Vision Transformer
- Computer Vision
- Dynamic Token Normalization
- Attention Mechanism

---

## Requirements

```commandline
pip install -r requirements.txt
```

---

## Train model

#### CIFAR-10 & Vit-S

```commandline
python train.py --num_classes 10 --model "vit-s" --data "cifar10" --summary_dir "./summary/vit_small_cifar10"
```

#### CIFAR-100 & Vit-S

```commandline
python train.py --num_classes 100 --model "vit-s" --data "cifar100" --summary_dir "./summary/vit_small_cifar100"
```

#### ImageNet-1k & Vit-S

```commandline
python train.py --num_classes 1000 --model "vit-s" --data "imagenet1k" --summary_dir "./summary/vit_small_imagenet1k"
```

You can modify the config of your command such as epochs, batch size, etc.

---

## Train model with pre-trained weights

summary_dir can be '/home/sdf/lrz/summary/vit_small_patch16_224_cifar100'

The example is as follows.

```commandline
python train.py --num_classes 10 --model "vit-s" --data "cifar10" --summary_dir "./summary/vit_small_cifar10" --weights "/home/sdf/lrz/summary/vit_small_patch16_224_cifar10"
```

## Result

- Dataset: __CIFAR-10__
- batch-size: 128
- epochs: 1200

| model | normalization | Top-1 acc | Top-5 acc |
| :--: | :--: | :--: | :--: |
|ViT-S|LN|91.88|99.67|
|ViT-S|BN|91.63|99.72|
|ViT-S|UN|89.52|99.68|
|ViT-S|DTN| __92.96__ | __99.85__ |
|T2T-ViT-S|LN| __93.14__ | __99.89__ |
|T2T-ViT-S|BN|92.29|99.68|
|T2T-ViT-S|UN|90.56|99.76|
|T2T-ViT-S|DTN|92.98| __99.89__ |

- Dataset: __CIFAR-100__
- batch-size: 128
- epochs: 1200

| model | normalization | Top-1 acc | Top-5 acc |
| :--: | :--: | :--: | :--: |
|ViT-S|LN| __71.75__ | __91.68__ |
|ViT-S|BN|68.53|89.72|
|ViT-S|UN|66.07|89.10|
|ViT-S|DTN|68.24|90.67|
|T2T-ViT-S|LN|71.61|91.61|
|T2T-ViT-S|BN|68.96|89.65|
|T2T-ViT-S|UN|73.03|92.31|
|T2T-ViT-S|DTN| __75.59__ | __93.63__ |

## Update

### May 24, 2023:

The thesis has been finished.

### May 23, 2023:

All the experiments have been implemented.

### April 26, 2023: 

Since April 1st, I have started my experiment. Now the process has been in the middle, and the result of experiment is in the __'Result'__ module.

### March 29, 2023: Code refactoring

All the file has been refactored because of terrible results. The results of CIFAR-10 and CIFAR-100 has been closed to original paper after code refactoring.

For the sake of weak computing power, I choose vit-small as test model, which can get results faster.

###### Reference code:

- https://github.com/lucidrains/vit-pytorch
- https://github.com/DeepVoltaire/AutoAugment


### March 27, 2023: Debug

The 'dataloader' file has serious bug and I have debugged.

Now, the 'getData' function has been divided into two functions named 'getTrainData' and 'getTestData'.

Theoretically, the accuracy of training and testing will increase after 1200 epochs.

What's more, I have added the function which can load ImageNet dataset. After the experiment of CIFAR-100, we can experiment for ImageNet.

### March 26, 2023

After 300 epochs, we got the accuracy of ViT model including training accuracy: 86.99% and test accuracy: 40.00%.

Actually, the result is too bad.

The accuracy of base model isn't high enough but the accuracy cannot improve with the increase of epoch.

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
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Unified Normalization for Accelerating and Stabilizing Transformers](https://arxiv.org/abs/2208.01313)
- [Dynamic Token Normalization Improves Vision Transformers](https://arxiv.org/abs/2112.02624)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
