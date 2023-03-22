# Dynamic Token Normalization

Dynamic Token Normalization(DTN) is performed both within each token(intra-token) and across different tokens(inter-token). DTN has several merits.

- DTN is built on a unified formulation and thus can represent various existing normalization methods.
- It learns to normalize tokens in both intra-token and inter-token manners,  enabling Transformers to capture both the global contextual information and the local positional context.
- By simply replacing LN layers, DTN can be readily plugged into
various vision transformers.

## 1 Definition

Given the feature of tokens $\boldsymbol{x} \in \mathbb{R}^{T \times C}$, DTN normalizes it through

$$\tilde{\boldsymbol{x}} = \gamma \frac{\boldsymbol{x} - Concate_{h \in [H]} \{ \pmb{\mu}^{h} \}}{\sqrt{Concate_{h \in [H]} \{ (\pmb{\sigma}^{2})^{h} \} + \epsilon}} + \beta$$

where $\pmb{\gamma}$, $\pmb{\beta}$ are two $C \times 1$ vectors by stacking all $\gamma_{c}$ and $\beta_{c}$ into a column, and $\pmb{\mu}^{h} \in \mathbb{R}^{T \times \frac{C}{H}}$, $(\pmb{\sigma}^{2})^{h} \in \mathbb{R}^{T \times \frac{C}{H}}$ are normalization constants of DTN in head $h$ where $H$ denotes the number of heads in transformer.

The $Concate$ notation indicates that DTN concatenates normalization constants from different heads.

## 2 Normalization constants

DTN obtains normalization constants by trading off intra- and inter-token statistics as given by

$$\pmb{\mu}^{h} = \lambda^{h}(\pmb{\mu}^{ln})^{h} + (1-\lambda^{h})\boldsymbol{P}^{h}\boldsymbol{x}^{h}$$

$$(\pmb{\sigma}^{2})^{h} = \lambda^{h}((\pmb{\sigma}^{2})^{ln})^{h} + (1-\lambda^{h})[\boldsymbol{P}^{h}(\boldsymbol{x}^{h} \odot \boldsymbol{x}^{h}) - (\boldsymbol{P}^{h}\boldsymbol{x}^{h} \odot \boldsymbol{P}^{h}\boldsymbol{x}^{h})]$$

where $(\pmb{\mu}^{ln})^{h} \in \mathbb{R}^{T \times \frac{C}{H}}$, $((\pmb{\sigma}^{2})^{ln})^{h} \in \mathbb{R}^{T \times \frac{C}{H}}$ are intra-token mean and variance obtained by stacking all $\mu_{t}^{ln}$ and $(\sigma^{2})_{t}^{ln}$ as given by

$$\mu_{t}^{ln} = \frac{1}{C}\sum_{c=1}^{C}x_{tc}$$

$$(\sigma^{2})_{t}^{ln} = \frac{1}{C}\sum_{c=1}^{C}(x_{tc} - \mu_{t}^{ln})^{2}$$

and then broadcasting it for $\frac{C}{H}$ columns, $\boldsymbol{x}^{h} \in \mathbb{R}^{T \times \frac{C}{H}}$ represents token embeddings in the head $h$ of $\boldsymbol{x}$, and $\boldsymbol{P}^{h}\boldsymbol{x}^{h} \in \mathbb{R}^{T \times \frac{C}{H}}$, $[\boldsymbol{P}^{h}(\boldsymbol{x}^{h} \odot \boldsymbol{x}^{h}) - (\boldsymbol{P}^{h}\boldsymbol{x}^{h} \odot \boldsymbol{P}^{h}\boldsymbol{x}^{h})] \in \mathbb{R}^{T \times \frac{C}{H}}$ are expected to represent inter-token mean and variance respectively. Toward this goal, we define $\boldsymbol{P}^{h}$ as a $T \times T$ learnable matrix satisfying that the sum of each row equals 1. Moreover, DTN utilizes a learnable weight ratio $\lambda^{h} \in [0, 1]$ to trade off intra-token and inter-token statistics.

## 3 Representation Capacity

When $\lambda^{h} = 1$, we have $\pmb{\mu}^{h} = (\pmb{\mu}^{ln})^{h}$ and $(\pmb{\sigma}^{2})^{h} = ((\pmb{\sigma}^{2})^{ln})^{h}$. Hence, DTN degrades into LN.

When $\lambda^{h} = 0$ and $\boldsymbol{P}^{h} = \frac{1}{T} \mathbf{1}$, we have $\pmb{\mu}^{h} = \frac{1}{T} \mathbf{1} \boldsymbol{x}^{h}$ and $(\pmb{\sigma}^{2})^{h} = \frac{1}{T} \mathbf{1} (\boldsymbol{x}^{h} \odot \boldsymbol{x}^{h}) - (\frac{1}{T} \mathbf{1} \boldsymbol{x}^{h} \odot \frac{1}{T} \mathbf{1} \boldsymbol{x}^{h})$. Therefore, DTN becomes IN in this case.

## 4 Construction of $P^{h}$

We employ positional self-attention with relative positional embedding to generate positional attention matrix $\boldsymbol{P}^{h}$ as given by

$$\boldsymbol{P}^{h} = softmax(\boldsymbol{R}\boldsymbol{a}^{h})$$

where $\boldsymbol{R} \in \mathbb{R}^{T \times T \times 3}$ is a constant tensor representing the relative positional embedding. To embed the relative position between image patches, we instantiate $\boldsymbol{R}_{ij}$ as written by 

$$\boldsymbol{R}_{ij} = [(\delta_{ij}^{x})^{2}+(\delta_{ij}^{y})^{2}, \delta_{ij}^{x},\delta_{ij}^{y}]^{T}$$

where $\delta_{ij}^{x}$ and $\delta_{ij}^{y}$ are relative horizontal and vertical shifts between patch $i$ and patch $j$ respectively. Moreover, $\boldsymbol{a}^{h} \in \mathbb{R}^{3 \times 1}$ are learnable parameters for each head. By initializing $\boldsymbol{a}^{h}$ as equation below, $\boldsymbol{P}^{h}$ gives larger weights to tokens in the neighborhood of size $\sqrt{H} \times \sqrt{H}$ relative to the underlying token,

$$\boldsymbol{a}^{h} = [-1, 2\Delta_{1}^{h}, 2\Delta_{2}^{h}]^{T}$$

where $\pmb{\Delta}^{h} = [\Delta_{1}^{h}, \Delta_{2}^{h}]$ is each of the possible positional offsets of the neighborhood of size $\sqrt{H} \times \sqrt{H}$ relative to the underlying token.
