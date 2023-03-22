# Unified Normalization

Unified Normalization(UN) can speed up the inference by being fused with other linear operation and achieve comparable performance on par with LN.

UN consists of **a unified framework** for leveraging offline methods, **a tailored fluctuation smoothing strategy** to mitigate the fluctuations and **an adaptive outlier filtration strategy** for stabilizing training.

## 1 Unified Framework

### Forward Propagation:

$$ Z_{t} = \frac{X_{t}- \hat{\mu_{t}}}{\sqrt{\hat{\sigma_{t}}^{2} + \epsilon}} $$

$$ Y_{t} = \gamma \cdot Z_{t} + \beta $$

Let $Z_{t}$ denote the normalized alternative to input $X_{t}$ at iteration $t$. The training statistics for normalizing are marked as $\hat{\mu_{t}}$ and $\hat{\sigma_{t}}^{2}$, given by

$$ \hat{\mu_{t}} = \Theta_{\mu}(\mu_{t}, \cdots, \mu_{t-M+1}) $$

$$ \hat{\sigma_{t}}^{2} = \Theta_{\sigma^{2}}(\sigma_{t}^{2}, \cdots, \sigma_{t-M+1}^{2}) $$

### Backward Propagation:

The gradients of loss $L$ pass as:

$$ \frac{\partial L}{\partial Z_{t}} = \gamma \cdot \frac{\partial L}{\partial Y_{t}} $$

$$ \frac{\partial L}{\partial X_{t}} = \frac{1}{\sqrt{\hat{\sigma_{t}}^{2} + \epsilon}} (\frac{\partial L}{\partial Z_{t}} - \psi_{\hat{\mu_{t}}} - Z_{t} \cdot \psi_{\hat{\sigma_t}^{2}}) $$

Giving gradients $\frac{\partial L}{\partial Y_{t}}$, $\psi_{\hat{\mu_{t}}}$ and $\psi_{\hat{\sigma_t}^{2}}$ indicate the gradient statistics that used for estimating $\frac{\partial L}{\partial X_{t}}$. In this framework, estimated gradients are gained from averaging functions $\Theta_{g_{\mu}}$ and $\Theta_{g_{\sigma^{2}}}$,

$$ \psi_{\hat{\mu_{t}}} = \Theta_{g_{\mu}}(g_{\hat{\mu_{t}}}, \cdots, g_{\hat{\mu}_{t-M+1}}) $$

$$ \psi_{\hat{\sigma_{t}}^{2}} = \Theta_{g_{\sigma^{2}}}(g_{\hat{\sigma_{t}}^{2}}, \cdots, g_{\hat{\sigma}_{t-M+1}^{2}}) $$

The gradients passed from $\hat{\mu_{t}}$ and $\hat{\sigma_{t}}^{2}$ are denoted as $g_{\hat{\mu_t}}$ and $g_{\hat{\sigma_{t}}^{2}}$.

## 2 Fluctuation Smoothing

We turn ot adopt geometric mean(GM) with less sensitivity to outliers instead of arithmetic mean(AM) to gain a better representation of activation statistics in a skewed distribution.

The averaging functions are defined as:

$$\hat{\mu_{t}} = 0,\quad \hat{\sigma_{t}}^{2} = \sqrt[M]{\prod_{i=0}^{M-1} \sigma_{t-i}^{2}}$$

$$\psi_{\hat{\mu_{t}}} = 0, \quad \psi_{\hat{\sigma}_{t}^{2}} = \alpha \psi_{\hat{\sigma}_{t-1}^{2}} + (1- \alpha)\frac{1}{M} \sum_{i=0}^{M-1} g_{\hat{\sigma}_{t-1}^{2}}$$

## 3 Outlier Filtration

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
