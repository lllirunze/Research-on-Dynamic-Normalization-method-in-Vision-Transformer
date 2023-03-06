import torch
from torch import nn

from functools import partial

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 norm_layer=None
                 ):
        """
        Add learned positional embeddings to the inputs
        :param image_size: input image size
        :param patch_size: patch size
        :param in_channels: number of input channels
        :param embed_dim: output embedding dimension = patch_size * patch_size * in_channels
        :param norm_layer: the function of normalization
        """
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        # grid_size is width and height of patch
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        # num_patches is the number of patch
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # The input tensor is divided into patches using patch_size*patch_size convolution
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x.shape = [batch_size, feature_dimension, height, width]
        B, C, H, W = x.shape
        assert ((H == self.image_size[0]) and (W == self.image_size[1])), \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        # Apply convolution layer
        x = self.conv(x)
        # Flatten: [B, C, H, W] -> [B, C, H*W]
        x = x.flatten(2)
        # transpose: [B, C, H*W] -> [B, H*W, C]
        x = x.transpose(1, 2)
        # Apply normalization function (default: None)
        x = self.norm(x)

        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention
    "Attention is All You Need"
    """

    # Attention is all you need
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 # linear_drop_ratio=0.
                 ):
        """
        MHA computes scaled multi-head attention for given query, key and value vectors
        :param dim: number of features in the query, key and value vectors
        :param num_heads: number of head
        :param qkv_bias: enable bias for qkv if True
        :param qk_scale: scaling factor before the softmax
        :param attn_drop_ratio: dropout rate of attention
        # :param linear_drop_ratio: dropout rate of projection
        """
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = qk_scale or (dim_head ** (-0.5))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.linear = nn.Linear(dim, dim)
        # self.linear_drop = nn.Dropout(linear_drop_ratio)

    def forward(self, x):
        # x.shape = [batch_size, num_patches + 1, embed_dimension]
        B, N, C = x.shape
        # qkv Linear: [B, N, C] -> [B, N, C*3]
        qkv = self.qkv(x)
        # reshape: [B, N, C*3] -> [B, N, 3, num_heads, embed_dimension per head]
        qkv = qkv.reshape(B, N, 3, self.num_heads, (C // self.num_heads))
        # permute: [B, N, 3, num_heads, embed_dimension per head] -> [3, B, num_heads, N, embed_dimension per head]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Divide qkv into q, k and v (.shape = [B, num_heads, N, embed_dimension per head])
        q, k, v = qkv[0], qkv[1], qkv[2]
        # transpose: [B, num_heads, N, embed_dimension per head] -> [B, num_heads, embed_dimension per head, N]
        # matmul: [B, num_heads, N, embed_dimension per head] * [B, num_heads, embed_dimension per head, N] = [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1))
        # Scale attention(scores)
        attn = attn * self.scale
        # Apply softmax
        attn = attn.softmax(dim=-1)
        # Apply dropout
        attn = self.attn_drop(attn)
        # matmul: [B, num_heads, N, N] * [B, num_heads, N, embed_dimension per head] = [B, num_heads, N, embed_dimension per head]
        # transpose: [B, num_heads, N, embed_dimension per head] -> [B, N, num_heads, embed_dimension per head]
        x = (attn @ v).transpose(1, 2)
        # reshape(Concat): [B, N, num_heads, embed_dimension per head] -> [B, N, C]
        x = x.reshape(B, N, C)
        # Apply linear
        x = self.linear(x)
        # TODO: Apply dropout ???
        # x = self.linear_drop(x)

        return x

class MultiLayerPerceptron(nn.Module):
    """
    Multilayer Perceptron (MLP)
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 act_layer=nn.GELU,
                 drop_ratio=0.):
        """
        MLP consists of three layers from in_channels, hidden_channels to out_channels
        :param in_channels: number of input channels
        :param hidden_channels: number of hidden channels = (in_channels * mlp_ratio)
        :param out_channels: number of out channels
        :param act_layer: activation function
        :param drop_ratio: dropout rate
        """
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        # TODO: self.act 报错：NoneType object is not callable，暂时没找到解决方法，故直接用 nn.GELU()
        # self.act = act_layer()
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(drop_ratio)
        self.fc2 = nn.Linear(in_features=hidden_channels, out_features=out_channels)
        self.dropout2 = nn.Dropout(drop_ratio)

    def forward(self, x):
        # Fully connected layer: in_channels -> hidden_channels
        x = self.fc1(x)
        # Activation function (default: GELU)
        x = self.act(x)
        # Dropout
        x = self.dropout1(x)
        # Fully connected layer: hidden_channels -> out_channels
        x = self.fc2(x)
        # Dropout
        x = self.dropout2(x)

        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Block
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 # drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """
        Basic Transformer blocks are included in Vision Transformer
        :param dim: embedding dimension = patch_size * patch_size * in_channels
        :param num_heads: number of attention heads
        :param mlp_ratio: ratio of hidden layer in MLP
        :param qkv_bias: enable bias for qkv if True
        :param qk_scale: override default qk scale of head_dim ** -0.5 if set
        :param drop_ratio: dropout rate
        :param attn_drop_ratio: attention dropout rate
        # :param drop_path_ratio: stochastic depth rate
        :param act_layer: activation function
        :param norm_layer: normalization function
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mha = MultiHeadAttention(dim=dim,
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      attn_drop_ratio=attn_drop_ratio,
                                      # linear_drop_ratio=drop_ratio
                                      )
        self.dropout1 = nn.Dropout(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_channels = int(dim * mlp_ratio)
        self.mlp = MultiLayerPerceptron(in_channels=dim,
                                        hidden_channels=hidden_channels,
                                        out_channels=dim,
                                        act_layer=act_layer,
                                        drop_ratio=drop_ratio)
        self.dropout2 = nn.Dropout(drop_ratio) if drop_ratio > 0. else nn.Identity()

    def forward(self, x):
        shortcut1 = x
        # Normalization Layer
        x = self.norm1(x)
        # Multi-head Attention
        x = self.mha(x)
        # Dropout
        x = self.dropout1(x)
        # Add shortcut
        x = x + shortcut1

        shortcut2 = x
        # Normalization Layer
        x = self.norm2(x)
        # Multilayer Perceptron
        x = self.mlp(x)
        # Dropout
        x = self.dropout2(x)
        # Add shortcut
        x = x + shortcut2

        return x

class MLPClassificationHead(nn.Module):
    """
    MLP Head
    This consists of Pre-Logits and Linear layer.
    Pre-Logits consists of Linear layer and Tanh function.
    """

    def __init__(self,
                 embed_dim,
                 num_classes,
                 representation_size=None):
        """
        MLP classification head
        :param embed_dim: embedding dimension = patch_size * patch_size * in_channels
        :param representation_size: final representation size after classification
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.representation_size = representation_size
        if self.representation_size is not None:
            self.hidden_dim = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.hidden_dim = embed_dim
            self.pre_logits = nn.Identity()
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # Pre-Logits
        x = self.pre_logits(x)
        # Fully connected layer
        x = self.fc(x)

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)
    "An image is worth 16x16 words: Transformers for image recognition at scale"
    """

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 representation_size=None,
                 layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 embed_layer=PatchEmbed,
                 act_layer=None,
                 norm_layer=nn.LayerNorm):
        """
        Vision Transformer network architecture
        :param image_size: image size
        :param patch_size: patch size
        :param in_channels: number of input channels
        :param num_classes: number of classes for classification head
        :param embed_dim: embedding dimension = patch_size * patch_size * in_channels
        :param representation_size: representation size in pre-logits of MLP classification head
        :param layers: layers of transformer
        :param num_heads: number of multi-head attention heads
        :param mlp_ratio: ratio of MLP from input channels to hidden channels
        :param qkv_bias: enable bias for qkv if True
        :param qk_scale: scaling factor before the softmax
        :param drop_ratio: dropout rate
        :param attn_drop_ratio: dropout rate of attention
        :param embed_layer: patch embedding layer
        :param act_layer: activation function
        :param norm_layer: normalization function
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.representation_size = representation_size
        self.act = act_layer or nn.GELU
        # self.norm = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.patch_embed = embed_layer(image_size=image_size,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.layers = layers
        """
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(dim=embed_dim,
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop_ratio=drop_ratio,
                               attn_drop_ratio=attn_drop_ratio,
                               act_layer=act_layer,
                               norm_layer=norm_layer)
            for layer_i in range(layers)
        ])
        """
        self.transformer_encoder = TransformerEncoder(dim=embed_dim, 
                                                      num_heads=num_heads, 
                                                      mlp_ratio=mlp_ratio, 
                                                      qkv_bias=qkv_bias, 
                                                      qk_scale=qk_scale, 
                                                      drop_ratio=drop_ratio, 
                                                      attn_drop_ratio=attn_drop_ratio, 
                                                      act_layer=act_layer, 
                                                      norm_layer=norm_layer)
        self.mlp_head = MLPClassificationHead(embed_dim=embed_dim,
                                              num_classes=num_classes,
                                              representation_size=representation_size)

    def forward(self, x):
        # Patch Embedding
        # [B, C, H, W] -> [B, H*W(N), C(embed_dim)]
        x = self.patch_embed(x)

        # Add parameterized positional encodings
        # [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, N, C] -> [B, 1+N, C]
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embed

        # Transformer Encoder (L layers)
        # [B, 1+N, C]
        # In this process, the input dimension is same as the output dimension.
        # x = self.transformer_encoder(x)
        for layer_i in range(self.layers):
            x = self.transformer_encoder(x)
        # Note: There is still a normalization layer after Transformer Encoder
        x = self.norm(x)
        # MLP Classification Head.
        # Note: We only need classified information,
        #       so we just need to extract the corresponding results generated by the class token
        # [B, 1+N, C] -> [B, C]
        x = x[:, 0]
        # [B, C] -> [B, num_classes]
        x = self.mlp_head(x)

        return x

def vit_base_patch16_224_cifar10(num_classes: int=10, in_channels: int=3, has_logits: bool=True):

    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              in_channels=in_channels,
                              num_classes=num_classes,
                              embed_dim=768,
                              representation_size=768 if has_logits else None,
                              layers=12,
                              num_heads=12)

    return model

def vit_base_patch7_28_mnist(num_classes: int=10, in_channels: int=1, has_logits: bool=True):

    model = VisionTransformer(image_size=28,
                              patch_size=7,
                              in_channels=in_channels,
                              num_classes=num_classes,
                              embed_dim=64,
                              representation_size=64 if has_logits else None,
                              layers=6,
                              num_heads=8)

    return model
