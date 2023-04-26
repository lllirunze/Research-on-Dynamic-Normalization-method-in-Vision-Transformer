import timm
feature_extractor = timm.create_model('vit_small_patch16_224', pretrained=True)

from timm.models.vision_transformer import vit_small_patch16_224, vit_small_patch16_224_in21k

# model = timm.create_model('vit_small_patch16_224_in21k')

# vit_small_patch16_224_in21k
# vit_small_patch16_224