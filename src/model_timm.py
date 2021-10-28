import timm
from pprint import pprint
import torch
import torch.nn as nn
from torchsummary import summary

# Listing all models with pretrained weights
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

tf_models = [
    'tf_mobilenetv3_large_075',
    'tf_mobilenetv3_large_100',
    'tf_mobilenetv3_large_minimal_100',
    'tf_mobilenetv3_small_075',
    'tf_mobilenetv3_small_100',
    'tf_mobilenetv3_small_minimal_100',
    'tf_inception_v3',
    'tf_mixnet_m',
    'tf_mixnet_s',
    'tf_efficientnetv2_s_in21ft1k',
    'tf_efficientnetv2_m_in21ft1k',
    'tf_efficientnet_lite0',
    'tf_efficientnet_lite1',
    'tf_efficientnet_lite2',
    'tf_efficientnet_lite3',
]
models_to_evaluate = [   
    'efficientnet_b0',
    'efficientnet_b1_pruned',
    'efficientnet_b2_pruned',

    'tf_efficientnet_lite0',
    'tf_efficientnet_lite1',
    'tf_efficientnet_lite2',
    'tf_efficientnet_lite3',

    'efficientnetv2_rw_m',
    'efficientnetv2_rw_s',
    'tf_efficientnetv2_s_in21ft1k',
    'tf_efficientnetv2_m_in21ft1k',

    'inception_v3',
    'inception_v4',

    'mixnet_l',
    'mixnet_m',
    'mixnet_s',

    'mobilenetv3_large_100_miil_in21k',
    'mobilenetv3_rw',
    'tf_mobilenetv3_small_075',
    'tf_mobilenetv3_small_100',
    'mobilenetv2_110d',
    'mobilenetv2_120d',

    'resnet18d',
    'resnet26d',
    'resnet34d',
    'resnet50d'
]

#pprint(timm.list_models(pretrained=True))
model = timm.create_model('resnet50d', pretrained=False, num_classes=16)
pprint(model)

for name, param in model.named_parameters():
    if 'classifier' not in name:
        #pass
        param.requires_grad = False

#summary(model, (3, 80, 80))

