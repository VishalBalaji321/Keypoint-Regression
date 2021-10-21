import timm
from pprint import pprint
import torch
import torch.nn as nn

# Listing all models with pretrained weights
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

#pprint(timm.list_models())
model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, checkpoint_path='../weights/tf_efficientnetv2_s_21ft1k_pretrained.pth')
print(type(model))

