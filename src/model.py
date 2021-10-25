import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import timm

class KeypointResNet(nn.Module):
    def __init__(self, pretrained, requires_grad, model_name='resnet18'):
        if model_name not in ['resnet18', 'resnet34', 'resnet50']:
            print("Invalid ResNet Model. Only accepted 'resnet18', 'resnet34' or 'resnet50'")
            exit()
        
        super(KeypointResNet, self).__init__()
        
        if pretrained == True:
            # Change to resnet18 or resnet50
            self.model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        else:
            # Change to resnet18 or resnet50
            self.model = pretrainedmodels.__dict__[model_name](pretrained=None)
        
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        
        # change the final layer
        # 136 represents the 68 keypoints (each containing two values x and y) for facial recognition
        # For Cone Detection, please change it to 16
        # Change to (512, 136) for resnet18
        # Change to (2048, 136) for resnet50
        self.l0 = nn.Linear(512, 136)



    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0


class KeypointEfficientNet():
    def __init__(self, pretrained, requires_grad, model_name='efficientnet-b0'):
        if model_name not in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5']:
            print("Invalid EfficientNet Model. Only accepted 'efficientnet-b0, 1, 2, 3, 4, 5")
            exit()
        
        if pretrained == True:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=136)
        else:
            self.model = EfficientNet.from_name(model_name, num_classes=136)

        if requires_grad == True:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            print("Training just the final layer and freezing intermediate layers...")

        if requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Freezing all layers for inference......")
    
    def return_loaded_model(self):   
        return self.model


class KeypointCustom():
    def __init__(self, isPretrained, requires_grad, model_name='tf_efficientnetv2_s_in21ft1k'):

        self.model = timm.create_model(model_name, pretrained=isPretrained, num_classes=16)

        if requires_grad == True:
            # for name, param in self.model.named_parameters():
            #     param.requires_grad = False

            print("Training all the layers")

        if requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Freezing all layers for inference......")
    
    def return_loaded_model(self):   
        return self.model