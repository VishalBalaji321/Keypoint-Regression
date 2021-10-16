import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class FaceKeypointResNet(nn.Module):
    def __init__(self, pretrained, requires_grad, model='resnet18'):
        if model not in ['resnet18', 'resnet34', 'resnet50']:
            print("Invalid ResNet Model. Only accepted 'resnet18', 'resnet34' or 'resnet50'")
            exit()
        
        super(FaceKeypointResNet, self).__init__()
        
        if pretrained == True:
            # Change to resnet18 or resnet50
            self.model = pretrainedmodels.__dict__[model](pretrained='imagenet')
        else:
            # Change to resnet18 or resnet50
            self.model = pretrainedmodels.__dict__[model](pretrained=None)
        
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        
        # change the final layer
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