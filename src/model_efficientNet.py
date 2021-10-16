from efficientnet_pytorch import EfficientNet
import torch.nn as nn
 
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

        print(self.model.parameters)

KeypointEfficientNet(True, False)