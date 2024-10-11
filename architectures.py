'''
This file contains custom models architectures. 
It is meant to be imported in the classifier_train.py script using : 

from architectures import EffNetB0
model = EffNetB0(num_classes=3, model_path).create_model()
'''

import torch
import torch.nn as nn
from torchvision import  models


# EfficientNetB0 Model with custom head
class EffNetB0(nn.Module):
    
    def __init__(self, num_classes=3, model_path=None):
        super(EffNetB0, self).__init__()
        self.model_path = model_path
        self.num_classes = num_classes
    
    def create_model(self):
        if self.model_path:
            # Load without pretrained weights
            model = models.efficientnet_b0(weights=None)
            # Load the pretrained weights
            model.load_state_dict(torch.load(self.model_path))
        else:
            # Load with pretrained weights
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        num_ftrs = model.classifier[1].in_features
        
        # Modify the classifier with custom layers
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes),
        )

        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))

        return model
    
