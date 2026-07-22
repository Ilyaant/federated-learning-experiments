import torch.nn as nn


class ConvNextV2Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(
            backbone.config.hidden_sizes[-1],
            num_classes
        )
    
    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        x = outputs.last_hidden_state.mean(dim=(2, 3))
        return self.classifier(x)
