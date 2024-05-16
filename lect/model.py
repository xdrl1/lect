
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import vit_b_16

# Define the ResNet-based model
class CustomResNet(nn.Module):
    def __init__(self, embedding_size=3):
        super(CustomResNet, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')

        # freeze params
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Add a new fully connected layer with the specified number of classes
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Linear(in_features=512, out_features=embedding_size, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x  # [N, embedding_size]

# ResNet-Softmax
class CustomSoftmaxResNet(nn.Module):
    def __init__(self, embedding_size=3):
        super(CustomSoftmaxResNet, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')

        # freeze params
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Add a new fully connected layer with the specified number of classes
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Linear(in_features=512, out_features=embedding_size, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x  # [N, embedding_size]

# Define the ViT-based model
class CustomViT(nn.Module):
    def __init__(self, embedding_size=3, pretrained=True):
        super(CustomViT, self).__init__()

        self.model = vit_b_16(weights = 'IMAGENET1K_V1')

        # freeze params
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.model.heads.parameters():
            param.requires_grad = True

        # last layer fc
        self.model.heads.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=embedding_size, bias=True)
        )

    def forward(self, x):
        # print(f"0 {x.shape}")
        x = self.model(x)
        return x


# Define the ViT-softmax model
class CustomSoftmaxViT(nn.Module):
    def __init__(self, embedding_size=3, pretrained=True):
        super(CustomSoftmaxViT, self).__init__()

        self.model = vit_b_16(weights = 'IMAGENET1K_V1')

        # freeze params
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.model.heads.parameters():
            param.requires_grad = True

        # last layer fc
        self.model.heads.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=embedding_size, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print(f"0 {x.shape}")
        x = self.model(x)
        return x