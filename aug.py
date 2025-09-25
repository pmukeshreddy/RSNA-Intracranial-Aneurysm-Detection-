import torch
import torch.nn.functional as F
import random

class Random3DFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):  # x is now a tensor, not dict
        if random.random() < self.prob:
            # Random flip along each spatial axis
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])  # Depth
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])  # Height
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])  # Width
        return x

class RandomIntensityScale:
    def __init__(self, factor=0.1, prob=0.5):
        self.factor = factor
        self.prob = prob

    def __call__(self, x):  # x is a tensor
        if random.random() < self.prob:
            scale = 1 + random.uniform(-self.factor, self.factor)
            x = x * scale
        return x

class RandomIntensityShift:
    def __init__(self, offset=0.1, prob=0.5):
        self.offset = offset
        self.prob = prob

    def __call__(self, x):  # x is a tensor
        if random.random() < self.prob:
            shift = random.uniform(-self.offset, self.offset)
            x = x + shift
        return x

class Random3DRotation:
    def __init__(self, max_angle=15, prob=0.5):
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, x):
        # Skip rotation for now - complex to implement
        return x

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):  # x is a tensor
        for transform in self.transforms:
            x = transform(x)
        return x

# Replace your MONAI transforms with this:
def get_train_transforms():
    return Compose([
        Random3DFlip(prob=0.5),
        RandomIntensityScale(factor=0.1, prob=0.5),
        RandomIntensityShift(offset=0.1, prob=0.5),
    ])
