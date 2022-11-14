
import torchvision
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

class OccTransform:
    """sdfgasfga"""

    def __init__(self, mask_size=224, random_horizontal_flip=True):
        """
            width and hieght extend from (0,0) which is the left corner of an image
        """
        self.mask_size= mask_size
        self.flip = random_horizontal_flip

    def __call__(self, sample):
        """
        sample is [n_channels, img_width, img_height]
        """
        # print(sample.shape)

        mask = torch.tensor([[0,1], [1,1]]).unsqueeze(0)
        mask = T.Resize(self.mask_size, interpolation=InterpolationMode.NEAREST)(mask)
        if self.flip:
            mask = T.RandomHorizontalFlip(p=0.5)(mask)

        # print(mask.shape)
        # print(sample.shape)
        assert mask[0].shape == sample[0].shape
        cropped = mask * sample
        # print(cropped.shape)
        return cropped