
import torchvision
import torch

class OccTransform:
    """sdfgasfga"""

    def __init__(self, width=112, height=112):
        """
            width and hieght extend from (0,0) which is the left corner of an image
        """
        self.crop_width = width
        self.crop_height = height

    def __call__(self, sample):
        """
        sample is [n_channels, img_width, img_height]
        """
        print(sample.shape)
        # you will hate the way I constructed this mask
        mask = torch.zeros((112,112))
        pad = torch.ones((112,112))
        mask = torch.cat((mask, pad))
        pad = torch.cat((pad, pad))
        mask = torch.cat((mask, pad), axis=1)

        assert mask.shape == sample[0].shape
        cropped = mask * sample

        return cropped