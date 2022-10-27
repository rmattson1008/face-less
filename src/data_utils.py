
import torchvision
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
        

        # The crop is taken from original, unblurred image
        cropped = torchvision.transforms.functional.crop()

        return cropped