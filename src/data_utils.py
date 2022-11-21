import torchvision
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

class OccTransform:

    def __init__(self, transform, mask_size=224): 
        """
            width and hieght extend from (0,0) which is the left corner of an image
            Default mask upper-left occlusion
        """
        self.mask_size = mask_size
        self.transform = transform 

    def __call__(self, sample):
        """
        sample is [n_channels, img_width, img_height]
        """
        print(sample.shape)
        print(sample)

        #upper-left mask 
        mask = torch.tensor([[0,1], [1,1]]).unsqueeze(0)    #(224, 224, 3) to (1, 3, 224, 224) 

        #applying custom transform
        if self.transform:
            mask = torch.tensor([[0,1], [0,1]]).unsqueeze(0)    
        
        mask = T.Resize(self.mask_size, interpolation=InterpolationMode.NEAREST)(mask)

        assert mask[0].shape == sample[0].shape
        cropped = mask * sample
        # print(cropped.shape)
        return cropped


