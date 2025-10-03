import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

to_tensor = [Resize((144,144)), ToTensor()]

class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
    
def show_images(images, num_samples=20, cols=4):
    plt.figure(figsize=(15,15))
    idx = int(len(dataset)/num_samples)
    print(images)
    for i, img in enumerate(images):
        if i%idx==0:

            plt.subplot(int(num_samples/cols)+1, cols, int(i/idx)+1)
            plt.imshow(to_pil_image(img[0]))
            # plt.savefig("my_plot.png")

dataset = OxfordIIITPet(root=".", download=True, transform=compose(to_tensor))
show_images(dataset)