import torch
from torchvision import transforms


old_tens = torch.rand(1,5,5)
print(old_tens)

transformer = transforms.Compose([
    transforms.Resize(3,3)
])

new_tens = transformer(old_tens)
print(new_tens)