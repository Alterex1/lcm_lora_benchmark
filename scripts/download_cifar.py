from torchvision.datasets import CIFAR10
from torchvision import transforms
import os

dataset = CIFAR10(
    root="data",
    download=True,
    transform=transforms.ToTensor()
)

os.makedirs("data/reference_images", exist_ok=True)

for i in range(500):  # 500 is enough for your project
    img, _ = dataset[i]
    transforms.ToPILImage()(img).save(f"data/reference_images/{i}.png")

print("Done.")