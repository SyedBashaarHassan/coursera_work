# ============================
# 1. Required Imports
# ============================
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.impute import SimpleImputer


# ============================
# 2. MNIST Validation Image
# Resize to 16x16, convert to tensor, display first sample
# ============================

# Assumes mnist_val_dataset contains the MNIST validation dataset
mnist_image, mnist_label = mnist_val_dataset[0]

if isinstance(mnist_image, torch.Tensor):
    mnist_image = transforms.Resize((16, 16))(mnist_image)
else:
    mnist_image = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])(mnist_image)

if not isinstance(mnist_image, torch.Tensor):
    mnist_image = transforms.ToTensor()(mnist_image)

plt.figure(figsize=(3, 3))
plt.imshow(mnist_image.squeeze().cpu().numpy(), cmap="gray")
plt.title(f"y = {int(mnist_label)}")
plt.axis("off")
plt.show()

print("MNIST image shape:", tuple(mnist_image.shape))


# ============================
# 3. FashionMNIST Validation Images
# Resize first 3 images to 16x16 and display
# ============================

# Assumes fashion_val_dataset contains the FashionMNIST validation dataset
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for i in range(3):
    image, label = fashion_val_dataset[i]

    if isinstance(image, torch.Tensor):
        image = transforms.Resize((16, 16))(image)
    else:
        image = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor()
        ])(image)

    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)

    axes[i].imshow(image.squeeze().cpu().numpy(), cmap="gray")
    axes[i].set_title(f"y = {int(label)}")
    axes[i].axis("off")

    print(
        f"FashionMNIST sample {i + 1}: "
        f"label = {int(label)}, "
        f"shape = {tuple(image.shape)}"
    )

plt.tight_layout()
plt.show()
