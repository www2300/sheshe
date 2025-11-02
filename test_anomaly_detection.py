import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib使用支持中文的字体，或者使用英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体或默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型定义（保持不变）
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.fc_mu = nn.Linear(256 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 5, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 5, 5)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# 加载模型
model = ConvVAE(latent_dim=100).to(device)
model_path = r"F:\pycharm\pythonProject4\vae_normal_digits_0-7.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("模型加载成功！")

# 数据预处理
transform = transforms.ToTensor()
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
fashion_test = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 测试函数 - 使用英文标签
def test_images(model, images, labels, title):
    errors = []
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title)

    for i in range(4):
        if i >= len(images):
            break

        # 原始图像
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original (Label: {labels[i]})')
        axes[0, i].axis('off')

        # 重构图像
        with torch.no_grad():
            image_tensor = images[i].unsqueeze(0).to(device)
            recon_image, _, _ = model(image_tensor)
            error = torch.mean((recon_image - image_tensor) ** 2).item()
            errors.append(error)

            axes[1, i].imshow(recon_image.squeeze().cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed\nError: {error:.4f}')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    avg_error = np.mean(errors)
    print(f"{title} - Average Error: {avg_error:.6f}")
    return avg_error

# 存储测试结果
results = {}

# 1. 测试正常数据
print("Testing Normal Data (Digits 0-7)...")
normal_indices = [i for i, label in enumerate(mnist_test.targets) if label <= 7][:4]
normal_images = [mnist_test[i][0] for i in normal_indices]
normal_labels = [mnist_test[i][1] for i in normal_indices]
results['normal'] = test_images(model, normal_images, normal_labels, "Normal Data (Digits 0-7)")

# 2. 测试数字8
print("\nTesting Digit 8...")
digit8_indices = [i for i, label in enumerate(mnist_test.targets) if label == 8][:4]
digit8_images = [mnist_test[i][0] for i in digit8_indices]
digit8_labels = [mnist_test[i][1] for i in digit8_indices]
results['digit8'] = test_images(model, digit8_images, digit8_labels, "Anomaly Data (Digit 8)")

# 3. 测试数字9
print("\nTesting Digit 9...")
digit9_indices = [i for i, label in enumerate(mnist_test.targets) if label == 9][:4]
digit9_images = [mnist_test[i][0] for i in digit9_indices]
digit9_labels = [mnist_test[i][1] for i in digit9_indices]
results['digit9'] = test_images(model, digit9_images, digit9_labels, "Anomaly Data (Digit 9)")

# 4. 测试Fashion-MNIST
print("\nTesting Fashion-MNIST...")
fashion_indices = list(range(4))
fashion_images = [fashion_test[i][0] for i in fashion_indices]
fashion_labels = [fashion_test[i][1] for i in fashion_indices]
fashion_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_label_names = [fashion_classes[label] for label in fashion_labels]
results['fashion'] = test_images(model, fashion_images, fashion_label_names, "Anomaly Data (Fashion-MNIST)")

# 5. 显示误差对比
print("\nError Comparison Summary:")
categories = ['Normal(0-7)', 'Digit 8', 'Digit 9', 'Fashion-MNIST']
errors = [results['normal'], results['digit8'], results['digit9'], results['fashion']]

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, errors, color=['green', 'red', 'red', 'orange'])
plt.ylabel('Average Reconstruction Error')
plt.title('Normal vs Anomaly Data Reconstruction Error Comparison')

for bar, error in zip(bars, errors):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{error:.4f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.show()

print("\nTesting Completed!")