import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. 数据集定义（使用MNIST官方数据）
class MNISTDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.label[idx]


# 2. 卷积VAE模型定义
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(ConvVAE, self).__init__()

        # 编码器（卷积部分）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 14, 14) → 输入28x28时
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 7, 7)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  # (B, 256, 5, 5) → 调整卷积核适应尺寸
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )

        # 潜在空间（均值和对数方差）
        self.fc_mu = nn.Linear(256 * 5 * 5, latent_dim)  # 256*5*5 = 6400
        self.fc_logvar = nn.Linear(256 * 5 * 5, latent_dim)

        # 解码器（转置卷积部分）
        self.decoder_input = nn.Linear(latent_dim, 256 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),  # (B, 128, 7, 7)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 14, 14)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # (B, 1, 28, 28)
            nn.Sigmoid()  # 输出像素值范围[0,1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 5, 5)  # 重塑为特征图
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# 3. 损失函数定义
def loss_function(recon_x, x, mu, logvar, lamda=1):
    rec_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')  # 重构损失（MSE）
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL散度
    return rec_loss + lamda * KLD, rec_loss.item(), KLD.item()


# 4. 显示图像的函数
def show_images(images, title="Generated Images"):
    """
    显示生成的图像
    images: tensor of shape (N, 1, 28, 28)
    """
    # 将图像转换为numpy数组并调整维度
    images = images.cpu().detach()
    images = images.numpy()

    # 创建子图
    n = min(images.shape[0], 16)  # 最多显示16张图像
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title)

    for i in range(n):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        ax.imshow(images[i, 0], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 超参数
    latent_dim = 100
    batch_size = 128
    epochs = 100
    lr = 1e-3
    lamda = 0.1  # KL散度系数

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
    ])

    # 加载MNIST数据集
    train_data = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )

    # 只选择数字0-7作为正常数据
    # 创建掩码，选择标签为0-7的样本
    normal_digits_mask = (train_data.targets >= 0) & (train_data.targets <= 7)

    # 获取正常数据的索引
    normal_indices = torch.where(normal_digits_mask)[0]

    print(f"正常数据样本数量: {len(normal_indices)}")
    print(f"正常数据标签分布: {torch.bincount(train_data.targets[normal_indices])}")

    # 使用自定义Dataset包装，只包含正常数据
    train_dataset = MNISTDataset(
        data=train_data.data[normal_indices].unsqueeze(1).float() / 255.0,  # 转为[0,1]范围
        label=train_data.targets[normal_indices],
        transform=None  # 已提前处理，这里无需额外转换
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # 初始化模型、优化器
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. 训练过程
    for epoch in range(epochs):
        model.train()
        epoch_loss = []  # 定义当前 epoch 的损失列表
        rec_epoch_loss = []
        kl_epoch_loss = []

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)  # 数据移至设备
            optimizer.zero_grad()  # 清空梯度

            # 前向传播
            recon_data, mu, logvar = model(data)

            # 计算损失
            loss, rec_loss, kl_loss = loss_function(recon_data, data, mu, logvar, lamda)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss.append(loss.item())
            rec_epoch_loss.append(rec_loss)
            kl_epoch_loss.append(kl_loss)

        # 打印每轮训练信息
        avg_total_loss = sum(epoch_loss) / len(epoch_loss)
        avg_rec_loss = sum(rec_epoch_loss) / len(rec_epoch_loss)
        avg_kl_loss = sum(kl_epoch_loss) / len(kl_epoch_loss)
        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'  Total Loss: {avg_total_loss:.2f}, Recon Loss: {avg_rec_loss:.2f}, KL Loss: {avg_kl_loss:.2f}')

        # 每10个epoch显示一次生成的图像
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                # 从标准正态分布采样生成潜在向量
                test_z = torch.randn(16, latent_dim).to(device)
                generated_images = model.decode(test_z)
                show_images(generated_images, title=f"Epoch {epoch + 1} Generated Images")

    # 训练完成后显示最终的生成结果
    print("\nFinal generation results:")
    with torch.no_grad():
        model.eval()
        # 从标准正态分布采样生成潜在向量
        test_z = torch.randn(16, latent_dim).to(device)
        generated_images = model.decode(test_z)
        show_images(generated_images, title="Final Generated Images")

    # 保存训练好的模型
    torch.save(model.state_dict(), 'vae_normal_digits_0-7.pth')
    print("Model saved as 'vae_normal_digits_0-7.pth'")

    # 可选：测试模型在正常数据上的重构效果
    print("\nTesting reconstruction on normal digits:")
    with torch.no_grad():
        model.eval()
        # 获取一批正常数据
        test_batch, _ = next(iter(train_loader))
        test_batch = test_batch[:8].to(device)  # 取前8个样本
        recon_batch, _, _ = model(test_batch)

        # 显示原始图像和重构图像
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle("Original (top) vs Reconstructed (bottom)")

        for i in range(8):
            # 原始图像
            axes[0, i].imshow(test_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')

            # 重构图像
            axes[1, i].imshow(recon_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
