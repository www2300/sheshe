import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用，便于调试

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import gc

# 清理内存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 设备配置
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU Memory: {gpu_mem:.2f} GB")
    # 设置GPU内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.8)  # 只使用80%的GPU内存
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
else:
    device = torch.device('cpu')
    print("Warning: CUDA not available, using CPU")

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


# 2. 卷积VAE模型定义（轻量化版本）
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(ConvVAE, self).__init__()

        # 编码器（减少通道数以降低内存使用）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 14, 14)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 7, 7)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  # (B, 128, 5, 5)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )

        # 潜在空间
        self.fc_mu = nn.Linear(128 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(128 * 5 * 5, latent_dim)

        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 128 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),  # (B, 64, 7, 7)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 14, 14)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (B, 1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, 5, 5)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# 3. 损失函数定义（优化版本）
def loss_function(recon_x, x, mu, logvar, lamda=1):
    # 使用mean reduction减少内存占用
    rec_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # 分批计算KL散度以减少内存占用
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = rec_loss + lamda * kl_loss
    return total_loss, rec_loss.item(), kl_loss.item()


# 4. 显示图像的函数
def show_images(images, title="Generated Images", save_path=None):
    images = images.cpu().detach().numpy()
    n = min(images.shape[0], 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title)

    for i in range(n):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        ax.imshow(images[i, 0], cmap='gray')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    try:
        # 超参数 - 大幅降低以适应小显存GPU
        latent_dim = 50  # 从100减小到50
        batch_size = 32  # 从64减小到32
        epochs = 100
        lr = 1e-3
        lamda = 0.1

        print(f"\n超参数设置 (低显存优化):")
        print(f"  Batch Size: {batch_size}")
        print(f"  Latent Dim: {latent_dim}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {lr}")

        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 加载MNIST数据集
        train_data = datasets.MNIST(
            root='./data', train=True, transform=transform, download=True
        )

        # 只选择数字0-7作为正常数据
        normal_digits_mask = (train_data.targets >= 0) & (train_data.targets <= 7)
        normal_indices = torch.where(normal_digits_mask)[0]

        print(f"\n正常数据样本数量: {len(normal_indices)}")
        print(f"正常数据标签分布: {torch.bincount(train_data.targets[normal_indices])}")

        # 使用自定义Dataset包装
        train_dataset = MNISTDataset(
            data=train_data.data[normal_indices].unsqueeze(1).float() / 255.0,
            label=train_data.targets[normal_indices],
            transform=None
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True  # 丢弃最后不完整的batch
        )

        # 初始化模型
        print("\n正在初始化轻量化模型...")
        model = ConvVAE(latent_dim=latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"初始GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # 5. 训练过程
        print("\n开始训练...\n")
        for epoch in range(epochs):
            model.train()
            epoch_loss = []
            rec_epoch_loss = []
            kl_epoch_loss = []

            for batch_idx, (data, _) in enumerate(train_loader):
                try:
                    # 将数据移到GPU
                    data = data.to(device)

                    # 清空梯度
                    optimizer.zero_grad()

                    # 前向传播
                    recon_data, mu, logvar = model(data)

                    # 计算损失
                    loss, rec_loss, kl_loss = loss_function(recon_data, data, mu, logvar, lamda)

                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告: Batch {batch_idx} 损失无效，跳过")
                        continue

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # 优化
                    optimizer.step()

                    # 记录损失
                    epoch_loss.append(loss.item())
                    rec_epoch_loss.append(rec_loss)
                    kl_epoch_loss.append(kl_loss)

                    # 定期清理内存
                    if batch_idx % 20 == 0:
                        del data, recon_data, mu, logvar, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                except RuntimeError as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg or "CUDA" in error_msg:
                        print(f"\n警告: Batch {batch_idx} GPU错误，清理缓存后继续")
                        optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

            # 打印训练信息
            if epoch_loss:
                avg_total_loss = sum(epoch_loss) / len(epoch_loss)
                avg_rec_loss = sum(rec_epoch_loss) / len(rec_epoch_loss)
                avg_kl_loss = sum(kl_epoch_loss) / len(kl_epoch_loss)

                gpu_info = ""
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    gpu_info = f", GPU: {gpu_mem:.2f}GB"

                print(f'Epoch [{epoch + 1}/{epochs}]{gpu_info}')
                print(f'  Total Loss: {avg_total_loss:.4f}, Recon Loss: {avg_rec_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

            # 每10个epoch保存图像
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    model.eval()
                    test_z = torch.randn(16, latent_dim).to(device)
                    generated_images = model.decode(test_z)
                    show_images(generated_images,
                                title=f"Epoch {epoch + 1} Generated Images",
                                save_path=f"generated_epoch_{epoch + 1}.png")
                    del test_z, generated_images

                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 保存最终结果
        print("\n生成最终结果...")
        with torch.no_grad():
            model.eval()
            test_z = torch.randn(16, latent_dim).to(device)
            generated_images = model.decode(test_z)
            show_images(generated_images,
                        title="Final Generated Images",
                        save_path="generated_final.png")

        # 保存模型
        torch.save(model.state_dict(), 'vae_normal_digits_0-7.pth')
        print("模型已保存为 'vae_normal_digits_0-7.pth'")

        # 测试重构
        print("\n测试重构效果...")
        with torch.no_grad():
            model.eval()
            test_batch, _ = next(iter(train_loader))
            test_batch = test_batch[:8].to(device)
            recon_batch, _, _ = model(test_batch)

            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            fig.suptitle("Original (top) vs Reconstructed (bottom)")

            for i in range(8):
                axes[0, i].imshow(test_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(recon_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig('reconstruction_comparison.png', dpi=100)
            print("重构对比图已保存")
            plt.close(fig)

        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n训练成功完成！")

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback

        traceback.print_exc()

        # 错误处理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        plt.close('all')
        gc.collect()
        print("\n程序结束")