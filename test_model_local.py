import torch
import sys
import os
import numpy as np

# 将 train 目录加入路径，以便导入 vint_train 和 diffusion_policy 下的模块
sys.path.append(os.path.join(os.getcwd(), "train"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

def test_nomad():
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- 1. 初始化模型组件 ---
    
    # 初始化视觉编码器 (Vision Encoder)
    # NoMaD_ViNT 基于 EfficientNet，用于提取观测图像和目标图像的特征
    # context_size=5 表示使用过去5帧作为上下文
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=512,      # 输出特征维度
        context_size=5,             # 上下文帧数
        mha_num_attention_heads=4,  # 多头注意力头数
        mha_num_attention_layers=4, # 注意力层数
        mha_ff_dim_factor=4,        # 前馈网络维度因子
    )
    # 将 BatchNorm 替换为 GroupNorm，通常在 Batch Size 较小时更稳定
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # 初始化噪声预测网络 (Noise Prediction Network)
    # 这是一个条件 UNet (Conditional UNet)，用于扩散过程中的去噪
    # 它接收带噪声的动作序列和视觉特征作为条件，预测添加的噪声
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,                # 动作维度 (例如: 线速度, 角速度)
        global_cond_dim=512,        # 全局条件维度 (来自视觉编码器的输出)
        down_dims=[256, 512, 1024], # 下采样通道数
        cond_predict_scale=True,    # 是否预测缩放因子
    )
    
    # 初始化距离预测网络 (Distance Prediction Network)
    # 用于预测到达目标的拓扑距离（可选辅助任务）
    dist_pred_network = DenseNetwork(embedding_dim=512)

    # 组装完整的 NoMaD 模型
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    ).to(device)

    print("Model initialized successfully!")

    # --- 2. 构造假数据 (Batch size=2) ---
    B = 2
    # 观测数据 (Observations): 
    # 形状: (B, (Context+1)*C, H, W) -> 6帧 * 3通道 = 18通道
    # NoMaD 将当前帧和过去5帧在通道维度上拼接作为输入
    dummy_obs = torch.randn(B, 6 * 3, 85, 64).to(device)
    
    # 目标图像 (Goal Image):
    # 形状: (B, C, H, W) -> 单张 RGB 图像
    dummy_goal = torch.randn(B, 3, 85, 64).to(device)
    
    # 动作序列 (Actions):
    # 形状: (B, Pred_Len, Action_Dim) -> 预测未来8步的动作，每步2维
    dummy_actions = torch.randn(B, 8, 2).to(device)
    
    # 目标掩码 (Goal Mask): 
    # 形状: (B,) 
    # 1 表示有特定目标导航，0 表示无目标（探索模式/图构建模式）
    dummy_mask = torch.ones(B).to(device).long()

    print("Attempting forward pass (Manual Pipeline)...")

    # --- 3. 手动执行训练步骤 (模拟训练循环) ---
    
    # [Step 1]: 视觉编码 (Vision Encoding)
    # 调用模型的 vision_encoder 部分
    # 输入: 观测历史图像栈, 目标图像, 目标掩码
    # 输出: 融合了观测和目标的 512 维特征向量
    obs_encoding = model(
        func_name="vision_encoder", 
        obs_img=dummy_obs, 
        goal_img=dummy_goal, 
        input_goal_mask=dummy_mask
    )
    print(f"Vision Encoding Shape: {obs_encoding.shape}") # 预期: (B, 512)

    # [Step 2]: 扩散过程 (Diffusion Process - Forward Process)
    # 在训练时，我们需要向真实动作中添加噪声，然后让网络预测这个噪声
    
    # 生成随机噪声
    noise = torch.randn_like(dummy_actions).to(device)
    # 随机采样时间步 (Timesteps)，范围 [0, 100)
    timesteps = torch.randint(0, 100, (B,), device=device).long()
    
    # 模拟加噪后的动作 (Noisy Action)
    # 注意：这里是简化版直接相加。真实的 DDPM/DDIM 扩散过程需要根据 alpha/beta schedule 进行加权求和
    noisy_actions = dummy_actions + noise 

    # [Step 3]: 噪声预测 (Noise Prediction - Reverse Process Step)
    # 调用模型的 noise_pred_net 部分
    # 输入: 加噪后的动作, 时间步, 全局条件(视觉特征)
    # 输出: 预测的噪声
    pred_noise = model(
        func_name="noise_pred_net", 
        sample=noisy_actions, 
        timestep=timesteps, 
        global_cond=obs_encoding
    )
    print(f"Predicted Noise Shape: {pred_noise.shape}")

    # [Step 4]: 计算损失 (Loss Calculation)
    # 扩散模型的目标是最小化预测噪声与真实添加噪声之间的均方误差 (MSE)
    loss = torch.nn.functional.mse_loss(pred_noise, noise)
    print(f"Forward pass successful! Mock Loss: {loss.item()}")

if __name__ == "__main__":
    test_nomad()