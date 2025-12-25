import torch
import sys
import os
import yaml
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# --- 路径设置 ---
# 确保能导入 train 文件夹下的模块
sys.path.append(os.path.join(os.getcwd(), "train"))
# 确保能导入 deployment/src 下的模块 (如 utils)
sys.path.append(os.path.join(os.getcwd(), "deployment/src"))

from vint_train.models.nomad.nomad import NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad import DenseNetwork

# 简单的加载模型函数 (仿照 utils.load_model 但简化)
def load_model_manual(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # --- 修正后的模型配置 (完全匹配官方 nomad.pth) ---
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=256,  # 视觉特征维度
        context_size=3,         # [重要修正] 上下文长度改为 3
        mha_num_attention_heads=4, 
        mha_num_attention_layers=4, 
        mha_ff_dim_factor=4
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2, 
        global_cond_dim=256,       # 必须与 visual_encoder 输出一致
        down_dims=[64, 128, 256],  # [重要修正] 使用小模型架构
        cond_predict_scale=False   # [重要修正] 官方权重不预测 Scale
    )
    
    dist_pred_network = DenseNetwork(embedding_dim=256)
    
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    ).to(device)
    
    # 加载权重
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model

def test_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- 配置参数 ---
    CKPT_PATH = "deployment/model_weights/nomad.pth" 
    CONTEXT_SIZE = 3  # [重要修正] 必须与模型一致
    NUM_DIFFUSION_ITERS = 10 
    
    # 1. 加载模型
    try:
        model = load_model_manual(CKPT_PATH, device)
    except RuntimeError as e:
        print(f"仍然报错: {e}")
        return

    # 2. 调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # 3. 构造假输入 (注意 Context Size 变了)
    B = 1 
    # 通道数 = (Context+1) * 3 = (3+1)*3 = 12
    dummy_obs = torch.randn(B, (CONTEXT_SIZE + 1) * 3, 85, 64).to(device)
    dummy_goal = torch.randn(B, 3, 85, 64).to(device)
    dummy_mask = torch.zeros(B).long().to(device)

    print("Starting Inference Loop...")
    
    with torch.no_grad():
        # [Step A]: 视觉编码
        obs_cond = model(
            func_name="vision_encoder", 
            obs_img=dummy_obs, 
            goal_img=dummy_goal, 
            input_goal_mask=dummy_mask
        )
        print(f"Context Shape: {obs_cond.shape}") # 应该是 (1, 256)
        
        # [Step B]: 初始化噪声动作
        noisy_action = torch.randn((B, 8, 2), device=device)
        naction = noisy_action

        # [Step C]: 设置调度器
        noise_scheduler.set_timesteps(NUM_DIFFUSION_ITERS)

        # [Step D]: 去噪循环
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                func_name='noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
            
        print("Inference finished!")
        print("Sample Action (First 3 waypoints):")
        print(naction[0, :3, :].cpu().numpy())

if __name__ == "__main__":
    test_inference()