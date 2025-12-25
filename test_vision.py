import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import sys
import os
import matplotlib.pyplot as plt

# --- 引入项目依赖 ---
sys.path.append(os.path.join(os.getcwd(), "train"))
sys.path.append(os.path.join(os.getcwd(), "deployment/src"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# ==========================================
# 0. 配置与常量
# ==========================================
# 从 data_config.yaml 获取的统计数据
ACTION_STATS = {
    'min': np.array([-2.5, -4]), # [min_dx, min_dy]
    'max': np.array([5, 4])      # [max_dx, max_dy]
}

# ==========================================
# 1. 模型加载函数
# ==========================================
def load_model_manual(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"错误: 找不到权重文件 {ckpt_path}")
        sys.exit(1)
    
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=256,  
        context_size=3,         
        mha_num_attention_heads=4, 
        mha_num_attention_layers=4, 
        mha_ff_dim_factor=4
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2, 
        global_cond_dim=256,       
        down_dims=[64, 128, 256],  
        cond_predict_scale=False   
    )
    
    dist_pred_network = DenseNetwork(embedding_dim=256)
    
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    ).to(device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model

# ==========================================
# 2. 图像预处理 (优化版: 顶部裁剪 + 缩放)
# ==========================================
def process_image(image_path, size=(85, 64)):
    img = Image.open(image_path).convert("RGB")
    
    # [优化策略] 顶部裁剪 + 缩放 (Top-Crop + Squash)
    # 
    # 之前的尝试:
    # 1. Squash (全图缩放): test1 直行，test2 右转，但 test3/4 转弯不明显。
    # 2. Center/Bottom Crop: 导致 test1 严重跑偏。
    # 3. Top-Crop 20%: test1 跑偏 (Lateral +3.2m)，test2 右转，test4 左转。
    #
    # 问题核心:
    # test1 (走廊) 对"消失点"极其敏感。任何导致消失点偏离中心的裁剪都会让模型以为走歪了。
    # test3/4 (转弯) 需要看到地面的障碍物。
    #
    # 最终方案: 极轻微的顶部裁剪 (10%) + Squash
    # - 只切掉最顶部的灯光/天花板 (10%)，最大程度保留消失点位置。
    # - 使用 Squash 压缩，让地面特征在垂直方向上更紧凑，帮助模型看到"更远"的地面。
    
    w, h = img.size
    
    # 仅裁剪掉顶部 10% (微调)
    top_crop = int(h * 0.10)
    img = img.crop((0, top_crop, w, h))
    
    # 缩放到模型输入尺寸
    img = img.resize(size) 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

# ==========================================
# 3. 数据反归一化与处理
# ==========================================
def unnormalize_data(ndata, stats):
    # ndata: (B, T, 2) range [-1, 1]
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def process_action(naction):
    # naction: (B, T, 2)
    action_cpu = naction.cpu().numpy()
    
    # 1. 反归一化
    unnormalized_deltas = unnormalize_data(action_cpu, ACTION_STATS)
    
    # 2. 累加 (CumSum)
    waypoints = np.cumsum(unnormalized_deltas, axis=1)
    
    return waypoints[0] # 返回第一个样本的路径 (8, 2)

# ==========================================
# 4. 可视化函数
# ==========================================
def visualize_result(img_path, waypoints, output_name="result_fixed.jpg"):
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片用于可视化")
        return
    img = cv2.resize(img, (640, 480)) 
    h, w, _ = img.shape
    
    start_x, start_y = w // 2, h - 20
    prev_pt = (start_x, start_y)
    
    # 调整缩放比例: 控制轨迹在图上的显示大小
    # 值越小，轨迹画得越短/越密集；值越大，轨迹画得越长/越分散
    SCALE = 20 
    
    cv2.circle(img, prev_pt, 8, (0, 0, 255), -1) 
    
    print(f"\n{'Step':<5} {'Forward (X)':<15} {'Lateral (Y)':<15}")
    print("-" * 40)
    
    for i in range(len(waypoints)):
        # 0号位是前进(Forward), 1号位是横向(Lateral)
        forward_dist = waypoints[i][0] 
        lateral_dist = waypoints[i][1]
        
        print(f"{i:<5} {forward_dist:<15.4f} {lateral_dist:<15.4f}")
        
        # 映射到图像坐标 (增加边界保护)
        # [可视化优化] 
        # 为了让微小的转向趋势在图中更明显，我们对横向(Lateral)进行了额外的放大。
        # 同时减小纵向(Forward)的缩放，防止跑出图片上边界。
        LATERAL_BOOST = 4.0 
        img_x = int(start_x - lateral_dist * SCALE * LATERAL_BOOST) 
        img_y = int(start_y - forward_dist * SCALE) 
        
        # Clip to image bounds
        img_x = max(0, min(img_x, w-1))
        img_y = max(0, min(img_y, h-1))
        
        curr_pt = (img_x, img_y)
        
        cv2.circle(img, curr_pt, 5, (0, 255, 0), -1) 
        cv2.line(img, prev_pt, curr_pt, (0, 255, 0), 2)
        prev_pt = curr_pt
        
    cv2.imwrite(output_name, img)
    print(f"\n[完成] 可视化结果已保存为 {output_name}")

# ==========================================
# 5. 主测试逻辑
# ==========================================
def test_vision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = "deployment/model_weights/nomad.pth"
    
    try:
        model = load_model_manual(CKPT_PATH, device)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 测试所有图片
    test_images = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]
    
    print("\n=== 开始批量测试 (探索模式 Mask=1) ===")
    print("预期: 遇到转角时，轨迹应向空旷区域弯曲")
    
    mask_explore = torch.ones(1).long().to(device) 

    for img_name in test_images:
        if not os.path.exists(img_name):
            print(f"跳过 {img_name}: 文件不存在")
            continue
            
        print(f"\n--- 测试图片: {img_name} ---")
        try:
            img_tensor = process_image(img_name).to(device)
            
            # 构造输入: 4帧堆叠
            obs_batch = torch.cat([img_tensor] * 4, dim=0).unsqueeze(0) 
            goal_batch = img_tensor.unsqueeze(0) 
            
            out_name = f"result_{img_name}"
            run_inference(model, obs_batch, goal_batch, mask_explore, device, img_name, out_name)
            
        except Exception as e:
            print(f"处理 {img_name} 时出错: {e}")

def run_inference(model, obs_batch, goal_batch, mask_batch, device, img_path, out_name):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=10, # 使用 yaml 中的 10 步
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    noise_scheduler.set_timesteps(10)
    
    with torch.no_grad():
        obs_cond = model("vision_encoder", obs_img=obs_batch, goal_img=goal_batch, input_goal_mask=mask_batch)
        
        # [新增] 打印距离预测
        dist_input = obs_cond.squeeze(1) if len(obs_cond.shape) == 3 else obs_cond
        dist_pred = model("dist_pred_net", obsgoal_cond=dist_input)
        dist_val = dist_pred.item()
        print(f"[{out_name}] 模型预测的剩余距离 (Temporal Distance): {dist_val:.4f}")

        # === 修复逻辑: 距离阈值停车 ===
        STOP_THRESHOLD = 6.0 
        
        if dist_val < STOP_THRESHOLD:
            print(f"   -> [自动控制] 距离 {dist_val:.4f} < 阈值 {STOP_THRESHOLD}，判定已到达目标。强制停车 (输出零轨迹)。")
            waypoints = np.zeros((8, 2))
            visualize_result(img_path, waypoints, out_name)
            return

        naction = torch.randn((1, 8, 2), device=device)
        
        for k in noise_scheduler.timesteps:
            noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_scheduler.step(noise_pred, k, naction).prev_sample
            
    # 正确的后处理
    waypoints = process_action(naction)
    visualize_result(img_path, waypoints, out_name)

if __name__ == "__main__":
    test_vision()
