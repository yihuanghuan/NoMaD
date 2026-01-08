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
# 2. 图像预处理 (Top-Crop + Squash)
# ==========================================
def process_image(image_path, size=(85, 64)):
    img = Image.open(image_path).convert("RGB")
    
    w, h = img.size
    
    # 仅裁剪掉顶部 10%
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
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def process_action(naction):
    action_cpu = naction.cpu().numpy()
    unnormalized_deltas = unnormalize_data(action_cpu, ACTION_STATS)
    waypoints = np.cumsum(unnormalized_deltas, axis=1)
    return waypoints[0]

# ==========================================
# 4. 可视化函数
# ==========================================
def visualize_result(obs_path, goal_path, waypoints, output_name):
    """
    修改后的可视化：在结果图中同时显示当前观测图（左）和目标图（右），
    并将轨迹绘制在当前观测图上。
    obs_path: 当前（最新）观测图的路径
    goal_path: 目标图的路径
    """
    # 读取当前观测图（最新的那一帧）
    img_obs = cv2.imread(obs_path)
    if img_obs is None:
        print(f"无法读取观测图片: {obs_path}")
        return
    img_obs = cv2.resize(img_obs, (640, 480))
    
    # 读取目标图
    img_goal = cv2.imread(goal_path)
    if img_goal is None:
        print(f"无法读取目标图片: {goal_path}")
        return
    img_goal = cv2.resize(img_goal, (640, 480))
    
    # === 绘制轨迹到观测图上 ===
    h, w, _ = img_obs.shape
    start_x, start_y = w // 2, h - 20
    prev_pt = (start_x, start_y)
    SCALE = 20 
    
    cv2.circle(img_obs, prev_pt, 8, (0, 0, 255), -1) 
    
    print(f"\n{'Step':<5} {'Forward (X)':<15} {'Lateral (Y)':<15} {'说明':<30}")
    print("-" * 70)
    
    for i in range(len(waypoints)):
        forward_dist = waypoints[i][0] 
        lateral_dist = waypoints[i][1]
        
        # 坐标约定说明：
        # lateral_dist > 0 表示机器人坐标系下的 +Y 方向
        # 根据标准右手坐标系（X前，Y左，Z上），dy > 0 表示向左
        direction = "左转" if lateral_dist > 0 else "右转" if lateral_dist < 0 else "直行"
        
        print(f"{i:<5} {forward_dist:<15.4f} {lateral_dist:<15.4f} {direction:<30}")
        
        # 映射到图像坐标系
        # 图像坐标：x向右，y向下
        # 机器人坐标：x向前，y向左
        # 所以需要: img_x = start_x - lateral_dist (左是负x方向)
        LATERAL_BOOST = 4.0 
        img_x = int(start_x - lateral_dist * SCALE * LATERAL_BOOST) 
        img_y = int(start_y - forward_dist * SCALE) 
        
        img_x = max(0, min(img_x, w-1))
        img_y = max(0, min(img_y, h-1))
        
        curr_pt = (img_x, img_y)
        cv2.circle(img_obs, curr_pt, 5, (0, 255, 0), -1) 
        cv2.line(img_obs, prev_pt, curr_pt, (0, 255, 0), 2)
        prev_pt = curr_pt
        
    # === 拼接图片 (左边是当前视野+轨迹，右边是目标) ===
    # 添加文字标签
    cv2.putText(img_obs, "Current Observation & Trajectory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img_goal, "Goal Image", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # 水平拼接
    combined_img = np.hstack((img_obs, img_goal))
    
    cv2.imwrite(output_name, combined_img)
    print(f"\n[完成] 可视化结果已保存为 {output_name}")

# ==========================================
# 5. 主测试逻辑 (已修改为 Navigation 模式)
# ==========================================
def test_vision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = "deployment/model_weights/nomad.pth"
    
    try:
        model = load_model_manual(CKPT_PATH, device)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    input_dir = "test_img"
    output_dir = "result_img_nav" # 结果保存到新文件夹
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # [关键修改] 定义导航任务: (观测图1, 观测图2, 目标图)
    # 观测图按时间顺序：obs1 在前，obs2 在后（最新）
    # ==========================================
    navigation_tasks = [
        # (历史观测, 当前观测, 目标)
        ("test_ob.jpg", "test_ob.jpg", "test_goal1.jpg"), 
        ("test_ob.jpg", "test_ob.jpg", "test_goal2.jpg"), 
        # 你可以添加更多任务...
    ]
    
    print("\n=== 开始批量测试 (导航模式 Mask=0) ===")
    print("预期: 输入连续两帧观测图和目标图，规划从当前位置到目标的路径")
    print("轨迹将绘制在最新的观测图上")
    
    # [关键修改] 设置 Mask = 0，激活 Goal Navigation
    mask_navigation = torch.zeros(1).long().to(device)

    for obs1_name, obs2_name, goal_name in navigation_tasks:
        obs1_path = os.path.join(input_dir, obs1_name)
        obs2_path = os.path.join(input_dir, obs2_name)
        goal_path = os.path.join(input_dir, goal_name)
        
        if not os.path.exists(obs1_path) or not os.path.exists(obs2_path) or not os.path.exists(goal_path):
            print(f"跳过任务: 文件不存在")
            continue
            
        print(f"\n--- 导航任务: [{obs1_name}, {obs2_name}] -> {goal_name} ---")
        try:
            # 1. 处理两张连续观测图像
            obs1_tensor = process_image(obs1_path).to(device)  # 历史帧
            obs2_tensor = process_image(obs2_path).to(device)  # 当前帧（最新）
            
            # 构造输入: context_size=3，需要4帧 (过去3帧+当前1帧)
            # 这里用 obs1 填充历史，obs2 作为当前帧
            obs_batch = torch.cat([
                obs1_tensor,  # t-3
                obs1_tensor,  # t-2
                obs1_tensor,  # t-1
                obs2_tensor   # t (当前帧)
            ], dim=0).unsqueeze(0)  # (1, 12, H, W)
            
            # 2. 处理目标图像 (Goal)
            goal_tensor = process_image(goal_path).to(device)
            goal_batch = goal_tensor.unsqueeze(0) # (1, 3, H, W)
            
            # 3. 定义输出文件名
            out_name = f"result_nav_{obs2_name.split('.')[0]}_to_{goal_name.split('.')[0]}.jpg"
            out_path = os.path.join(output_dir, out_name)
            
            # 4. 执行推理（轨迹绘制在最新观测图 obs2 上）
            run_inference(model, obs_batch, goal_batch, mask_navigation, device, obs2_path, goal_path, out_path)
            
        except Exception as e:
            print(f"处理任务时出错: {e}")
            import traceback
            traceback.print_exc()

def run_inference(model, obs_batch, goal_batch, mask_batch, device, obs_path, goal_path, out_name):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=10, 
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    noise_scheduler.set_timesteps(10)
    
    with torch.no_grad():
        # [关键] 传入 goal_img 和 Mask=0
        obs_cond = model("vision_encoder", obs_img=obs_batch, goal_img=goal_batch, input_goal_mask=mask_batch)
        
        # 打印距离预测
        dist_input = obs_cond.squeeze(1) if len(obs_cond.shape) == 3 else obs_cond
        dist_pred = model("dist_pred_net", obsgoal_cond=dist_input)
        dist_val = dist_pred.item()
        print(f"模型预测的剩余距离 (Temporal Distance): {dist_val:.4f}")
        
        # [DEBUG] 打印视觉编码的特征向量（前10个值）
        print(f"[DEBUG] obs_cond 前10个值: {obs_cond[0, :10].cpu().numpy()}")

        # 距离阈值停车逻辑
        STOP_THRESHOLD = 0.5 # 导航模式下阈值可能需要调小一点，或者保持 6.0 看情况
        # 注意: 6.0 是 temporal distance (时间步)，不是米。
        
        if dist_val < STOP_THRESHOLD:
            print(f"   -> [自动控制] 距离 {dist_val:.4f} < 阈值，判定已到达目标。")
            waypoints = np.zeros((8, 2))
            visualize_result(obs_path, goal_path, waypoints, out_name)
            return

        naction = torch.randn((1, 8, 2), device=device)
        
        for k in noise_scheduler.timesteps:
            noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_scheduler.step(noise_pred, k, naction).prev_sample
        
        # [DEBUG] 打印模型原始输出（归一化后的动作，范围 [-1, 1]）
        print(f"[DEBUG] naction (normalized, [-1,1]): {naction[0].cpu().numpy()}")
            
    waypoints = process_action(naction)
    
    # [DEBUG] 打印反归一化后的deltas和累积waypoints
    action_cpu = naction.cpu().numpy()
    unnormalized_deltas = unnormalize_data(action_cpu, ACTION_STATS)
    print(f"[DEBUG] unnormalized deltas: {unnormalized_deltas[0]}")
    print(f"[DEBUG] cumsum waypoints: {waypoints}")
    
    visualize_result(obs_path, goal_path, waypoints, out_name)

if __name__ == "__main__":
    test_vision()