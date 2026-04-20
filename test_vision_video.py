import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import sys
import os
from collections import deque # 用于维护历史帧队列

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

# 绘图配置
SCALE = 20 
LATERAL_BOOST = 4.0
COLORS = [
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 255, 255),  # 黄
    (255, 0, 255),  # 品红
    (255, 128, 0),  # 橙
]

# ==========================================
# 1. 模型加载 (保持不变)
# ==========================================
def load_model_manual(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"错误: 找不到权重文件 {ckpt_path}")
        sys.exit(1)
    
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=256, context_size=3, mha_num_attention_heads=4, 
        mha_num_attention_layers=4, mha_ff_dim_factor=4
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2, global_cond_dim=256, down_dims=[64, 128, 256], cond_predict_scale=False
    )
    
    dist_pred_network = DenseNetwork(embedding_dim=256)
    
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    ).to(device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model

# ==========================================
# 2. 视频帧预处理 (适配 CV2 格式)
# ==========================================
def transform_frame(frame, size=(85, 64)):
    """
    将 OpenCV 的帧 (BGR numpy array) 转换为模型输入的 Tensor
    包含 Top-Crop 和 Squash
    """
    # CV2 BGR -> PIL RGB
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    w, h = img.size
    
    # 1. Top-Crop 10%
    top_crop = int(h * 0.10)
    img = img.crop((0, top_crop, w, h))
    
    # 2. Squash to (85, 64)
    img = img.resize(size)
    
    # 3. ToTensor + Normalize
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(img)

# ==========================================
# 3. 动作后处理 (Batch版)
# ==========================================
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def process_action_batch(naction_batch):
    # naction_batch: (B, T, 2)
    action_cpu = naction_batch.detach().cpu().numpy()
    unnormalized_deltas = unnormalize_data(action_cpu, ACTION_STATS)
    # cumsum along time dimension (axis=1)
    waypoints = np.cumsum(unnormalized_deltas, axis=1)
    return waypoints # (B, 8, 2)

# ==========================================
# 4. 绘图工具 (直接在 Frame 上画)
# ==========================================
def draw_overlays(frame, waypoints_batch, dist_val=None):
    """
    在当前帧上绘制多条预测轨迹和距离信息
    """
    h, w, _ = frame.shape
    start_x, start_y = w // 2, h - 20
    
    # 绘制起点
    cv2.circle(frame, (start_x, start_y), 8, (0, 0, 255), -1)
    
    # 绘制距离提示
    if dist_val is not None:
        text = f"Dist: {dist_val:.2f}"
        color = (0, 255, 0) if dist_val > 6.0 else (0, 0, 255) # 大于6安全，小于6危险
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if dist_val < 6.0:
            cv2.putText(frame, "STOP", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 绘制所有样本轨迹
    for idx, waypoints in enumerate(waypoints_batch):
        color = COLORS[idx % len(COLORS)]
        prev_pt = (start_x, start_y)
        
        for i in range(len(waypoints)):
            # 坐标转换: x前y左 -> 图x图y
            forward = waypoints[i][0]
            lateral = waypoints[i][1]
            
            img_x = int(start_x - lateral * SCALE * LATERAL_BOOST)
            img_y = int(start_y - forward * SCALE)
            
            # 边界截断
            img_x = max(0, min(img_x, w-1))
            img_y = max(0, min(img_y, h-1))
            
            curr_pt = (img_x, img_y)
            
            cv2.circle(frame, curr_pt, 2, color, -1)
            cv2.line(frame, prev_pt, curr_pt, color, 2)
            prev_pt = curr_pt
            
    return frame

# ==========================================
# 5. 核心视频处理逻辑
# ==========================================
def process_video(video_path, model, device, output_path="output_video.mp4", num_samples=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    # 获取视频属性
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码，兼容性更好
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 检查 VideoWriter 是否成功初始化
    if not out.isOpened():
        print(f"错误: 无法创建视频写入器")
        print(f"尝试使用 AVI 格式...")
        # 尝试 AVI 格式作为备选
        output_path = output_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"错误: 仍然无法创建视频写入器，退出")
            cap.release()
            return
        else:
            print(f"成功! 将输出为: {output_path}")
    else:
        print(f"VideoWriter 初始化成功，输出: {output_path}")
    
    # 调度器初始化
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    noise_scheduler.set_timesteps(10)
    
    # --- 关键：上下文队列 (Context Queue) ---
    # NoMaD需要过去3帧+当前1帧 = 4帧
    # 队列中存储的是处理好的 Tensor (3, 85, 64)
    context_queue = deque(maxlen=4)
    
    # 探索模式 Mask (全1)
    # 因为我们要一次生成 num_samples 个轨迹，Mask 需要扩展到 Batch 维度
    mask_batch = torch.ones(num_samples).long().to(device)

    print(f"\n=== 开始处理视频: {video_path} ===")
    print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. 预处理当前帧 -> Tensor (1, 3, 85, 64)
        curr_frame_tensor = transform_frame(frame).unsqueeze(0).to(device)
        
        # 2. 更新上下文队列
        if len(context_queue) == 0:
            # 视频刚开始，没有历史，就把第一帧复制4次填满队列
            for _ in range(4):
                context_queue.append(curr_frame_tensor)
        else:
            # 正常运行，加入新帧，自动挤出最旧帧
            context_queue.append(curr_frame_tensor)
            
        # 3. 构造模型输入 Batch
        # 将队列中的4帧在通道维度拼接: (1, 12, 85, 64)
        obs_tensor = torch.cat(list(context_queue), dim=1)
        
        # 扩展到 num_samples 大小以进行批量推理: (B, 12, 85, 64)
        obs_batch = obs_tensor.repeat(num_samples, 1, 1, 1)
        # Goal 图像在探索模式下不重要，给一个全零或者当前帧均可，这里给当前帧
        goal_batch = curr_frame_tensor.repeat(num_samples, 1, 1, 1)

        # 4. 执行推理 (Batch Parallel)
        with torch.no_grad():
            # A. 视觉编码
            obs_cond = model("vision_encoder", obs_img=obs_batch, goal_img=goal_batch, input_goal_mask=mask_batch)
            
            # B. 距离预测 (取第一个样本即可，因为大家都一样)
            dist_input = obs_cond[0].unsqueeze(0).squeeze(1) if len(obs_cond.shape) == 3 else obs_cond[0].unsqueeze(0)
            dist_pred = model("dist_pred_net", obsgoal_cond=dist_input)
            dist_val = dist_pred.item()
            
            # C. 扩散去噪 (Batch size = num_samples)
            # 自动停车逻辑
            if dist_val < 6.0: # 距离太近，输出零轨迹
                waypoints_batch = np.zeros((num_samples, 8, 2))
            else:
                # 初始化噪声动作 (B, 8, 2)
                naction = torch.randn((num_samples, 8, 2), device=device)
                
                # 循环去噪
                for k in noise_scheduler.timesteps:
                    noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
                    naction = noise_scheduler.step(noise_pred, k, naction).prev_sample
                
                # 后处理
                waypoints_batch = process_action_batch(naction)
        
        # 5. 绘制并保存
        draw_overlays(frame, waypoints_batch, dist_val)
        out.write(frame)
        
        # 打印进度
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} | Dist: {dist_val:.2f}")
            
        frame_idx += 1

    cap.release()
    out.release()
    print(f"\n[完成] 视频已保存至: {output_path}")

# ==========================================
# 6. 主入口
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = "deployment/model_weights/nomad.pth"
    
    # 1. 加载模型
    try:
        model = load_model_manual(CKPT_PATH, device)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        sys.exit(1)
        
    # 2. 设置视频路径 (请修改这里为你真正的视频文件路径)
    INPUT_VIDEO = "test_video_corri.mp4"   # <--- 修改这里！
    OUTPUT_VIDEO = "corridor_result.mp4"
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"错误: 找不到输入视频文件 {INPUT_VIDEO}")
        print("请将代码中的 INPUT_VIDEO 变量修改为你的视频文件路径。")
    else:
        # 3. 开始处理
        # num_samples=5 表示同时生成5条轨迹，对应5种颜色
        process_video(INPUT_VIDEO, model, device, OUTPUT_VIDEO, num_samples=5)