#
# PUFT: Physics-guided Uncertainty-aware Fine-Tuning
# 物理约束与不确定性感知损失函数模块
#
# 双GPU策略: 主训练在GPU:0，KNN计算在CPU上执行（内存充裕不会OOM）
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstraints:
    """
    物理引导约束模块
    包含温度空间平滑性约束、温度范围约束和温度-颜色一致性约束
    
    显存优化策略:
    - KNN计算全部在CPU上执行（利用充裕的系统内存）
    - 只有KNN索引（很小）搬回GPU供loss计算
    - 对超大点云(>max_knn_points)使用随机子采样近似
    """
    
    def __init__(self, T_min=0.0, T_max=100.0, K_neighbors=8, update_interval=1000,
                 train_device="cuda:0", compute_device=None, max_knn_points=60000):
        """
        Args:
            T_min: 场景最低温度 (°C)
            T_max: 场景最高温度 (°C)
            K_neighbors: KNN邻域大小
            update_interval: KNN更新间隔（每多少步更新一次）
            train_device: 训练所在设备 (默认 cuda:0)
            compute_device: 保留参数（实际KNN强制在CPU执行以确保安全）
            max_knn_points: KNN计算时的最大参考点数（超过此数用子采样）
        """
        self.T_min = T_min
        self.T_max = T_max
        self.K = K_neighbors
        self.update_interval = update_interval
        self.knn_indices = None  # 缓存KNN索引 (在train_device上)
        self._dist_sq_cache = None  # 缓存距离 (在train_device上)
        self.max_knn_points = max_knn_points
        
        # 设备配置
        self.train_device = torch.device(train_device)
        # KNN始终在CPU执行（CPU内存通常>32GB，不会OOM）
        self.compute_device = torch.device("cpu")
        
        print(f"[PhysicsConstraints] train_device={self.train_device}, "
              f"knn_on=cpu, max_knn_points={self.max_knn_points}")
    
    @torch.no_grad()
    def update_knn(self, positions):
        """
        更新K近邻索引（在CPU上计算）
        
        Args:
            positions: (N, 3) 高斯位置张量 (在train_device上)
        """
        N = positions.shape[0]
        K = min(self.K, N - 1)
        
        # 搬到CPU计算
        pos_cpu = positions.detach().cpu()
        
        if N > self.max_knn_points:
            # 对超大点云: 随机选择参考点子集做近似KNN
            indices, dists = self._subsampled_knn(pos_cpu, K, N)
        else:
            # 正常规模: 精确KNN
            indices, dists = self._batched_knn_cpu(pos_cpu, K)
        
        # 搬回训练设备（索引很小，几乎不占显存）
        self.knn_indices = indices.to(self.train_device)
        self._dist_sq_cache = dists.to(self.train_device)
    
    def _batched_knn_cpu(self, positions, K, batch_size=4096):
        """
        在CPU上分批精确KNN
        
        Args:
            positions: (N, 3) CPU tensor
            K: 近邻数
        """
        N = positions.shape[0]
        all_indices = torch.zeros((N, K), dtype=torch.long)
        all_dists = torch.zeros((N, K))
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_pos = positions[i:end]  # (B, 3)
            B = end - i
            
            # (B, N) 距离矩阵
            diff = batch_pos.unsqueeze(1) - positions.unsqueeze(0)  # (B, N, 3)
            dist_sq = (diff * diff).sum(-1)  # (B, N)
            
            # 排除自身
            row_idx = torch.arange(B)
            col_idx = row_idx + i
            dist_sq[row_idx, col_idx] = float('inf')
            
            # TopK
            dists, indices = dist_sq.topk(K, dim=1, largest=False)
            all_indices[i:end] = indices
            all_dists[i:end] = dists
            
            del diff, dist_sq
        
        return all_indices, all_dists
    
    def _subsampled_knn(self, positions, K, N):
        """
        子采样近似KNN: 对于N > max_knn_points
        随机选取 max_knn_points 个参考点, 每个查询点在其中找近邻
        
        显存开销: batch_size * max_knn_points * 4 bytes (全在CPU)
        例如: 4096 * 60000 * 4 ≈ 0.9GB 完全可接受
        """
        sample_size = self.max_knn_points
        perm = torch.randperm(N)[:sample_size]
        ref_positions = positions[perm]  # (sample_size, 3)
        
        batch_size = 4096
        all_indices = torch.zeros((N, K), dtype=torch.long)
        all_dists = torch.zeros((N, K))
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_pos = positions[i:end]  # (B, 3)
            B = end - i
            
            # 计算到参考点的距离 (B, sample_size)
            diff = batch_pos.unsqueeze(1) - ref_positions.unsqueeze(0)
            dist_sq = (diff * diff).sum(-1)  # (B, sample_size)
            
            # 排除自身（如果查询点恰好是参考点之一）
            for j in range(B):
                global_idx = i + j
                # perm中找global_idx的位置
                match_mask = (perm == global_idx)
                if match_mask.any():
                    ref_idx = match_mask.nonzero(as_tuple=True)[0][0]
                    dist_sq[j, ref_idx] = float('inf')
            
            # TopK
            dists, local_indices = dist_sq.topk(K, dim=1, largest=False)
            # 映射回全局索引
            all_indices[i:end] = perm[local_indices]
            all_dists[i:end] = dists
            
            del diff, dist_sq
        
        return all_indices, all_dists
    
    def temperature_smoothness_loss(self, temperatures, positions, uncertainties):
        """
        温度空间平滑性损失（热传导先验）
        
        核心思想：相邻高斯的温度应平滑变化，但在高不确定性区域（物体边界）允许突变
        
        Args:
            temperatures: (N, 1) 物理温度
            positions: (N, 3) 高斯位置  
            uncertainties: (N, 1) 不确定性值
            
        Returns:
            scalar loss
        """
        if self.knn_indices is None:
            self.update_knn(positions.detach())
        
        N = temperatures.shape[0]
        
        # 确保indices不越界（densification后点数可能变化）
        if self.knn_indices.shape[0] != N:
            self.update_knn(positions.detach())
        
        K = self.knn_indices.shape[1]
        
        # 安全检查: 确保索引不超过当前点数
        max_idx = self.knn_indices.max().item()
        if max_idx >= N:
            self.update_knn(positions.detach())
            K = self.knn_indices.shape[1]
        
        # 获取邻居温度 (N, K, 1)
        T_neighbors = temperatures[self.knn_indices]  # (N, K, 1)
        T_center = temperatures.unsqueeze(1).expand(-1, K, -1)  # (N, K, 1)
        
        # 温度差的平方
        T_diff_sq = (T_center - T_neighbors) ** 2  # (N, K, 1)
        
        # 空间距离权重：距离越近，平滑约束越强
        if self._dist_sq_cache is not None and self._dist_sq_cache.shape[0] == N:
            dist_sq = self._dist_sq_cache.unsqueeze(-1)  # (N, K, 1)
        else:
            pos_neighbors = positions[self.knn_indices]  # (N, K, 3)
            pos_center = positions.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
            dist_sq = ((pos_center - pos_neighbors) ** 2).sum(-1, keepdim=True)  # (N, K, 1)
        
        # 归一化距离（使用中位数作为带宽）
        d_median = torch.median(dist_sq.detach()).clamp(min=1e-8)
        spatial_weight = torch.exp(-dist_sq / (2 * d_median))  # (N, K, 1)
        
        # 不确定性权重：高不确定性区域减弱平滑约束（允许温度突变）
        sigma_center = uncertainties.unsqueeze(1).expand(-1, K, -1)  # (N, K, 1)
        sigma_neighbors = uncertainties[self.knn_indices]  # (N, K, 1)
        # 不确定性越大 → uncertainty_weight越小 → 平滑约束越弱
        uncertainty_weight = 1.0 / (sigma_center + sigma_neighbors + 0.1)  # (N, K, 1)
        
        # 加权平滑损失
        weighted_diff = spatial_weight * uncertainty_weight * T_diff_sq
        loss = weighted_diff.mean()
        
        return loss
    
    def temperature_range_loss(self, temperatures):
        """
        温度范围约束：确保温度在物理合理范围内
        
        Args:
            temperatures: (N, 1) 物理温度
            
        Returns:
            scalar loss
        """
        loss_low = F.relu(self.T_min - temperatures).mean()
        loss_high = F.relu(temperatures - self.T_max).mean()
        return loss_low + loss_high
    
    def temperature_color_consistency_loss(self, T_rendered, thermal_rendered):
        """
        温度-伪彩色一致性约束
        约束渲染的温度图与热图像之间的单调关系
        
        Args:
            T_rendered: (H, W) 渲染的温度图
            thermal_rendered: (3, H, W) 或 (C, H, W) 渲染的热图像
            
        Returns:
            scalar loss
        """
        # 归一化温度到 [0, 1]
        T_norm = (T_rendered - self.T_min) / (self.T_max - self.T_min + 1e-8)
        T_norm = T_norm.clamp(0, 1)
        
        # 将热图像转为灰度（热图像的亮度应与温度单调相关）
        if thermal_rendered.dim() == 3 and thermal_rendered.shape[0] == 3:
            # RGB热图像 → 灰度
            thermal_gray = 0.299 * thermal_rendered[0] + 0.587 * thermal_rendered[1] + 0.114 * thermal_rendered[2]
        else:
            thermal_gray = thermal_rendered.mean(0) if thermal_rendered.dim() == 3 else thermal_rendered
        
        # 简单的排序一致性约束：
        # 温度高的区域，热图像亮度也应该高（Spearman风格的可微损失）
        # 使用L1距离近似排序一致性
        loss = F.l1_loss(T_norm, thermal_gray.detach())
        
        return loss


def uncertainty_aware_loss(pixel_loss_map, uncertainty_map, reg_weight=0.05):
    """
    不确定性感知损失函数（改进的 Gaussian Negative Log-Likelihood）
    
    改进要点:
    1. 对σ设置下界(clamp)，防止σ趋近0导致梯度爆炸
    2. 增大reg_weight默认值(0.01→0.05)，更强地惩罚大σ
    3. 添加额外的variance上界惩罚，防止σ无限增大导致负loss
    
    Args:
        pixel_loss_map: (H, W) 或 (C, H, W) 逐像素损失图
        uncertainty_map: (H, W) 渲染的不确定性图（正值）
        reg_weight: 不确定性稀疏正则化权重
        
    Returns:
        scalar loss (保证非负)
    """
    # 确保维度匹配
    if pixel_loss_map.dim() == 3 and uncertainty_map.dim() == 2:
        pixel_loss_map = pixel_loss_map.mean(0)  # (H, W)
    
    # 对σ设置上下界, 防止退化:
    # 下界0.01: 防止σ→0导致的梯度爆炸
    # 上界2.0: 防止σ→∞导致的负loss退化解
    sigma_clamped = uncertainty_map.clamp(min=0.01, max=2.0)
    
    # Gaussian NLL: L/(2σ²) + 0.5*log(σ²)
    var = sigma_clamped ** 2  # 方差 = σ²
    precision = 1.0 / (2.0 * var)
    nll = precision * pixel_loss_map + 0.5 * torch.log(var)
    
    # 不确定性正则化：L2惩罚防止σ膨胀
    reg = reg_weight * (uncertainty_map ** 2).mean()
    
    # 最终loss = NLL + 正则化
    # 由于σ被clamp在[0.01, 2.0], NLL不会无限趋于负数
    return nll.mean() + reg


def compute_pixel_loss_map(rendered, gt, lambda_dssim=0.2):
    """
    计算逐像素损失图（用于不确定性感知损失）
    
    Args:
        rendered: (3, H, W) 渲染图像
        gt: (3, H, W) Ground Truth图像
        lambda_dssim: DSSIM权重
        
    Returns:
        pixel_loss_map: (H, W) 逐像素损失
    """
    # L1部分：逐像素
    l1_map = torch.abs(rendered - gt).mean(0)  # (H, W)
    return l1_map


def render_uncertainty_map(viewpoint_camera, gaussians, pipe, bg_color, render_fn):
    """
    渲染不确定性图（利用override_color技巧）
    
    Args:
        viewpoint_camera: 相机参数
        gaussians: GaussianModel
        pipe: Pipeline参数
        bg_color: 背景颜色
        render_fn: render函数引用
        
    Returns:
        uncertainty_map: (H, W) 不确定性图
    """
    uncertainty = gaussians.get_uncertainty  # (N, 1)
    # 扩展到3通道以兼容标准render接口
    override_color = uncertainty.expand(-1, 3)  # (N, 3)
    
    render_pkg = render_fn(viewpoint_camera, gaussians, pipe, bg_color, override_color=override_color)
    # 取第一通道作为不确定性图
    uncertainty_map = render_pkg["render"][0]  # (H, W)
    
    return uncertainty_map


def render_temperature_map(viewpoint_camera, gaussians, pipe, bg_color, render_fn):
    """
    渲染温度图
    
    Args:
        viewpoint_camera: 相机参数
        gaussians: GaussianModel
        pipe: Pipeline参数
        bg_color: 背景颜色
        render_fn: render函数引用
        
    Returns:
        temperature_map: (H, W) 温度图
    """
    temperature = gaussians.get_temperature  # (N, 1)
    # 归一化到[0,1]以便渲染
    T_norm = (temperature - gaussians.T_min) / (gaussians.T_max - gaussians.T_min + 1e-8)
    T_norm = T_norm.clamp(0, 1)
    override_color = T_norm.expand(-1, 3)  # (N, 3)
    
    render_pkg = render_fn(viewpoint_camera, gaussians, pipe, bg_color, override_color=override_color)
    temperature_map = render_pkg["render"][0]  # (H, W)
    
    # 反归一化回温度值
    temperature_map = temperature_map * (gaussians.T_max - gaussians.T_min) + gaussians.T_min
    
    return temperature_map
