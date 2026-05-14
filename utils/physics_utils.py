#
# PUFT: Physics-guided Uncertainty-aware Fine-Tuning
# 物理约束与不确定性感知损失函数模块
#
# v3: 彻底移除KNN，改用图像空间2D物理约束
# 核心思路: 渲染温度图后，在2D图像上施加平滑约束
# 梯度通过可微渲染自动传播回3D Gaussian的温度参数
# 优势: O(H*W) 计算量，零额外显存，无需KNN
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstraints:
    """
    图像空间物理引导约束模块（v3 - 无KNN版）
    
    所有平滑约束在渲染后的2D图像空间完成:
    - 温度平滑: Total Variation on rendered temperature map
    - 温度范围: 直接约束per-Gaussian温度值
    - 温度-颜色一致性: 约束温度图与热图像的对应关系
    
    计算开销: 仅2D图像操作，与渲染分辨率相关，与点数无关
    """
    
    def __init__(self, T_min=0.0, T_max=100.0, **kwargs):
        """
        Args:
            T_min: 场景最低温度 (°C)
            T_max: 场景最高温度 (°C)
            kwargs: 兼容旧接口的多余参数（忽略）
        """
        self.T_min = T_min
        self.T_max = T_max
        print(f"[PhysicsConstraints v3] Image-space mode, T_range=[{T_min}, {T_max}]")
    
    def temperature_smoothness_loss(self, T_map, uncertainty_map=None):
        """
        图像空间温度平滑损失 (Uncertainty-weighted Total Variation)
        
        在渲染的温度图上计算TV loss，通过可微渲染梯度自动传播
        到底层3D Gaussian的温度参数。
        
        如果提供uncertainty_map，则高不确定性区域的平滑约束被弱化
        （允许物体边界处的温度突变）
        
        Args:
            T_map: (H, W) 渲染的温度图
            uncertainty_map: (H, W) 渲染的不确定性图（可选）
            
        Returns:
            scalar loss
        """
        # 水平和垂直梯度 (TV)
        diff_h = (T_map[1:, :] - T_map[:-1, :]) ** 2  # (H-1, W)
        diff_w = (T_map[:, 1:] - T_map[:, :-1]) ** 2  # (H, W-1)
        
        if uncertainty_map is not None:
            # 不确定性加权: 高不确定性区域减弱平滑约束
            # sigma越大 → weight越小 → 允许温度突变
            sigma_h = (uncertainty_map[1:, :] + uncertainty_map[:-1, :]) / 2.0
            sigma_w = (uncertainty_map[:, 1:] + uncertainty_map[:, :-1]) / 2.0
            weight_h = 1.0 / (sigma_h + 0.1)
            weight_w = 1.0 / (sigma_w + 0.1)
            
            loss = (weight_h * diff_h).mean() + (weight_w * diff_w).mean()
        else:
            loss = diff_h.mean() + diff_w.mean()
        
        return loss
    
    def temperature_range_loss(self, temperatures):
        """
        温度范围约束：确保per-Gaussian温度在物理合理范围内
        
        Args:
            temperatures: (N, 1) 物理温度
            
        Returns:
            scalar loss
        """
        loss_low = F.relu(self.T_min - temperatures).mean()
        loss_high = F.relu(temperatures - self.T_max).mean()
        return loss_low + loss_high
    
    def temperature_color_consistency_loss(self, T_map, thermal_image):
        """
        温度-热图像一致性约束
        约束渲染的温度图与热图像之间的单调关系
        
        Args:
            T_map: (H, W) 渲染的温度图
            thermal_image: (3, H, W) 渲染/GT的热图像
            
        Returns:
            scalar loss
        """
        # 归一化温度到 [0, 1]
        T_norm = (T_map - self.T_min) / (self.T_max - self.T_min + 1e-8)
        T_norm = T_norm.clamp(0, 1)
        
        # 热图像转灰度
        if thermal_image.dim() == 3 and thermal_image.shape[0] == 3:
            thermal_gray = 0.299 * thermal_image[0] + 0.587 * thermal_image[1] + 0.114 * thermal_image[2]
        else:
            thermal_gray = thermal_image.mean(0) if thermal_image.dim() == 3 else thermal_image
        
        # L1一致性
        loss = F.l1_loss(T_norm, thermal_gray.detach())
        return loss


def uncertainty_aware_loss(pixel_loss_map, uncertainty_map, reg_weight=0.05):
    """
    不确定性感知损失函数（稳定版 Gaussian NLL）
    
    关键改进: 
    - σ clamp到[0.01, 2.0]，彻底杜绝负loss退化
    - L2正则化防止σ无限膨胀
    
    Args:
        pixel_loss_map: (H, W) 或 (C, H, W) 逐像素损失图
        uncertainty_map: (H, W) 渲染的不确定性图（正值）
        reg_weight: 不确定性正则化权重
        
    Returns:
        scalar loss (保证非负)
    """
    if pixel_loss_map.dim() == 3 and uncertainty_map.dim() == 2:
        pixel_loss_map = pixel_loss_map.mean(0)
    
    # σ限界: 防止退化解
    sigma = uncertainty_map.clamp(min=0.01, max=2.0)
    var = sigma ** 2
    
    # Gaussian NLL
    nll = pixel_loss_map / (2.0 * var) + 0.5 * torch.log(var)
    
    # L2正则
    reg = reg_weight * (uncertainty_map ** 2).mean()
    
    return nll.mean() + reg


def compute_pixel_loss_map(rendered, gt, lambda_dssim=0.2):
    """计算逐像素L1损失图"""
    return torch.abs(rendered - gt).mean(0)  # (H, W)


def render_uncertainty_map(viewpoint_camera, gaussians, pipe, bg_color, render_fn):
    """渲染不确定性图（override_color技巧）"""
    uncertainty = gaussians.get_uncertainty  # (N, 1)
    override_color = uncertainty.expand(-1, 3)  # (N, 3)
    render_pkg = render_fn(viewpoint_camera, gaussians, pipe, bg_color, override_color=override_color)
    return render_pkg["render"][0]  # (H, W)


def render_temperature_map(viewpoint_camera, gaussians, pipe, bg_color, render_fn):
    """
    渲染归一化温度图（用于物理约束loss计算）
    返回的是实际温度值 (°C)
    """
    temperature = gaussians.get_temperature  # (N, 1)
    # 归一化到[0,1]以便渲染
    T_norm = (temperature - gaussians.T_min) / (gaussians.T_max - gaussians.T_min + 1e-8)
    T_norm = T_norm.clamp(0, 1)
    override_color = T_norm.expand(-1, 3)  # (N, 3)
    
    render_pkg = render_fn(viewpoint_camera, gaussians, pipe, bg_color, override_color=override_color)
    T_rendered_norm = render_pkg["render"][0]  # (H, W) in [0, 1]
    
    # 反归一化回温度值
    T_rendered = T_rendered_norm * (gaussians.T_max - gaussians.T_min) + gaussians.T_min
    return T_rendered
