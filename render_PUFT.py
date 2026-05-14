#
# PUFT Render Script
# 渲染热图像、不确定性图和温度图
#

import torch
from scene import Scene_1, Scene_2
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.physics_utils import render_uncertainty_map, render_temperature_map


def apply_colormap(tensor_2d, colormap='iron'):
    """
    将2D张量应用伪彩色映射，生成3通道彩色图像
    
    Args:
        tensor_2d: (H, W) 归一化到[0,1]的张量
        colormap: 颜色映射类型
        
    Returns:
        (3, H, W) 彩色张量
    """
    t = tensor_2d.clamp(0, 1).cpu().numpy()
    
    # Iron colormap (类似热像仪显示)
    r = np.clip(1.5 * t - 0.25, 0, 1)
    g = np.clip(1.5 * t - 0.75, 0, 1)
    b = np.clip(3.0 * t, 0, 0.5) * (1 - t) + np.clip(t - 0.5, 0, 0.5) * 2
    
    colored = np.stack([r, g, b], axis=0)
    return torch.from_numpy(colored).float()


def render_set_puft(model_path, name, iteration, views, gaussians, pipeline, background, render_extra=True):
    """
    渲染图像集，包含热图像、不确定性图和温度图
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    # PUFT额外输出目录
    if render_extra and gaussians.puft_enabled:
        uncertainty_path = os.path.join(model_path, name, "ours_{}".format(iteration), "uncertainty")
        temperature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "temperature")
        makedirs(uncertainty_path, exist_ok=True)
        makedirs(temperature_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 标准热图像渲染
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # PUFT: 渲染不确定性图和温度图
        if render_extra and gaussians.puft_enabled:
            # 不确定性图
            unc_map = render_uncertainty_map(view, gaussians, pipeline, background, render)
            # 归一化到[0,1]用于可视化
            unc_norm = (unc_map - unc_map.min()) / (unc_map.max() - unc_map.min() + 1e-8)
            unc_colored = apply_colormap(unc_norm.detach())
            torchvision.utils.save_image(unc_colored, os.path.join(uncertainty_path, '{0:05d}'.format(idx) + ".png"))
            
            # 温度图
            temp_map = render_temperature_map(view, gaussians, pipeline, background, render)
            temp_norm = (temp_map - gaussians.T_min) / (gaussians.T_max - gaussians.T_min + 1e-8)
            temp_norm = temp_norm.clamp(0, 1)
            temp_colored = apply_colormap(temp_norm.detach())
            torchvision.utils.save_image(temp_colored, os.path.join(temperature_path, '{0:05d}'.format(idx) + ".png"))


def render_sets_puft(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                     skip_train: bool, skip_test: bool):
    """渲染所有集合"""
    with torch.no_grad():
        # RGB场景
        gaussians_rgb = GaussianModel(dataset.sh_degree)
        scene_rgb = Scene_1(dataset, gaussians_rgb, load_iteration=iteration, shuffle=False)
        
        # 热场景
        gaussians_thermal = GaussianModel(dataset.sh_degree)
        scene_thermal = Scene_2(dataset, gaussians_thermal, load_iteration=iteration, shuffle=False)
        
        # 检查是否有PUFT属性（通过检测ply中的额外字段或尝试加载）
        # 简单方案：始终初始化PUFT属性以启用渲染
        gaussians_thermal.puft_enabled = True
        gaussians_thermal.T_min = 0.0
        gaussians_thermal.T_max = 100.0
        # 如果没有保存的PUFT属性，不确定性和温度将返回默认值

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set_puft(dataset.model_path, "rgb_train", scene_rgb.loaded_iter, 
                           scene_rgb.getTrainCameras(), gaussians_rgb, pipeline, background, 
                           render_extra=False)
            render_set_puft(dataset.model_path, "thermal_train", scene_thermal.loaded_iter, 
                           scene_thermal.getTrainCameras(), gaussians_thermal, pipeline, background,
                           render_extra=True)

        if not skip_test:
            render_set_puft(dataset.model_path, "rgb_test", scene_rgb.loaded_iter, 
                           scene_rgb.getTestCameras(), gaussians_rgb, pipeline, background,
                           render_extra=False)
            render_set_puft(dataset.model_path, "thermal_test", scene_thermal.loaded_iter, 
                           scene_thermal.getTestCameras(), gaussians_thermal, pipeline, background,
                           render_extra=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="PUFT Rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--puft_T_min", type=float, default=0.0)
    parser.add_argument("--puft_T_max", type=float, default=100.0)
    args = get_combined_args(parser)
    print("Rendering (PUFT) " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets_puft(model.extract(args), args.iteration, pipeline.extract(args), 
                     args.skip_train, args.skip_test)
