#
# PUFT: Physics-guided Uncertainty-aware Fine-Tuning for Thermal 3D Gaussian Splatting
# 
# 基于ThermalGaussian的MFTG框架，引入物理约束和不确定性感知的改进方法
# 
# Stage 1: RGB训练（与原始MFTG完全相同）
# Stage 2: PUFT三阶段微调
#   Phase 2a (warmup): 仅不确定性感知损失，学习σ_i
#   Phase 2b (physics): 逐步引入物理约束，启用不确定性引导的densification
#   Phase 2c (refine): 稳定精炼，停止densification
#

import logging
import os
import torch
import time
from random import randint
from utils.loss_utils import l1_loss, ssim, smoothness_loss
from utils.physics_utils import (
    PhysicsConstraints,
    uncertainty_aware_loss,
    compute_pixel_loss_map,
    render_uncertainty_map,
    render_temperature_map,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene_1, Scene_2, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ========================== PUFT 超参数 ==========================
class PUFTConfig:
    """PUFT方法的超参数配置"""
    
    def __init__(self):
        # --- 训练阶段划分 ---
        self.stage1_iterations = 30_000      # Stage 1 (RGB) 迭代数
        self.stage2_iterations = 30_000      # Stage 2 (PUFT) 总迭代数
        self.phase2a_ratio = 0.15            # Phase 2a 占Stage 2的比例 (warmup)
        self.phase2b_ratio = 0.55            # Phase 2b 占Stage 2的比例 (physics)
        # Phase 2c = 1 - 2a - 2b = 0.30     # Phase 2c (refine)
        
        # --- 不确定性相关 ---
        self.uncertainty_lr = 0.005          # 不确定性学习率
        self.temperature_lr = 0.01           # 温度学习率
        self.uncertainty_reg_weight = 0.01   # 不确定性稀疏正则权重
        self.uncertainty_split_threshold = 1.5  # 不确定性触发分裂的阈值
        
        # --- 物理约束相关 ---
        self.lambda_smooth = 0.01            # 温度平滑损失权重
        self.lambda_range = 0.001            # 温度范围损失权重
        self.lambda_color_consist = 0.05     # 温度-颜色一致性权重
        self.K_neighbors = 8                 # KNN邻居数
        self.knn_update_interval = 1000      # KNN更新间隔
        
        # --- 温度范围（根据数据集调整）---
        self.T_min = 0.0                     # 最低温度 (°C)
        self.T_max = 100.0                   # 最高温度 (°C)
        
        # --- 训练策略 ---
        self.smoothness_weight = 0.6         # 原始MFTG的smoothness_loss权重
        self.densify_until_phase = "2b"      # 在哪个phase停止densification
    
    @property
    def phase2a_iters(self):
        return int(self.stage2_iterations * self.phase2a_ratio)
    
    @property
    def phase2b_iters(self):
        return int(self.stage2_iterations * self.phase2b_ratio)
    
    @property
    def phase2c_iters(self):
        return self.stage2_iterations - self.phase2a_iters - self.phase2b_iters


def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from, step, puft_cfg=None):
    """
    训练函数
    step=1: RGB训练（与原始MFTG相同）
    step=2: PUFT热图像微调
    """
    logging.info(f"○ dataset:{dataset.name}, opt:{opt}, pipe:{pipe}, step:{step}, debug_from:{debug_from}\n")
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    if step == 1:
        global scene_temp
        global gaussians
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene_1(dataset, gaussians)
        print("=" * 60)
        print("[PUFT Stage 1] Start training RGB images (same as MFTG)")
        print("=" * 60)
    
    if step == 2:
        scene = Scene_2(dataset, gaussians)
        scene.gaussians = scene_temp.gaussians
        print("=" * 60)
        print("[PUFT Stage 2] Start PUFT thermal fine-tuning")
        print(f"  Phase 2a (warmup):  iters 1 ~ {puft_cfg.phase2a_iters}")
        print(f"  Phase 2b (physics): iters {puft_cfg.phase2a_iters+1} ~ {puft_cfg.phase2a_iters + puft_cfg.phase2b_iters}")
        print(f"  Phase 2c (refine):  iters {puft_cfg.phase2a_iters + puft_cfg.phase2b_iters + 1} ~ {puft_cfg.stage2_iterations}")
        print("=" * 60)
        
        # 初始化PUFT属性
        gaussians.init_puft_attributes(T_min=puft_cfg.T_min, T_max=puft_cfg.T_max)
        
        # 初始化物理约束模块
        physics = PhysicsConstraints(
            T_min=puft_cfg.T_min,
            T_max=puft_cfg.T_max,
            K_neighbors=puft_cfg.K_neighbors,
            update_interval=puft_cfg.knn_update_interval
        )
    
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )

        # Ground Truth
        gt_image = viewpoint_cam.original_image.cuda()
        
        # ============================================================
        # 损失计算
        # ============================================================
        if step == 1:
            # Stage 1: 标准RGB训练（与原始MFTG完全相同）
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        elif step == 2:
            # Stage 2: PUFT三阶段微调
            Ll1 = l1_loss(image, gt_image)
            
            # 确定当前处于哪个Phase
            phase2a_end = puft_cfg.phase2a_iters
            phase2b_end = puft_cfg.phase2a_iters + puft_cfg.phase2b_iters
            
            # 渲染不确定性图
            uncertainty_map = render_uncertainty_map(viewpoint_cam, gaussians, pipe, bg, render)
            
            # 计算逐像素损失图
            pixel_loss_map = compute_pixel_loss_map(image, gt_image, opt.lambda_dssim)
            
            if iteration <= phase2a_end:
                # === Phase 2a: Warmup — 仅不确定性感知 + 原始smoothness ===
                loss_ua = uncertainty_aware_loss(
                    pixel_loss_map, uncertainty_map, 
                    reg_weight=puft_cfg.uncertainty_reg_weight
                )
                smoothloss_thermal = smoothness_loss(image)
                loss = loss_ua + puft_cfg.smoothness_weight * smoothloss_thermal
                
            elif iteration <= phase2b_end:
                # === Phase 2b: 引入物理约束 + 不确定性引导的densification ===
                loss_ua = uncertainty_aware_loss(
                    pixel_loss_map, uncertainty_map,
                    reg_weight=puft_cfg.uncertainty_reg_weight
                )
                smoothloss_thermal = smoothness_loss(image)
                
                # Warmup系数：物理约束逐步增强
                progress_in_2b = (iteration - phase2a_end) / (phase2b_end - phase2a_end)
                warmup = min(progress_in_2b, 1.0)
                
                # 更新KNN（周期性）
                if iteration % puft_cfg.knn_update_interval == 0:
                    physics.update_knn(gaussians.get_xyz.detach())
                
                # 温度平滑损失
                loss_smooth = physics.temperature_smoothness_loss(
                    gaussians.get_temperature,
                    gaussians.get_xyz,
                    gaussians.get_uncertainty
                )
                
                # 温度范围损失
                loss_range = physics.temperature_range_loss(gaussians.get_temperature)
                
                # 温度-颜色一致性
                T_map = render_temperature_map(viewpoint_cam, gaussians, pipe, bg, render)
                loss_color_consist = physics.temperature_color_consistency_loss(
                    T_map, image
                )
                
                loss = (loss_ua 
                        + puft_cfg.smoothness_weight * smoothloss_thermal
                        + warmup * puft_cfg.lambda_smooth * loss_smooth
                        + warmup * puft_cfg.lambda_range * loss_range
                        + warmup * puft_cfg.lambda_color_consist * loss_color_consist)
                
            else:
                # === Phase 2c: 稳定精炼 ===
                loss_ua = uncertainty_aware_loss(
                    pixel_loss_map, uncertainty_map,
                    reg_weight=puft_cfg.uncertainty_reg_weight
                )
                smoothloss_thermal = smoothness_loss(image)
                
                # 物理约束全力施加
                loss_smooth = physics.temperature_smoothness_loss(
                    gaussians.get_temperature,
                    gaussians.get_xyz,
                    gaussians.get_uncertainty
                )
                loss_range = physics.temperature_range_loss(gaussians.get_temperature)
                
                loss = (loss_ua
                        + puft_cfg.smoothness_weight * smoothloss_thermal
                        + puft_cfg.lambda_smooth * loss_smooth
                        + puft_cfg.lambda_range * loss_range)
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                if step == 1:
                    progress_bar.set_postfix({"color_Loss": f"{ema_loss_for_log:.{7}f}"})
                elif step == 2:
                    phase_name = "2a" if iteration <= phase2a_end else ("2b" if iteration <= phase2b_end else "2c")
                    progress_bar.set_postfix({
                        "thermal_Loss": f"{ema_loss_for_log:.{5}f}",
                        "phase": phase_name,
                        "N_pts": gaussians.get_xyz.shape[0]
                    })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, 
                          scene, render, (pipe, background), step)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
                if step == 1 and iteration == opt.iterations:
                    scene_temp = scene

            # Densification
            if step == 1:
                # Stage 1: 标准densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            
            elif step == 2:
                # Stage 2: PUFT不确定性引导的densification（仅在Phase 2b）
                phase2a_end_iter = puft_cfg.phase2a_iters
                phase2b_end_iter = puft_cfg.phase2a_iters + puft_cfg.phase2b_iters
                
                if phase2a_end_iter < iteration <= phase2b_end_iter:
                    # Phase 2b: 启用不确定性引导的densification
                    if iteration < opt.densify_until_iter:
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(
                                opt.densify_grad_threshold, 0.005, 
                                scene.cameras_extent, size_threshold)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # Stage 2结束时，保存不确定性图的统计信息
    if step == 2:
        uncertainty_vals = gaussians.get_uncertainty.detach()
        temperature_vals = gaussians.get_temperature.detach()
        print(f"\n[PUFT] Final Statistics:")
        print(f"  Uncertainty - mean: {uncertainty_vals.mean():.4f}, std: {uncertainty_vals.std():.4f}")
        print(f"  Temperature - mean: {temperature_vals.mean():.2f}, range: [{temperature_vals.min():.2f}, {temperature_vals.max():.2f}]")
        print(f"  Number of Gaussians: {gaussians.get_xyz.shape[0]}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, step):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    prefix = 'color' if step == 1 else 'thermal_puft'
                    tb_writer.add_scalar(config['name'] + f'/{prefix}/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + f'/{prefix}/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + f'/{prefix}/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + f'/{prefix}/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            # PUFT额外日志
            if step == 2 and scene.gaussians.puft_enabled:
                tb_writer.add_scalar('puft/uncertainty_mean', scene.gaussians.get_uncertainty.mean().item(), iteration)
                tb_writer.add_scalar('puft/temperature_mean', scene.gaussians.get_temperature.mean().item(), iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="PUFT Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    # === PUFT 专用参数 ===
    parser.add_argument("--puft_T_min", type=float, default=0.0, help="Scene minimum temperature (°C)")
    parser.add_argument("--puft_T_max", type=float, default=100.0, help="Scene maximum temperature (°C)")
    parser.add_argument("--puft_lambda_smooth", type=float, default=0.01, help="Temperature smoothness loss weight")
    parser.add_argument("--puft_lambda_range", type=float, default=0.001, help="Temperature range loss weight")
    parser.add_argument("--puft_lambda_color", type=float, default=0.05, help="Temperature-color consistency weight")
    parser.add_argument("--puft_uncertainty_lr", type=float, default=0.005, help="Uncertainty learning rate")
    parser.add_argument("--puft_temperature_lr", type=float, default=0.01, help="Temperature learning rate")
    parser.add_argument("--puft_uncertainty_reg", type=float, default=0.01, help="Uncertainty regularization weight")
    parser.add_argument("--puft_K_neighbors", type=int, default=8, help="KNN neighbors for temperature smoothness")
    parser.add_argument("--puft_phase2a_ratio", type=float, default=0.15, help="Phase 2a ratio in Stage 2")
    parser.add_argument("--puft_phase2b_ratio", type=float, default=0.55, help="Phase 2b ratio in Stage 2")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 构建PUFT配置
    puft_cfg = PUFTConfig()
    puft_cfg.T_min = args.puft_T_min
    puft_cfg.T_max = args.puft_T_max
    puft_cfg.lambda_smooth = args.puft_lambda_smooth
    puft_cfg.lambda_range = args.puft_lambda_range
    puft_cfg.lambda_color_consist = args.puft_lambda_color
    puft_cfg.uncertainty_reg_weight = args.puft_uncertainty_reg
    puft_cfg.K_neighbors = args.puft_K_neighbors
    puft_cfg.phase2a_ratio = args.puft_phase2a_ratio
    puft_cfg.phase2b_ratio = args.puft_phase2b_ratio
    puft_cfg.stage2_iterations = args.iterations
    
    # 将PUFT学习率传递给OptimizationParams
    # 通过动态添加属性实现
    opt_args = op.extract(args)
    opt_args.uncertainty_lr = args.puft_uncertainty_lr
    opt_args.temperature_lr = args.puft_temperature_lr

    print("=" * 60)
    print("  PUFT: Physics-guided Uncertainty-aware Fine-Tuning")
    print("  for Thermal 3D Gaussian Splatting")
    print("=" * 60)
    print(f"● Optimizing " + args.model_path)
    print(f"  Temperature range: [{puft_cfg.T_min}, {puft_cfg.T_max}] °C")
    print(f"  λ_smooth={puft_cfg.lambda_smooth}, λ_range={puft_cfg.lambda_range}, λ_color={puft_cfg.lambda_color_consist}")
    print(f"  σ_lr={args.puft_uncertainty_lr}, T_lr={args.puft_temperature_lr}")
    print(f"  Phase ratios: 2a={puft_cfg.phase2a_ratio}, 2b={puft_cfg.phase2b_ratio}, 2c={1-puft_cfg.phase2a_ratio-puft_cfg.phase2b_ratio:.2f}")
    print("=" * 60)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Stage 1: RGB training (same as MFTG)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.start_checkpoint, args.debug_from, step=1, puft_cfg=puft_cfg)
    
    print(f"\033[1;97m●\033[0m Color training complete, preparing PUFT thermal fine-tuning...")
    
    # Stage 2: PUFT thermal fine-tuning
    training(lp.extract(args), opt_args, pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, step=2, puft_cfg=puft_cfg)

    # All done
    print(f"\033[1;32m●\033[0m PUFT Training complete.")
