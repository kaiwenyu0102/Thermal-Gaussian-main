# ThermalGaussian OMMG 分支技术文档

> 本文档针对 OMMG（One Multi-Modal Gaussian）分支，详细梳理项目架构、模块关系、数据流动和核心算法。

---

## 一、项目总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       train-OMMG.py (入口)                       │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │ arguments │    │    scene/     │    │  gaussian_renderer │     │
│  │ 参数配置   │───▶│ 场景+数据加载  │───▶│    渲染引擎        │     │
│  └──────────┘    └──────┬───────┘    └────────┬───────────┘     │
│                         │                     │                  │
│                    ┌────▼────┐          ┌─────▼──────────┐      │
│                    │ cameras │          │ diff-gaussian-  │      │
│                    │ 相机模型 │          │ rasterization   │      │
│                    └─────────┘          │ CUDA光栅化器    │      │
│                                         └────────────────┘      │
│  ┌──────────┐    ┌──────────────┐                               │
│  │  utils/   │    │ gaussian_model│                              │
│  │ 损失/工具 │◀───│ 3D高斯数据结构 │                              │
│  └──────────┘    └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 模块调用关系

```
train-OMMG.py
  ├── arguments/          参数定义
  │     ModelParams          数据路径、SH阶数、分辨率
  │     OptimizationParams   学习率、迭代次数、密集化参数
  │     PipelineParams       渲染管线配置
  │
  ├── scene/              场景管理
  │     ├── __init__.py       Scene 类 → 统一加载 RGB+Thermal
  │     ├── dataset_readers   读取 COLMAP 数据 + 配对图像
  │     ├── cameras           Camera/MiniCam → 包含 original_image + original_thermal
  │     ├── colmap_loader     解析 COLMAP 二进制/文本格式
  │     └── gaussian_model    GaussianModel → 共享几何 + 双通道SH
  │
  ├── gaussian_renderer/  渲染
  │     ├── __init__.py       render() → 一次输出 render_color + render_thermal
  │     └── network_gui       实时可视化
  │
  ├── utils/              工具
  │     ├── loss_utils        l1_loss, ssim, smoothness_loss
  │     ├── sh_utils          球谐函数: RGB2SH, SH2RGB, eval_sh
  │     ├── graphics_utils    BasicPointCloud, 投影矩阵, FoV转换
  │     ├── camera_utils      loadCam → 同时加载 RGB + Thermal GT
  │     ├── image_utils       psnr, mse
  │     ├── general_utils     inverse_sigmoid, build_rotation
  │     └── system_utils      目录管理, 迭代搜索
  │
  └── submodules/         编译子模块
        ├── diff-gaussian-rasterization/  OMMG定制版光栅化器
        │     ├── cuda_rasterizer/        CUDA核心(支持双SH)
        │     └── diff_gaussian_rasterization/  Python接口
        └── simple-knn/                   KNN距离(高斯初始化)
```

---

## 二、数据输入输出

### 2.1 输入数据结构

训练前需准备如下目录结构：

```
<location>/
├── rgb/
│   ├── train/           # RGB 训练图像
│   │   ├── image_1.jpg
│   │   └── ...
│   └── test/            # RGB 测试图像
│       ├── image_0.jpg
│       └── ...
├── thermal/
│   ├── train/           # 热红外训练图像 (与RGB同名配对)
│   │   ├── image_1.jpg
│   │   └── ...
│   └── test/            # 热红外测试图像
│       ├── image_0.jpg
│       └── ...
└── sparse/
    └── 0/
        ├── cameras.bin   # COLMAP 相机内参
        ├── images.bin    # COLMAP 相机外参
        └── points3D.bin  # COLMAP 稀疏3D点云
```

**配对要求**: `rgb/train/image_1.jpg` 与 `thermal/train/image_1.jpg` 必须同名，表示同一视角的 RGB 和热红外观测。

### 2.2 输出数据结构

训练完成后输出：

```
<model_path>/
├── cfg_args                    # 运行参数快照
├── cameras.json                # 相机参数JSON
├── input.ply                   # 初始点云副本
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply     # 训练好的高斯 (含 t_dc, t_rest 等热红外属性)
├── train/                      # 渲染结果 (render.py 生成)
│   └── ours_30000/
│       ├── renders_color/      # RGB 渲染图
│       ├── gt_color/           # RGB GT
│       ├── renders_thermal/    # 热红外渲染图
│       └── gt_thermal/         # 热红外 GT
├── test/                       # 同上，测试集
└── results.json                # 评估指标 (metrics.py 生成)
```

### 2.3 PLY 文件中的高斯属性

OMMG 分支的 `point_cloud.ply` 包含以下字段：

| 字段前缀 | 含义 | 维度 |
|----------|------|------|
| `x, y, z` | 3D位置 | 每点3个 |
| `nx, ny, nz` | 法线(占位) | 每点3个 |
| `f_dc_0, f_dc_1, f_dc_2` | RGB SH 直流分量 | 每点3个 |
| `f_rest_0 ... f_rest_44` | RGB SH 高阶分量 | 每点45个 |
| `t_dc_0, t_dc_1, t_dc_2` | 热红外 SH 直流分量 | 每点3个 |
| `t_rest_0 ... t_rest_44` | 热红外 SH 高阶分量 | 每点45个 |
| `opacity` | 不透明度 | 每点1个 |
| `scale_0, scale_1, scale_2` | 缩放 | 每点3个 |
| `rot_0, rot_1, rot_2, rot_3` | 旋转四元数 | 每点4个 |

---

## 三、数据流动详解

### 3.1 完整数据流总览

```
原始数据                    预处理(COLMAP)              训练循环
───────                    ──────────────              ────────

rgb/*.jpg ─┐                                    
            ├→ convert.py → sparse/ ─→ Scene 初始化
            │   (SfM)       │           │
            │               │           ├── CameraInfo(image + thermal)
            │               │           │      ↓
            │               │           ├── Camera(original_image + original_thermal)
            │               │           │      ↓
            │               │           └── GaussianModel 初始化
            │               │                  │
thermal/*.jpg ─────────────────────────────────┘ (同名配对读取)
                                │
                                ▼
                    COLMAP points3D → create_from_pcd()
                         │
                    ┌────┴────┐
                    │ N个高斯  │
                    │ xyz      │ ← COLMAP 3D坐标
                    │ f_dc     │ ← COLMAP RGB颜色 → RGB2SH
                    │ t_dc     │ ← 零初始化
                    │ scaling  │ ← distCUDA2(KNN距离)
                    │ rotation │ ← 单位四元数
                    │ opacity  │ ← 0.1
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │     训练循环         │
              │                     │
              │  ① 选取Camera       │
              │  ② render()         │
              │     ↓               │
              │  一次光栅化同时输出:  │
              │  render_color       │──→ vs gt_image   → loss_color
              │  render_thermal     │──→ vs gt_thermal → loss_thermal
              │                     │
              │  ③ total_loss       │
              │  = (loss_color      │
              │   + loss_thermal)×0.5│
              │                     │
              │  ④ loss.backward()  │
              │  → 更新所有高斯参数   │
              │                     │
              │  ⑤ 自适应密度控制     │
              │  (clone/split/prune) │
              └─────────────────────┘
```

### 3.2 数据加载流

```
readColmapSceneInfo(path)
    │
    ├── read_extrinsics_binary("sparse/0/images.bin")
    │     → cam_extrinsics: 每张图的位姿(R, T)
    │
    ├── read_intrinsics_binary("sparse/0/cameras.bin")
    │     → cam_intrinsics: 焦距、主点
    │
    ├── readColmapCameras(extrinsics, intrinsics,
    │       images_folder="rgb/train",
    │       thermal_folder="thermal/train")
    │     │
    │     ├── image_path = "rgb/train/{basename}"
    │     ├── thermal_path = "thermal/train/{basename}"
    │     ├── image = Image.open(image_path)       ← RGB GT
    │     ├── thermal = Image.open(thermal_path)   ← Thermal GT
    │     └── CameraInfo(image=image, thermal=thermal, R, T, FovX, FovY)
    │
    ├── read_points3D_binary("sparse/0/points3D.bin")
    │     → xyz, rgb → BasicPointCloud
    │
    └── SceneInfo(point_cloud, train_cameras, test_cameras)
```

**关键设计**: OMMG 的 `readColmapCameras` 接受 `images_folder` 和 `thermal_folder` 两个参数，**同步读取同名 RGB+热红外图像**，存入同一个 `CameraInfo` 中。这与 main 分支分开两个函数（`readColmapSceneInfo` + `readTemperSceneInfo`）形成对比。

### 3.3 Camera 构建流

```
CameraInfo (image + thermal)
    │
    ▼
loadCam()  [utils/camera_utils.py]
    ├── resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    ├── gt_image = resized_image_rgb[:3, ...]
    ├── resized_image_thermal = PILtoTorch(cam_info.thermal, resolution)
    ├── gt_thermal = resized_image_thermal[:3, ...]
    │
    └── Camera(image=gt_image, thermal=gt_thermal, R, T, FoVx, FoVy)
           ├── original_image:   (3, H, W)  RGB GT
           ├── original_thermal: (3, H, W)  Thermal GT
           ├── world_view_transform:  (4, 4)
           ├── projection_matrix:     (4, 4)
           └── full_proj_transform:   (4, 4)
```

### 3.4 渲染流

```
render(viewpoint_camera, pc, pipe, bg)
    │
    ├── 构建光栅化配置 GaussianRasterizationSettings
    │     (image_height, image_width, tanfovx, tanfovy, bg,
    │      viewmatrix, projmatrix, sh_degree, campos)
    │
    ├── 准备高斯属性
    │     means3D  = pc.get_xyz             (N, 3)   共享位置
    │     opacity  = pc.get_opacity         (N, 1)   共享不透明度
    │     scales   = pc.get_scaling         (N, 3)   共享缩放
    │     rotations= pc.get_rotation        (N, 4)   共享旋转
    │     shs      = pc.get_features        (N, 16, 3) RGB SH系数
    │     thermal_shs = pc.get_thermal_features (N, 16, 3) Thermal SH系数
    │
    ├── 一次光栅化
    │     rasterizer(means3D, means2D,
    │         thermal_shs=thermal_shs,      ← 热红外SH
    │         color_shs=shs,                ← RGB SH
    │         opacities=opacity,
    │         scales=scales,
    │         rotations=rotations)
    │
    │     [CUDA 内部流程]
    │     1. 投影: 3D→2D, 计算每个高斯的屏幕空间位置
    │     2. 排序: 按深度排序高斯
    │     3. Alpha Blending (RGB通道):    用 color_shs  → rendered_color
    │     3. Alpha Blending (Thermal通道): 用 thermal_shs → rendered_thermal
    │        ↑ 同一组几何, 两套SH系数, 同一次前向传播
    │
    └── 返回 {"render_color": ..., "render_thermal": ..., 
              "viewspace_points": ..., "visibility_filter": ..., "radii": ...}
```

### 3.5 损失计算流

```
render_color    vs  gt_image      →  loss_color
render_thermal  vs  gt_thermal    →  loss_thermal

详细展开:

Ll1_color = l1_loss(render_color, gt_image)
loss_color = (1 - λ_dssim) * Ll1_color + λ_dssim * (1 - ssim(render_color, gt_image))

Ll1_thermal = l1_loss(render_thermal, gt_thermal)
smoothloss_thermal = smoothness_loss(render_thermal)    ← 热红外平滑先验
loss_thermal = (1 - λ_dssim) * Ll1_thermal 
             + λ_dssim * (1 - ssim(render_thermal, gt_thermal))
             + 0.6 * smoothloss_thermal

total_loss = (loss_color + loss_thermal) * 0.5

total_loss.backward()
  → 梯度回传至: _xyz, _features_dc, _features_rest,
                _thermal_dc, _thermal_rest, _opacity, _scaling, _rotation
```

### 3.6 自适应密度控制流

```
每 densification_interval 次迭代:

  xyz_gradient_accum / denom → 平均梯度

  ├── densify_and_clone: 梯度大的小高斯 → 复制一份(含RGB+Thermal SH)
  ├── densify_and_split: 梯度大的大高斯 → 分裂为2份(含RGB+Thermal SH)
  └── prune: 删除低不透明度/过大的高斯(含RGB+Thermal SH)

每 opacity_reset_interval 次迭代:
  └── reset_opacity: 重置所有高斯不透明度为 min(opacity, 0.01)
```

---

## 四、核心模块详解

### 4.1 GaussianModel (scene/gaussian_model.py)

OMMG 的核心数据结构，每个高斯点同时编码 RGB 和热红外外观。

#### 属性清单

| 属性 | 形状 | 初始化来源 | 激活函数 |
|------|------|-----------|---------|
| `_xyz` | (N, 3) | COLMAP 点云坐标 | 无 (直接优化) |
| `_features_dc` | (N, 1, 3) | COLMAP RGB→RGB2SH | 无 |
| `_features_rest` | (N, 15, 3) | 零 | 无 |
| `_thermal_dc` | (N, 1, 3) | 零 | 无 |
| `_thermal_rest` | (N, 15, 3) | 零 | 无 |
| `_scaling` | (N, 3) | log(sqrt(KNN距离)) | `torch.exp` |
| `_rotation` | (N, 4) | (1,0,0,0) | `F.normalize` |
| `_opacity` | (N, 1) | `inverse_sigmoid(0.1)` | `torch.sigmoid` |

#### 优化器参数组

```python
[
  {'params': [_xyz],           'lr': 0.00016 * spatial_lr_scale,  "name": "xyz"},
  {'params': [_features_dc],   'lr': 0.0025,                      "name": "f_dc"},
  {'params': [_features_rest], 'lr': 0.0025/20,                   "name": "f_rest"},
  {'params': [_thermal_dc],    'lr': 0.0025,                      "name": "thermal_dc"},
  {'params': [_thermal_rest],  'lr': 0.0025/20,                   "name": "t_rest"},
  {'params': [_opacity],       'lr': 0.05,                        "name": "opacity"},
  {'params': [_scaling],       'lr': 0.005,                       "name": "scaling"},
  {'params': [_rotation],      'lr': 0.001,                       "name": "rotation"}
]
```

#### 关键方法

- **`create_from_pcd(pcd, spatial_lr_scale)`**: 从 COLMAP 点云初始化所有高斯属性
- **`get_features`**: 拼接 `_features_dc` + `_features_rest`，返回 (N, 16, 3) RGB SH
- **`get_thermal_features`**: 拼接 `_thermal_dc` + `_thermal_rest`，返回 (N, 16, 3) Thermal SH
- **`densify_and_clone(grads, ...)`**: 克隆小高斯，同时复制 RGB + Thermal SH
- **`densify_and_split(grads, ...)`**: 分裂大高斯，同时复制 RGB + Thermal SH
- **`save_ply`/`load_ply`**: 保存/加载包含 `t_dc_*`, `t_rest_*` 字段的 PLY 文件

### 4.2 Scene (scene/__init__.py)

OMMG 只有一个统一的 `Scene` 类，同时管理 RGB 和热红外的相机数据。

```python
class Scene:
    def __init__(self, args, gaussians, load_iteration=None, ...):
        # 1. 加载场景信息 (RGB + Thermal 配对)
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images)
        
        # 2. 创建相机列表 (每个Camera同时包含 original_image + original_thermal)
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, ...)
        self.test_cameras  = cameraList_from_camInfos(scene_info.test_cameras, ...)
        
        # 3. 初始化或加载高斯
        if loaded_iter:
            gaussians.load_ply("point_cloud/iteration_{iter}/point_cloud.ply")
        else:
            gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    
    def save(self, iteration):
        # 保存到 point_cloud/iteration_{iter}/point_cloud.ply
```

### 4.3 Camera (scene/cameras.py)

```python
class Camera(nn.Module):
    def __init__(self, ..., image, thermal, gt_alpha_mask, ...):
        self.original_image   = image.clamp(0, 1)      # (3, H, W) RGB GT
        self.original_thermal = thermal.clamp(0, 1)    # (3, H, W) Thermal GT
        
        # 共享几何变换矩阵
        self.world_view_transform  = getWorld2View2(R, T, trans, scale)
        self.projection_matrix     = getProjectionMatrix(znear, zfar, fovX, fovY)
        self.full_proj_transform   = world_view @ projection
        self.camera_center         = world_view.inverse()[3, :3]
```

### 4.4 渲染器 (gaussian_renderer/__init__.py)

```python
def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0,
           override_color=None, override_thermal=None):
    
    # 准备两组 SH 系数
    shs = pc.get_features              # (N, 16, 3) RGB SH
    thermal_shs = pc.get_thermal_features  # (N, 16, 3) Thermal SH
    
    # 一次光栅化，同时输出两种模态
    rendered_thermal, rendered_color, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        thermal_shs = thermal_shs,     # 热红外 SH
        color_shs = shs,               # RGB SH
        opacities = opacity,
        scales = scales,
        rotations = rotations)
    
    return {
        "render_color": rendered_color,     # (3, H, W) RGB 渲染图
        "render_thermal": rendered_thermal, # (3, H, W) Thermal 渲染图
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }
```

### 4.5 定制光栅化器 (submodules/diff-gaussian-rasterization)

OMMG 分支对原始 diff-gaussian-rasterization 进行了关键修改：

**Python 接口变更** (`diff_gaussian_rasterization/__init__.py`):
- `GaussianRasterizer.forward()` 新增参数: `thermal_shs`, `thermals_precomp`
- 前向输出: `thermal, color, radii` (原来是 `color, radii`)
- 反向输入: `grad_out_thermal, grad_out_color` (双通道梯度)

**C++/CUDA 层变更**:
- `rasterize_points.h`: `RasterizeGaussiansCUDA` 新增 `thermal_sh`, `thermals` 参数
- `forward.cu`: 对每个像素同时用 `color_sh` 和 `thermal_sh` 计算 alpha blending
- `backward.cu`: 双通道梯度 `dL_dout_thermal`, `dL_dout_color` 回传
- `auxiliary.h`: `GaussianData` 结构体新增 `thermal_sh` 字段

### 4.6 损失函数 (utils/loss_utils.py)

#### 通用损失

```python
l1_loss(output, gt)        # L1 绝对值损失
ssim(img1, img2)           # 结构相似性 (11×11 高斯窗口)
```

#### 热红外平滑损失

```python
smoothness_loss(image_map):
    # 利用热红外图像温度分布平滑的物理先验
    # 计算4邻域(上下左右)像素差绝对值之和
    adj = generate_adj_neighbors(image_map, k=4)
    loss = Σ |image_map - adj[..., i]|  (i=0..3)
    loss /= (C * H * W)
    return loss
```

该损失惩罚热红外渲染图中的像素级剧烈跳变，促使温度场平滑。

---

## 五、训练算法伪代码

```
输入: 数据集路径 source_path, 输出路径 model_path
输出: 训练好的 GaussianModel (包含 RGB + Thermal SH)

1.  加载场景
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(source_path, gaussians)
    → 内部: COLMAP点云初始化高斯, Camera含RGB+Thermal GT

2.  配置优化器
    gaussians.training_setup(opt)
    → Adam优化器, 8组参数(xyz, f_dc, f_rest, thermal_dc, t_rest, opacity, scaling, rotation)

3.  FOR iteration = 1 TO 30000:

    3.1 随机选取训练视角
        viewpoint_cam = random_choice(scene.getTrainCameras())

    3.2 渲染 (一次光栅化, 双模态输出)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image   = render_pkg["render_color"]     # RGB 渲染图
        thermal = render_pkg["render_thermal"]    # Thermal 渲染图

    3.3 计算损失
        gt_image   = viewpoint_cam.original_image
        gt_thermal = viewpoint_cam.original_thermal
        
        loss_color   = (1-λ) * L1(image, gt_image) + λ * (1-SSIM(image, gt_image))
        loss_thermal = (1-λ) * L1(thermal, gt_thermal) + λ * (1-SSIM(thermal, gt_thermal))
                       + 0.6 * smoothness_loss(thermal)
        total_loss   = (loss_color + loss_thermal) * 0.5

    3.4 反向传播
        total_loss.backward()
        → 梯度回传至所有8组参数

    3.5 自适应密度控制 (500 ≤ iteration ≤ 15000)
        IF iteration % 100 == 0:
            densify_and_clone()   # 梯度大的小高斯 → 克隆
            densify_and_split()   # 梯度大的大高斯 → 分裂
            prune()               # 删除低不透明度/过大的高斯
        IF iteration % 3000 == 0:
            reset_opacity()       # 重置不透明度

    3.6 优化器步进
        optimizer.step()
        optimizer.zero_grad()

4.  保存模型
    scene.save(30000)
    → point_cloud/iteration_30000/point_cloud.ply
```

---

## 六、评估流程

### 渲染

```powershell
python render.py -m output/Truck
```

对每个视角调用 `render()` 获取 `render_color` 和 `render_thermal`，分别保存到：
- `renders_color/` / `gt_color/` (RGB)
- `renders_thermal/` / `gt_thermal/` (Thermal)

### 指标计算

```powershell
python metrics.py -m output/Truck
```

计算 6 项指标：

| 指标 | RGB | Thermal |
|------|-----|---------|
| PSNR | color_PSNR | thermal_PSNR |
| SSIM | color_SSIM | thermal_SSIM |
| LPIPS | color_LPIPS | thermal_LPIPS |

结果写入 `results.json` 和 `per_view.json`。

---

## 七、关键超参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `sh_degree` | 3 | 球谐阶数 (0-3, 共16个系数) |
| `iterations` | 30000 | 总训练迭代 |
| `position_lr_init` | 0.00016 | 位置学习率 |
| `feature_lr` | 0.0025 | RGB SH 学习率 |
| `thermal_feature_lr` | 0.0025 | Thermal SH 学习率 |
| `opacity_lr` | 0.05 | 不透明度学习率 |
| `scaling_lr` | 0.005 | 缩放学习率 |
| `rotation_lr` | 0.001 | 旋转学习率 |
| `lambda_dssim` | 0.2 | SSIM 损失权重 |
| `densify_from_iter` | 500 | 密集化起始 |
| `densify_until_iter` | 15000 | 密集化终止 |
| `densification_interval` | 100 | 密集化间隔 |
| `opacity_reset_interval` | 3000 | 不透明度重置间隔 |
| `densify_grad_threshold` | 0.0002 | 密集化梯度阈值 |
| smooth_loss 系数 | 0.6 | 热红外平滑损失权重 |
| RGB/Thermal 损失权重 | 各 0.5 | 等权平均 |

---

## 八、与 main 分支的关键差异

| 维度 | main 分支 | OMMG 分支 |
|------|-----------|-----------|
| 训练脚本 | train_MSMG.py + train_MFTG.py | train-OMMG.py |
| 场景类 | Scene_1(RGB) + Scene_2(Thermal) | Scene(统一) |
| 数据读取 | 分两步: readColmapSceneInfo + readTemperSceneInfo | 一步: readColmapSceneInfo 同时配对读取 |
| CameraInfo | 仅 image | image + thermal + thermal_path |
| Camera | original_image | original_image + original_thermal |
| GaussianModel | _features_dc + _features_rest | 额外增加 _thermal_dc + _thermal_rest |
| 渲染输出 | "render" | "render_color" + "render_thermal" |
| 光栅化 | 两次独立 | 一次双模态 (定制CUDA) |
| 优化器 | 7组参数 | 8组参数 (多 thermal_dc + thermal_rest) |
| 损失 | MSMG: 动态权重; MFTG: 分阶段 | 等权 0.5 + 0.5 |
| PLY 属性 | f_dc, f_rest, opacity, scale, rot | 增加 t_dc, t_rest |
| 子模块 | 原版 diff-gaussian-rasterization | OMMG 定制版 (支持 thermal_shs) |
