# ThermalGaussian MFTG 算法技术文档

> 本文档针对 main 分支下的 MFTG（Multi-modal Fine-tuning Gaussian）算法，详细梳理算法原理、数据处理流程、训练/评估/渲染的完整复现步骤。

---

## 一、MFTG 算法概述

### 1.1 核心思想

MFTG 采用**两阶段训练策略**：

1. **Phase 1 (step=1)**：只用 RGB 图像训练一套 3D 高斯，让高斯学会场景的几何结构和 RGB 外观
2. **Phase 2 (step=2)**：复用 Phase 1 训练好的高斯（几何、不透明度等已固定好），只微调 SH 颜色系数来适配热红外图像

```
Phase 1: RGB 训练 (30000 iter)          Phase 2: Thermal 微调 (30000 iter)
┌──────────────────────┐               ┌──────────────────────┐
│  GaussianModel        │               │  同一个 GaussianModel  │
│  从COLMAP点云初始化     │   ──复用──→   │  继承Phase1的全部参数   │
│  用RGB GT监督训练      │               │  用Thermal GT微调      │
│  输出: 完整的RGB高斯    │               │  输出: 兼容热红外的高斯  │
└──────────────────────┘               └──────────────────────┘
```

### 1.2 与 MSMG / OMMG 的对比

| 维度 | MSMG | MFTG | OMMG |
|------|------|------|------|
| 高斯数量 | 2套独立 | 1套共享 | 1套共享 |
| 训练方式 | 并行训练RGB+Thermal | 先RGB后Thermal微调 | 同时训练RGB+Thermal |
| 几何共享 | 否 | 是(Phase1→Phase2) | 是 |
| 渲染方式 | 两次独立光栅化 | 两次独立光栅化 | 一次双模态光栅化 |
| 损失函数 | 动态权重融合 | 分阶段、各自独立 | 等权0.5+0.5 |
| 显存占用 | 最高 | 中等 | 最低 |

---

## 二、数据输入

### 2.1 数据集目录结构

```
<source_path>/
├── input/               # 原始 RGB 图像 (COLMAP 使用)
│   ├── IMG_0001.jpg
│   └── ...
├── rgb/
│   ├── train/           # RGB 训练图像 (与COLMAP同名)
│   │   ├── IMG_0001.jpg
│   │   └── ...
│   └── test/            # RGB 测试图像
│       ├── IMG_0000.jpg
│       └── ...
├── thermal/
│   ├── train/           # 热红外训练图像 (与RGB同名配对)
│   │   ├── IMG_0001.jpg
│   │   └── ...
│   └── test/            # 热红外测试图像
│       ├── IMG_0000.jpg
│       └── ...
└── sparse/
    └── 0/
        ├── cameras.bin   # COLMAP 相机内参
        ├── images.bin    # COLMAP 相机外参
        └── points3D.bin  # COLMAP 稀疏3D点云
```

### 2.2 数据配对要求

- `rgb/train/` 和 `thermal/train/` 中的图像**必须同名**
- 热红外图像共享 COLMAP 从 RGB 图像计算出的相机位姿
- 热红外图像应与 RGB 图像**空间配准**（像素级对齐，或通过 MSX 融合实现）

### 2.3 数据加载流程

MFTG 使用两个独立的 Scene 类分别加载 RGB 和 Thermal 数据：

```
Phase 1 (step=1):
  Scene_1(dataset, gaussians)
    └── sceneLoadTypeCallbacks["Colmap"](source_path, images)
        ├── 读取 COLMAP 位姿: images.bin, cameras.bin
        ├── 读取 RGB 图像: rgb/train/*.jpg, rgb/test/*.jpg
        └── 读取点云: sparse/0/points3D.bin → 初始化高斯

Phase 2 (step=2):
  Scene_2(dataset, gaussians)
    └── sceneLoadTypeCallbacks["Temper"](source_path, images)
        ├── 读取 COLMAP 位姿: images.bin, cameras.bin (同一套!)
        ├── 读取 Thermal 图像: thermal/train/*.jpg, thermal/test/*.jpg
        └── 使用 Phase 1 已训练的高斯 (不再重新初始化)
```

**关键代码**（[train_MFTG.py](file:///d:/科研教学/科研/项目文件/Thermal-Gaussian-main/train_MFTG.py#L39-L48)）：

```python
if step == 1:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene_1(dataset, gaussians)       # 加载 RGB + COLMAP 点云
    print("start training color pictures")

if step == 2:
    scene = Scene_2(dataset, gaussians)       # 加载 Thermal，复用高斯
    scene.gaussians = scene_temp.gaussians     # 关键：继承 Phase 1 的高斯！
    print("start training thermal pictures")
```

---

## 三、算法核心流程

### 3.1 Phase 1：RGB 训练

```
输入: COLMAP 点云 + RGB 图像
目标: 训练一套能渲染 RGB 图像的3D高斯

每个迭代:
  1. 随机选取 RGB 训练视角
  2. 渲染: render(viewpoint_cam, gaussians, pipe, bg)
     → 输出 (3, H, W) 的 RGB 渲染图
  3. 损失:
     Ll1 = l1_loss(rendered, gt_rgb)
     loss = (1 - λ_dssim) * Ll1 + λ_dssim * (1 - ssim(rendered, gt_rgb))
  4. loss.backward() → 更新所有高斯参数
  5. 自适应密度控制 (clone/split/prune)

迭代 30000 次后 → 保存到 point_cloud_color/iteration_30000/
同时保存 scene_temp = scene (供 Phase 2 使用)
```

### 3.2 Phase 2：Thermal 微调

```
输入: Phase 1 训练好的高斯 + Thermal 图像
目标: 微调高斯的 SH 颜色系数以渲染热红外图像

每个迭代:
  1. 随机选取 Thermal 训练视角
  2. 渲染: render(viewpoint_cam, gaussians, pipe, bg)
     → 输出 (3, H, W) 的 Thermal 渲染图
  3. 损失:
     Ll1 = l1_loss(rendered, gt_thermal)
     smoothloss = smoothness_loss(rendered)         ← 热红外平滑先验!
     loss = (1 - λ_dssim) * Ll1
          + λ_dssim * (1 - ssim(rendered, gt_thermal))
          + 0.6 * smoothloss
  4. loss.backward() → 更新所有高斯参数
  5. 自适应密度控制 (clone/split/prune)

迭代 30000 次后 → 保存到 point_cloud_thermal/iteration_30000/
```

### 3.3 两阶段的关键差异

| 差异点 | Phase 1 (RGB) | Phase 2 (Thermal) |
|--------|---------------|-------------------|
| Scene 类 | Scene_1 (读取 rgb/) | Scene_2 (读取 thermal/) |
| GT 图像 | viewpoint.original_image (RGB) | viewpoint.original_image (Thermal) |
| 平滑损失 | 无 | 有 (0.6 × smoothness_loss) |
| 保存路径 | point_cloud_color/ | point_cloud_thermal/ |
| 初始化 | 从 COLMAP 点云初始化 | 继承 Phase 1 的高斯 |
| TensorBoard 标签 | color/ | thermal/ |

### 3.4 热红外平滑损失

热红外图像具有**温度分布平滑**的物理先验，因此 Phase 2 额外引入平滑损失：

```python
# loss_utils.py
smoothness_loss(image_map):
    adj = generate_adj_neighbors(image_map, k=4)  # 上下左右4邻域
    loss = Σ |image_map - adj[..., i]|  (i=0..3)  # 4邻域差绝对值之和
    loss /= (C * H * W)
    return loss
```

该损失惩罚渲染图中相邻像素的剧烈跳变，促使温度场过渡平滑。

---

## 四、数据流动详解

### 4.1 完整数据流总览

```
COLMAP 预处理 (离线)
    │
    ▼
rgb/*.jpg ──→ readColmapSceneInfo() ──→ Scene_1
thermal/*.jpg ──→ readTemperSceneInfo() ──→ Scene_2
sparse/ ──→ 点云 + 位姿 ──→ 共享
    
    ┌────────── Phase 1 (step=1) ──────────┐
    │                                        │
    │  Scene_1 ──→ RGB Camera 列表           │
    │  COLMAP 点云 ──→ GaussianModel 初始化   │
    │                                        │
    │  循环 30000 次:                         │
    │    render(RGB_cam, gaussians) → RGB图   │
    │    vs gt_rgb → loss_rgb                │
    │    backward → 更新高斯参数              │
    │                                        │
    │  保存: point_cloud_color/iter_30000/    │
    │  保存: scene_temp = scene              │
    └────────────────────────────────────────┘
                        │
                        ▼ 继承高斯
    ┌────────── Phase 2 (step=2) ──────────┐
    │                                        │
    │  Scene_2 ──→ Thermal Camera 列表       │
    │  gaussians = scene_temp.gaussians      │
    │                                        │
    │  循环 30000 次:                         │
    │    render(Thermal_cam, gaussians) → T图 │
    │    vs gt_thermal → loss_thermal        │
    │      + 0.6 * smoothness_loss           │
    │    backward → 更新高斯参数              │
    │                                        │
    │  保存: point_cloud_thermal/iter_30000/ │
    └────────────────────────────────────────┘
```

### 4.2 渲染流

MFTG 使用 main 分支的**标准单模态光栅化器**，每次只输出一种模态：

```
render(viewpoint_camera, gaussians, pipe, bg)
  │
  ├── 光栅化配置 (来自 Camera 对象)
  │     image_height, image_width, tanfovx, tanfovy
  │     viewmatrix, projmatrix, sh_degree, campos
  │
  ├── 高斯属性
  │     means3D, means2D, shs (= get_features), opacity, scales, rotations
  │
  └── 光栅化输出
        rendered_image: (3, H, W)  ← 单模态，RGB 或 Thermal
        radii, visibility_filter
```

**注意**：与 OMMG 分支不同，main 分支的光栅化器**不支持双模态**，每次 `render()` 只能输出一张图。

### 4.3 损失计算流

```
Phase 1:
  rendered_image  vs  gt_rgb
  ├── Ll1 = l1_loss(rendered, gt_rgb)
  └── loss = (1 - 0.2) * Ll1 + 0.2 * (1 - ssim(rendered, gt_rgb))

Phase 2:
  rendered_image  vs  gt_thermal
  ├── Ll1 = l1_loss(rendered, gt_thermal)
  ├── ssim_term = 0.2 * (1 - ssim(rendered, gt_thermal))
  ├── smooth_term = 0.6 * smoothness_loss(rendered)
  └── loss = 0.8 * Ll1 + ssim_term + smooth_term
```

---

## 五、模型结构

### 5.1 GaussianModel 属性

MFTG 使用标准的 GaussianModel（与原版 3DGS 一致）：

| 属性 | 形状 | 初始化来源 | 激活函数 |
|------|------|-----------|---------|
| `_xyz` | (N, 3) | COLMAP 点云坐标 | 无 |
| `_features_dc` | (N, 1, 3) | COLMAP RGB → RGB2SH | 无 |
| `_features_rest` | (N, 15, 3) | 零 | 无 |
| `_scaling` | (N, 3) | log(sqrt(KNN距离)) | torch.exp |
| `_rotation` | (N, 4) | (1,0,0,0) 单位四元数 | F.normalize |
| `_opacity` | (N, 1) | inverse_sigmoid(0.1) | torch.sigmoid |

**关键点**：MFTG 的 GaussianModel **没有** `_thermal_dc` / `_thermal_rest`，热红外和 RGB 共享同一套 `_features_dc` / `_features_rest`。Phase 2 微调时，这些 SH 系数从 RGB 颜色逐渐转变为 Thermal 颜色。

### 5.2 Scene_1 vs Scene_2

| 属性 | Scene_1 | Scene_2 |
|------|---------|---------|
| 数据加载回调 | `sceneLoadTypeCallbacks["Colmap"]` | `sceneLoadTypeCallbacks["Temper"]` |
| 图像目录 | `rgb/train`, `rgb/test` | `thermal/train`, `thermal/test` |
| 保存目录 | `point_cloud_color/` | `point_cloud_thermal/` |
| Camera GT | RGB 图像 | Thermal 图像 |
| 共享位姿 | COLMAP 算出 | 同一套 COLMAP 位姿 |
| 点云初始化 | 从 COLMAP 点云初始化 | 继承 Phase 1 已训练的高斯 |

### 5.3 优化器参数

```python
# 标准六组参数（无 thermal 专属参数组）
[
  {'params': [_xyz],           'lr': 0.00016 * spatial_lr_scale,  "name": "xyz"},
  {'params': [_features_dc],   'lr': 0.0025,                      "name": "f_dc"},
  {'params': [_features_rest], 'lr': 0.0025/20,                   "name": "f_rest"},
  {'params': [_opacity],       'lr': 0.05,                        "name": "opacity"},
  {'params': [_scaling],       'lr': 0.005,                       "name": "scaling"},
  {'params': [_rotation],      'lr': 0.001,                       "name": "rotation"}
]
```

---

## 六、复现指南

### 6.1 环境配置

#### 硬件要求
- GPU: NVIDIA 显卡，CUDA 11.6 兼容，建议 ≥ 8GB 显存
- 系统: Linux / Windows (WSL2)

#### 软件环境

```powershell
# 1. 创建 conda 环境
conda env create --file environment.yml
conda activate thermal_gaussian

# environment.yml 核心依赖:
#   python=3.7.13
#   pytorch=1.12.1
#   torchvision=0.13.1
#   cudatoolkit=11.6
#   plyfile, tqdm, opencv, pillow 等

# 2. 编译自定义子模块
cd submodules/diff-gaussian-rasterization
pip install .

cd ../simple-knn
pip install .
```

### 6.2 数据准备

#### 方式一：使用官方数据集

从 [Google Drive](https://drive.google.com/drive/folders/1A6kdIjDe7kw-iKQkzjHNw0wgk_3V7hcp) 下载 RGBT-Scenes 数据集，解压后目录结构应符合 2.1 节的规范。

#### 方式二：自定义数据

```powershell
# Step 1: 准备原始图像
# 将 RGB 图像放入 <source_path>/input/
# 将配准后的热红外图像放入 <source_path>/thermal/

# Step 2: 运行 COLMAP 预处理
python convert.py -s <source_path>

# convert.py 会自动执行:
#   1. feature_extractor    → SIFT 特征提取
#   2. exhaustive_matcher   → 特征匹配
#   3. mapper               → SfM 重建 (位姿 + 点云)
#   4. image_undistorter    → 去畸变

# Step 3: 整理图像目录
# 将去畸变后的图像分别放入 rgb/train, rgb/test, thermal/train, thermal/test
# COLMAP 输出的 sparse/0/ 目录保持不变
```

### 6.3 训练

```powershell
# MFTG 训练 (自动执行 Phase 1 + Phase 2)
python train_MFTG.py -s <source_path> -m <output_path>

# 常用参数:
python train_MFTG.py \
    -s ./data/Truck \          # 数据集路径
    -m ./output/Truck_MFTG \   # 输出路径
    --iterations 30000 \       # 每个 Phase 的迭代次数
    --test_iterations 7000 30000 \  # 测试检查点
    --save_iterations 7000 30000    # 保存检查点
```

**训练过程说明**：

1. 脚本会自动按序执行两个 Phase
2. Phase 1 完成后打印 `"color training complete, prepare to training thermal pictures"`
3. Phase 2 完成后打印 `"Training complete."`
4. 总训练时间约为 2 × 30000 迭代

**可选参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-s / --source_path` | 必填 | 数据集路径 |
| `-m / --model_path` | 自动生成 | 输出路径 |
| `--iterations` | 30000 | 每个 Phase 的迭代次数 |
| `--sh_degree` | 3 | 球谐阶数 |
| `-r / --resolution` | -1 | 分辨率缩放 (1/2/4/8 或像素宽度) |
| `--white_background` | False | 白色背景 |
| `--eval` | True | 训练中评估 |
| `--port` | 6009 | GUI 端口 |

### 6.4 渲染

```powershell
# 渲染 RGB 和 Thermal 图像
python render.py -m <output_path>

# 渲染结果保存位置:
#   <output_path>/rgb_train/ours_<iter>/renders/    ← RGB 渲染图
#   <output_path>/rgb_train/ours_<iter>/gt/          ← RGB GT
#   <output_path>/thermal_train/ours_<iter>/renders/ ← Thermal 渲染图
#   <output_path>/thermal_train/ours_<iter>/gt/      ← Thermal GT
#   同理有 rgb_test/ 和 thermal_test/
```

**render.py 的工作方式**：
- 分别加载 `Scene_1`(RGB高斯) 和 `Scene_2`(Thermal高斯)
- 对每个视角调用标准 `render()` 函数
- RGB 高斯 → 渲染 RGB 图，Thermal 高斯 → 渲染 Thermal 图

### 6.5 评估

```powershell
# 计算评估指标
python metrics.py -m <output_path>

# 输出指标 (写入 results.json):
#   color PSNR, color SSIM, color LPIPS
#   thermal PSNR, thermal SSIM, thermal LPIPS
```

### 6.6 完整复现流程

```powershell
# 1. 环境准备
conda activate thermal_gaussian

# 2. 数据预处理 (自定义数据时)
python convert.py -s ./data/MyScene

# 3. 训练 MFTG
python train_MFTG.py -s ./data/MyScene -m ./output/MyScene_MFTG

# 4. 渲染
python render.py -m ./output/MyScene_MFTG

# 5. 评估
python metrics.py -m ./output/MyScene_MFTG

# 6. (可选) 查看 TensorBoard
tensorboard --logdir ./output/MyScene_MFTG
```

---

## 七、输出文件结构

```
<output_path>/
├── cfg_args                        # 运行参数快照
├── cameras.json                    # 相机参数 JSON
├── input.ply                       # 初始点云副本
├── point_cloud_color/              # Phase 1 (RGB) 保存
│   ├── iteration_7000/
│   │   └── point_cloud.ply
│   └── iteration_30000/
│       └── point_cloud.ply         # RGB 训练完成的高斯
├── point_cloud_thermal/            # Phase 2 (Thermal) 保存
│   ├── iteration_7000/
│   │   └── point_cloud.ply
│   └── iteration_30000/
│       └── point_cloud.ply         # Thermal 微调完成的高斯
├── rgb_train/                      # RGB 训练集渲染结果
│   └── ours_30000/
│       ├── renders/
│       └── gt/
├── rgb_test/                       # RGB 测试集渲染结果
│   └── ours_30000/
│       ├── renders/
│       └── gt/
├── thermal_train/                  # Thermal 训练集渲染结果
│   └── ours_30000/
│       ├── renders/
│       └── gt/
├── thermal_test/                   # Thermal 测试集渲染结果
│   └── ours_30000/
│       ├── renders/
│       └── gt/
├── results.json                    # 评估指标汇总
└── per_view.json                   # 逐视角评估指标
```

---

## 八、关键超参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sh_degree` | 3 | 球谐阶数 (0-3, 共16个系数) |
| `iterations` | 30000 | 每个 Phase 的总迭代数 |
| `position_lr_init` | 0.00016 | 位置学习率初始值 |
| `position_lr_final` | 0.0000016 | 位置学习率终值 |
| `feature_lr` | 0.0025 | SH 颜色学习率 |
| `opacity_lr` | 0.05 | 不透明度学习率 |
| `scaling_lr` | 0.005 | 缩放学习率 |
| `rotation_lr` | 0.001 | 旋转学习率 |
| `lambda_dssim` | 0.2 | SSIM 损失权重 |
| `densify_from_iter` | 500 | 密集化起始迭代 |
| `densify_until_iter` | 15000 | 密集化终止迭代 |
| `densification_interval` | 100 | 密集化间隔 |
| `opacity_reset_interval` | 3000 | 不透明度重置间隔 |
| `densify_grad_threshold` | 0.0002 | 密集化梯度阈值 |
| smooth_loss 系数 | 0.6 | 热红外平滑损失权重 (仅 Phase 2) |
| `percent_dense` | 0.01 | 密集化百分比阈值 |

---

## 九、训练算法伪代码

```
输入: source_path, model_path
输出: RGB 高斯模型 + Thermal 高斯模型

# ============ Phase 1: RGB 训练 ============
gaussians = GaussianModel(sh_degree=3)
scene = Scene_1(source_path, gaussians)
  → 加载 COLMAP 位姿 + RGB 图像
  → 从 COLMAP 点云初始化高斯

gaussians.training_setup(opt)
  → Adam 优化器, 6组参数

FOR iteration = 1 TO 30000:
    viewpoint_cam = random_choice(scene.getTrainCameras())
    rendered = render(viewpoint_cam, gaussians, pipe, bg)
    gt = viewpoint_cam.original_image

    Ll1 = l1_loss(rendered, gt)
    loss = (1-0.2) * Ll1 + 0.2 * (1 - ssim(rendered, gt))

    loss.backward()
    optimizer.step()

    IF iteration % 100 == 0 AND 500 ≤ iteration ≤ 15000:
        densify_and_prune()
    IF iteration % 3000 == 0:
        reset_opacity()

scene.save(30000)  → point_cloud_color/iteration_30000/
scene_temp = scene  # 保存 Phase 1 结果

# ============ Phase 2: Thermal 微调 ============
scene = Scene_2(source_path, gaussians)
  → 加载 COLMAP 位姿 + Thermal 图像 (同一套位姿!)
scene.gaussians = scene_temp.gaussians  # 继承 Phase 1 的高斯

gaussians.training_setup(opt)  # 重新初始化优化器

FOR iteration = 1 TO 30000:
    viewpoint_cam = random_choice(scene.getTrainCameras())
    rendered = render(viewpoint_cam, gaussians, pipe, bg)
    gt = viewpoint_cam.original_image  # 此时是 Thermal GT

    Ll1 = l1_loss(rendered, gt)
    smoothloss = smoothness_loss(rendered)
    loss = (1-0.2) * Ll1 + 0.2 * (1 - ssim(rendered, gt)) + 0.6 * smoothloss

    loss.backward()
    optimizer.step()

    IF iteration % 100 == 0 AND 500 ≤ iteration ≤ 15000:
        densify_and_prune()
    IF iteration % 3000 == 0:
        reset_opacity()

scene.save(30000)  → point_cloud_thermal/iteration_30000/
```

---

## 十、常见问题与注意事项

### Q1: Phase 2 是否会破坏 Phase 1 学到的 RGB 渲染能力？

**是的。** Phase 2 微调时，高斯的 SH 颜色系数会从 RGB 值逐渐偏移向 Thermal 值。因此，Phase 2 完成后，**RGB 渲染质量会下降**。这是 MFTG 的固有缺陷——同一套 SH 无法同时精确表达两种模态的颜色。

如果需要同时渲染高质量 RGB 和 Thermal，应使用 OMMG 分支（双通道 SH）。

### Q2: Phase 2 为什么需要重新 `training_setup`？

```python
gaussians.training_setup(opt)  # 重新初始化优化器
```

这会重置 Adam 的动量状态（exp_avg, exp_avg_sq），避免 Phase 1 的动量信息干扰 Phase 2 的优化方向。但高斯参数本身（_xyz, _features_dc 等）从 Phase 1 继承。

### Q3: render.py 如何分别渲染 RGB 和 Thermal？

render.py 分别加载两套独立的高斯模型：
- `point_cloud_color/iteration_30000/point_cloud.ply` → RGB 渲染
- `point_cloud_thermal/iteration_30000/point_cloud.ply` → Thermal 渲染

两套高斯的几何可能不同（Phase 2 的密集化/剪枝改变了高斯分布）。

### Q4: 如何从 checkpoint 恢复训练？

```powershell
python train_MFTG.py -s <source_path> -m <output_path> \
    --start_checkpoint <output_path>/chkpnt<iter>.pth
```

注意：checkpoint 恢复时，两个 Phase 都会从同一个 checkpoint 开始，可能不符合预期。建议完整训练而不中断。

### Q5: 显存不足怎么办？

- 降低分辨率：`-r 2` 或 `-r 4`
- 减少 SH 阶数：`--sh_degree 1`
- 使用更小的训练图像
- 考虑使用 OMMG 分支（单套高斯，显存更低）
