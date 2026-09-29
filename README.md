# An Artifact-Type Aware DIBR Method for View Synthesis —— 复现 + 通用填洞工具

> **复现驱动与断言套件已移除**（按用户要求，2026-09-29）：原 `step0–4*.py`、`check_step1–8.py`、
> `run_all.py`（共 1,804 行）已从仓库删除；**本文档保留其算法规格、参数含义与全部实测结论**，
> 需要时可按 §4 的规格重建。工具本体（`viewfill/` + `dibr/`）不受影响。

论文：A. Q. de Oliveira, M. Walter, C. R. Jung, *An Artifact-type Aware DIBR Method for
View Synthesis*, IEEE Signal Processing Letters, 2018, DOI 10.1109/LSP.2018.2870342。
论文无开源代码，本仓库是完整的自行实现 + 逐步验证。

**详细方案、逐步规格、验收标准、25 条论文歧义的处理记录、全部实测结论 →
[复现方案.md](复现方案.md)**（先读这一份）。

> 📘 **要直接用来填洞？先看 [`使用说明.md`](使用说明.md)**：需要哪些文件、如何调用、36 个参数的含义与默认值、常见场景配方与排错对照表。

## 两个入口：复现实验 vs 通用工具

| | 用途 | 入口 |
| --- | --- | --- |
| 论文复现（结论已归档） | 复现实验（绑定 MSR Ballet 官方标定 + 真实相邻相机做真值评价，40 次运行 89 项断言） | `复现方案.md` §3 |
| **`viewfill/`** | **通用填洞：给"原图 + 逆深度"或"任意 warp 出来的带空洞图"，输出无空洞结果** | 本 README 下方，或 `复现方案.md` §9 |

## 通用填洞工具（2D→3D 流程）

约定：`视差(px) = 逆深度(0..1) × scale`，默认 `scale=-44.8`（内容左移、OOFA 在右边缘）。

```python
from viewfill import fill_holes

fixed = fill_holes(image, inv_depth)                      # 一步到位：原图 + 归一化逆深度
fixed = fill_holes(image, inv_depth, scale=-44.8)         # 指定视差尺度
info  = fill_holes(image, inv_depth, return_info=True)    # 还要掩码/各阶段统计
fixed = fill_holes(image, inv_depth, verbose=True)        # 打印各阶段日志
```

`image` 可为 `HxWx3`（RGB/BGR，通道顺序原样保留）或 `HxW` 灰度；`inv_depth` 可为 0..1 浮点或 0..255
整数（自动识别），约定 **近=大**。返回值就是修好空洞的图像（`uint8`，形状/通道与输入一致）。

| 参数（都可省略） | 默认 | 说明 |
| --- | --- | --- |
| `scale` | `-44.8` | 视差(px) = 归一化逆深度 × scale；符号决定方向，绝对值决定空洞大小 |
| `depth_range` | `"auto"` | `auto` / `"01"` / `"255"` 强制深度约定 |
| `splat` | `"sub"` | `sub` 亚像素（裂纹少）／`floor`/`round` 整数单点（经典 DIBR，裂纹多） |
| `rule` | `"zbuf"` | `zbuf` 最近样本优先／`avg` 权重平均 |
| `lam` | `5.0` | 裂纹检测阈值（0..255 深度尺度） |
| `depth_dilate` | `"auto7"` | warp 前按需加宽 splat 足迹，闭合轮廓处的 1–2 px 裂纹（可见接缝主因）；需求图先做 7×7 最大值滤波避免加宽交界自身造缝。8 帧标定几何配对检验：`auto7` 接缝 6.08 / p90 **16.52** / GT PSNR **28.233**，`3` 5.99 / **15.80** / 28.231 —— **两者无显著差异**（p=0.38–0.84），`auto5` 明显更差（6.67 / 17.56 / 28.062）。`3` 逐样本胜率更高、`auto7` 条纹场景更顺（如 f004：5.68 vs 7.10）。`0` 关闭 |
| `struct_pen` | `8.0` | **结构感知跨行惩罚**：允许从别的行借用源块，但按 `struct_pen · w · dy²` 收费（`w` = 空洞邻域水平结构强度，`dy` = 行偏移）。竖直同质背景 `w≈0`（好匹配不受影响），栏杆处 `w` 大（位移被罚掉）。实测（全图/带内/横杆行 PSNR，行偏移 p90/max）：`0` 28.40/24.66/24.64, 22.5/49；`0.5` 28.26/24.00/25.08, 4.0/21；**`8`（默认，横杆对齐最好）28.41/24.74/24.42, 1.0/8** |
| `epipolar` | `None` | 硬性限制匹配器相对**几何反投影行**的偏离行数。**实测它并不能改善结果**（全图/带内/横杆行 PSNR：`None` 28.40/24.66/24.64、`2` 28.33/24.30/24.29、`0` 28.12/23.40/**19.92**）：disocclusion 的正确背景在参考图同一行上本就被遮挡，同行候选池装不下它。仅作逃生口，一般用 `struct_pen` 代替 |
| `beta`, `beta_mode` | `150`, `"mean"` | 自适应 patch 尺寸的接受阈值 |
| `skip_ghosts` | `False` | 跳过鬼影矫正 |
| `n_window`, `sizes` | `69`, `(9,7,5,3)` | 搜索窗边长 / patch 尺寸链 |
| `return_info` | `False` | 返回完整结果字典（掩码、各阶段统计、深度） |
| `verbose` | `False` | 打印各阶段日志 |

更底层/更灵活的两种用法：

```powershell
# A) 自带 warp（等价于 fill_holes，但可落盘所有中间产物）
python -m viewfill --image color.jpg --depth depth.png --scale -44.8 --out out\run1

# B) 只填已有 warp（你自己的 warp 代码产出的图/掩码/深度）
python -m viewfill --warped warped.png --hole hole.png --depth-warped warped_depth.png `
                   --image color.jpg --depth depth.png --out out\run1

# 可选：给真值就顺便算 PSNR/SSIM
python -m viewfill --image color.jpg --depth depth.png --scale -44.8 --gt gt.png --out out\run1
```

```python
from viewfill import FillConfig, warp_and_fill, fill_warped

res = warp_and_fill(rgb, inv_depth, FillConfig(scale=-44.8))        # 自带 warp
res = fill_warped(warped, hole_mask, warped_depth, rgb, inv_depth)  # 只填
res["filled"]   # HxWx3 uint8，已无空洞
res["stats"]    # 各阶段指标（含 GT-free 回投一致性）
```

**实测（`python test_viewfill.py`，15/15 通过，你的数据 cam6-f000）**

| 项 | 结果 |
| --- | --- |
| **一次调用** `fill_holes(color, inv_depth)` | 35 812 空洞 px → **0 px**，10 s（1024×768） |
| warp | scale −44.8 → 空洞 4.55 %，OOFA 在右边缘 |
| 填充 | 27 709 px → **0 px**，9.4 s |
| 喂入你提供的 `warping.scatter_image` 输出 | 35 812 px → **0 px**，9.6 s（`inverse_ordering=False`） |
| GT-free 回投一致性 | 28.50 → **35.15 dB** |
| scale −20 / −44.8 / −80 | 空洞 1.84 / 4.55 / 8.15 %，三档全部填满（4.4 / 10 / 19.7 s） |
| 真实相机压力测试（Ballet cam6→cam7，纯 1D） | 整帧 18.14 → **20.39 dB**，SSIM 0.677 → 0.710 |

注意：贪心填充顺序对输入微小扰动敏感（同一张图量化成 uint8 再喂回去，结果**质量等价**但非逐像素相同），验收请用"空洞清零 + 回投一致性 + 目视"，不要用逐像素比对。参数调节建议见 `复现方案.md` §9.4。

### 前景被"吃掉一半"？先看 warp 的碰撞规则

如果你自己 warp 出来的图前景人物像被切掉（背景穿透到人身上），根因通常是**前向 splat 没有深度测试**：
`warping.scatter_image(inverse_ordering=True)` 会让**远处样本赢得碰撞**，把前景纹理替换成背景。
实测与正确 Z-buffer warp 的逐像素偏差：`inverse_ordering=True` **1.88 %**（前景 0.54 %）、
`inverse_ordering=False` **0.08 %**、本包 `viewfill`（亚像素 + Z-buffer）**0.11 %**。

三种修法（任选）：
1. 让 `viewfill` 自己 warp（`--splat sub --rule zbuf`，默认）——推荐；
2. 用你自己的 warp 时把 `inverse_ordering` 设为 **`False`**；
3. 已经有损坏的图：交给 `viewfill`，`--repair-warp auto`（默认）会检测偏差并自动用 Z-buffer 重 warp
   （实测修复后与干净流程逐像素一致）；不想被改动就加 `--repair-warp never`。

若你的调用是 `scatter_image(frame, inv, direction=-1, scale_factor=44.8, inverse_ordering=True)`
（**这就是被判定为损坏的那一档**：偏差 1.88 %、前景 0.54 %），最小改动是把 `inverse_ordering` 改成
`False`（偏差 0.08 %，空洞掩码完全不变）；想要严格 Z-buffer 就用签名与返回值完全一致的 drop-in：

```python
from viewfill.compat import scatter_image_safe
img, mask, depth = scatter_image_safe(frame, inverse_depth,
                                      direction=-1, scale_factor=44.8,
                                      reproject_depth=True)   # 偏差 0.00 %
```

详细数据与验证见 `复现方案.md` §9.5 / §9.6。

## 流水线

```
参考视图 I + 逆深度 P
  └─ Step 0  标定：官方 calibParams-ballet.txt + 逆深度公式 → 逐像素位移场
  └─ Step 1  前向 warp（DIBR）：整数单点 / 亚像素 Z-buffer，输出 I_w、D_w（空洞 = −1）、空洞掩码
  └─ Step 2  裂纹检测与 HHF 填补        (论文 II-A)  λ=5，线状结构元，掩膜高斯金字塔
  └─ Step 3  鬼影检测与搬移到 p_FG      (论文 II-B)  截尾均值阈值 + 掩膜中位数一致性检验
  └─ Step 4  空洞分流：OOFA / disocclusion (论文 II-C)
  └─ Step 5  局部 FG-BG 提取 + 优先级 P(p)=B·E / C·D
  └─ Step 6  参考图上的 patch 匹配：N=69 搜索窗、仅背景候选、自适应 9×9→3×3
  └─ Step 7  逐空洞迭代填充直到无空洞
  └─ Step 8  与真实相邻相机图像算 PSNR/SSIM（4 组相机对 × 10 帧）
```

## 快速开始


## 逐步验证（每一步都有断言，共 89 项）

```powershell
```

## 主要结果（40 次运行，与真实目标相机图像比较）

| 相机对 | 空洞占比 | warp → +裂纹 → +鬼影 → **+填充** | SSIM |
| --- | --- | --- | --- |
| cam6→cam7 | 15.97 % | 14.64 → 17.01 → 17.06 → **27.53 dB** | 0.298 → **0.824** |
| cam6→cam5 | 15.32 % | 14.97 → 15.75 → 15.77 → **27.00 dB** | 0.528 → **0.808** |
| cam3→cam0（论文同款，大基线） | 33.00 % | 11.19 → 12.20 → 12.21 → **22.98 dB** | 0.192 → **0.721** |
| cam3→cam2（论文同款） | 17.04 % | 14.51 → 15.26 → 15.26 → **26.42 dB** | 0.484 → **0.800** |

40/40 次运行空洞全部填满。论文三项贡献中，"参考图搜索 + 仅背景候选"被强烈验证
（+9.28 / +3.00 dB），`B`/`E` 各有正贡献（+1.37 / +1.09 dB）；而鬼影步在本数据上无可测收益
（+0.018 dB），且用 `B·E` 取代 Criminini `C·D` 亦无增益 —— 详见 [复现方案.md](复现方案.md) §6。

## 目录


## 依赖

见 [requirements.txt](requirements.txt)：numpy、opencv-python-headless、scipy、Pillow。

## 被 git 忽略的内容（可自行决定是否入库）

- `output/`：约 3.9 GB 的中间产物与结果，全部可由上面的命令重新生成。若想版本化关键结果：
  `git add -f output/step8_eval/report.md output/step8_eval/progress.png`
- 论文 PDF 与 `input/` 图像：第三方/受限素材（IEEE 版权；MSR 数据声明仅限研究用途）。
  `input/` 需要放回 `color-cam6-f000..f009.jpg` 与 `depth-cam6-f000..f009.png`；
  完整数据集与官方标定放在 `D:\项目\3DVideos-distrib\MSR3DVideo-Ballet`（路径可用
  `--dataset_root` 修改）。
