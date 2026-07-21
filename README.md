# AI Video Face Masking

面向工业视频的本地化人脸检测、人工审核与隐私遮罩流水线。系统以事件为单位压缩审核工作量，支持局部时间范围修正，并通过 CUDA 硬件解码与智能渲染缩短长视频处理时间。

当前稳定版本：`4.0.0`

## 核心能力

- 本地人脸检测、跟踪、事件聚合与马赛克遮罩。
- Streamlit 人工审核界面，支持接受、拒绝、跳过及局部范围修正。
- NVIDIA HEVC 硬件解码；生产推荐 `cuda-full` 模式。
- 智能渲染：无须遮罩的 GOP 片段直接复制，仅重编码遮罩片段。
- 完整审计数据与传统渲染回退路径。
- Torch 为默认生产推理后端；ONNX 与 TensorRT 可用于独立评测。

## 已验证性能

在 37 分 05 秒、1920×1080、29.97 fps 的 HEVC 工业视频上：

| 阶段 | 优化前 | v4.0.0 | 缩短 |
| --- | ---: | ---: | ---: |
| 审核数据生成 | 20 分 15.5 秒 | 13 分 32.5 秒 | 33.2% |
| 最终成片渲染 | 约 28 分 46 秒 | 3 分 53 秒 | 约 86.5% |

性能结果取决于视频编码、GOP 结构、GPU、驱动和遮罩覆盖率。详细记录见 [发布说明](RELEASE_NOTES.md)。

## 系统要求

- Windows 10/11
- Python 3.10 或更高版本
- NVIDIA GPU（硬件加速模式需要 NVDEC/NVENC）
- FFmpeg，且硬件模式需包含 `hevc_cuvid` 与 `hevc_nvenc`
- CUDA 可用的 PyTorch 环境
- 人脸检测模型 `models/face.pt`

基础依赖：

```powershell
python -m pip install -r requirements.txt
```

可选加速依赖：

```powershell
python -m pip install --extra-index-url https://pypi.nvidia.com/ -r requirements-accel.txt
```

请根据本机 CUDA 与驱动版本安装匹配的 PyTorch。模型文件的获取与使用要求见 `models/README.md`（如仓库发行包中提供）。

## 快速开始

### 1. 生成审核数据

```powershell
python pipeline.py `
  -i "D:\path\to\input.MP4" `
  -o ".\production_outputs" `
  --mode fast_review `
  --review-only `
  --infer-backend torch `
  --decode-backend cuda-full `
  --detect-isolated `
  --detect-batch 4 `
  --interval 3 `
  --imgsz 1280
```

不具备 NVIDIA HEVC 硬解条件时，将 `--decode-backend` 改为 `opencv`。

### 2. 启动审核 UI

```powershell
streamlit run review_ui.py -- --review-dir ".\production_outputs\<video-name>\review"
```

常用操作：

- `A`：接受整个提案。
- `R`：拒绝整个提案。
- `S`：暂时跳过。
- `E`：展开局部时间范围修正，仅接受选中的一个或多个范围。

审核决定会持续写入 JSON 文件，可中断后继续。

### 3. 生成最终成片

```powershell
python confirm.py `
  --output-dir ".\production_outputs\<video-name>" `
  --smart-render `
  --final-name final.mp4
```

交付前需要完整解码验证时，添加 `--smart-render-full-validation`。如果输入编码或运行环境不适合智能渲染，移除 `--smart-render` 即可使用传统全片渲染。

## 处理流程

```text
输入视频
  -> 检测与跟踪
  -> 事件聚合与审核预览
  -> 人工审核（全部或局部范围）
  -> 遮罩时间轴
  -> 智能渲染 / 传统渲染
  -> 最终视频与审计结果
```

核心入口：

| 文件 | 用途 |
| --- | --- |
| `pipeline.py` | 检测、跟踪、事件与审核数据生成 |
| `review_ui.py` | Streamlit 人工审核界面 |
| `confirm.py` | 应用审核结果并生成最终成片 |
| `smart_render.py` | GOP 分段、流复制与按需重编码 |
| `render.py` | 传统全片马赛克渲染 |

## 目录结构

```text
docs/engineering/     性能优化、可行性与实测记录
models/               本地模型文件与说明
tests/                自动化测试
tools/benchmarks/     性能评测与实验工具
tools/exports/        模型导出工具
```

运行数据、视频、模型权重和 benchmark 产物默认不应提交到 Git。

## 验证

```powershell
python -m py_compile pipeline.py confirm.py review_ui.py smart_render.py
python -m unittest discover -s tests
```

生产发布前还应使用目标机器和代表性视频完成：审核抽检、音视频流检查、首中尾解码以及隐私覆盖验证。

## 隐私与安全

- 优先在受控本地环境中处理原始视频。
- 不要将输入视频、抽帧、模型权重、审核产物或运行日志提交到公共仓库。
- 输出视频必须经过人工抽检；自动检测结果不能替代最终隐私审核。
- 对外分发前应清理路径、人员信息、现场标识及其他敏感元数据。

## 文档

- [v4.0.0 发布说明](RELEASE_NOTES.md)
- [变更记录](CHANGELOG.md)
- [系统架构](ARCHITECTURE.md)
- [工程验证资料](docs/engineering/)

## 许可证

本仓库尚未包含许可证文件。项目所有者需在正式商业分发前确定闭源商业授权、开源授权或双重授权方案；在此之前，不应仅凭仓库可访问性推定获得复制、再分发或商业使用许可。
