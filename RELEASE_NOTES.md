# Production Release Notes

发布日期：2026-07-21
版本：`4.0.0`
发布分支：`release/v4.0.0`

## 版本概述

本版本完成了从人脸检测、人工审核到最终视频渲染的完整生产流程，并针对 37 分钟、
1920×1080、29.97 fps 的 HEVC 工业视频进行了端到端验证。

主要目标：

- 缩短审核数据生成时间，同时保持原有检测参数和隐私覆盖能力。
- 将最终成片渲染从全片重编码改为按需分段重编码。
- 改进人工审核 UI，使常规提案保持一键处理，混合提案支持精确时间范围。
- 保留稳定回退路径，避免将实验性的 TensorRT 后端设为默认方案。

## 主要功能

### 1. CUDA HEVC 硬件解码

新增检测解码参数：

- `--decode-backend opencv`：原始 OpenCV 解码路径。
- `--decode-backend cuda`：NVIDIA HEVC 硬解并预缩放，速度优先。
- `--decode-backend cuda-full`：NVIDIA HEVC 硬解并保留源分辨率，推荐生产模式。

推荐使用 `cuda-full`。它不改变模型尺寸、检测间隔、置信度或 batch，只替换视频解码路径。

### 2. 独立检测进程与后端工具

- 检测与跟踪可在独立进程运行，完成后释放 GPU/运行时资源。
- 支持 Torch、ONNX 和 TensorRT 的显式性能测试。
- 提供后端 benchmark、ONNX 导出和结果对比工具。
- 生产默认仍为 Torch；TensorRT 保留为实验选项。

### 3. 混合智能渲染

`confirm.py --smart-render` 会：

- 仅重编码包含遮罩的片段。
- 对不含遮罩的片段进行视频流复制。
- 合并片段并重新复用原始音频。
- 在智能渲染失败时回退到传统全片渲染；可使用严格模式禁止回退。

### 4. 范围审核 UI

正式审核 UI 采用混合交互：

- `A`：接受整个提案。
- `R`：拒绝整个提案。
- `S`：跳过。
- `E`：仅在混合提案中展开局部时间范围修正。

局部修正支持一个或多个时间范围。保存后，最终渲染只处理所选范围内的遮罩帧。
旧版 `accepted_first_half` 和 `accepted_second_half` 决定仍可读取，保证历史数据兼容。

## 37 分钟整片实测

测试视频参数：

- 时长：约 37 分 05 秒
- 分辨率：1920×1080
- 帧率：29.97 fps
- 视频编码：HEVC
- 音频编码：AAC，48 kHz 双声道

| 阶段 | 原耗时 | 本版本耗时 | 改善 |
| --- | ---: | ---: | ---: |
| 审核数据生成 | 20 分 15.5 秒 | 13 分 32.5 秒 | 缩短 33.2% |
| 最终成片渲染 | 约 28 分 46 秒 | 3 分 53 秒 | 缩短约 86.5% |

审核结果：

- 主审核提案：291
- 全部接受：144
- 局部范围接受：47
- 拒绝：100
- 跳过：0
- 最终渲染遮罩：4956 帧、6723 个框

最终文件验证：

- 时长：37 分 05.37 秒
- 分辨率：1920×1080
- 帧率：29.97 fps
- HEVC 视频流和 AAC 音频流均存在
- 开头、中段和结尾均可正常解码

## 推荐生产命令

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

### 2. 启动审核 UI

```powershell
streamlit run review_ui.py -- --review-dir ".\production_outputs\<video-name>\review"
```

### 3. 生成最终成片

```powershell
python confirm.py `
  --output-dir ".\production_outputs\<video-name>" `
  --smart-render `
  --final-name final.mp4
```

如需对最终视频进行完整解码验证，可额外添加：

```powershell
--smart-render-full-validation
```

## 环境要求

- Windows
- NVIDIA GPU，支持 HEVC NVDEC/NVENC
- CUDA 可用的 PyTorch 环境
- FFmpeg 位于 `PATH`，并包含 `hevc_cuvid` 与 `hevc_nvenc`
- Ultralytics YOLO
- Streamlit
- 人脸模型文件：`models/face.pt`

加速相关依赖见 `requirements-accel.txt`。

## 回退方案

如果当前机器不支持 NVIDIA HEVC 硬解：

```powershell
--decode-backend opencv
```

如果智能渲染不适用于当前输入编码，可移除：

```powershell
--smart-render
```

程序会使用传统全片渲染路径，结果兼容但耗时更长。

## 已知限制

- CUDA 解码路径当前面向 HEVC 输入，内部使用 `hevc_cuvid`。
- 智能渲染依赖 GOP 边界，输出总帧数可能比源文件少数帧；本次整片验证相差 3 帧，
  音视频时长和首尾解码均正常。
- `cuda` 预缩放模式速度更快，但相对原流程的遮罩一致性低于 `cuda-full`；生产建议使用后者。
- TensorRT 在当前完整视频测试中没有获得稳定收益，因此不是默认后端。
- 审核范围使用原视频绝对时间戳；修改输入视频后不能复用旧审核范围。

## 验证

- `python -m py_compile`：核心脚本及新增工具全部通过。
- `python -m unittest discover -s tests`：17 项自动化测试全部通过。
- 范围审核回归测试：只选择指定时间区间内的遮罩帧，测试通过。
- 37 分钟完整视频：审核生成、291 项人工审核、智能渲染和抽点解码全部完成。

## 相关文档

- `docs/engineering/ACCELERATION_NOTES.md`
- `docs/engineering/FAST_PIPELINE_NOTES.md`
- `docs/engineering/SMART_RENDER_FEASIBILITY.md`
- `docs/engineering/SMART_RENDER_PROTOTYPE_RESULTS.md`
- `docs/engineering/SMART_RENDER_FULL_RUN_RESULTS.md`
