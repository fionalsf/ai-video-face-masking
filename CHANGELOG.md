# Changelog

本项目的显著变更记录在此文件中。版本格式遵循语义化版本。

## [4.0.0] - 2026-07-21

### Added

- CUDA HEVC 硬件解码模式 `cuda` 与 `cuda-full`。
- 独立检测进程及 Torch、ONNX、TensorRT 后端评测工具。
- 基于 GOP 边界的智能渲染，仅重编码包含遮罩的片段。
- 审核 UI 的局部时间范围选择及多范围决定格式。
- 智能渲染的自动化测试与完整视频验证记录。

### Changed

- 正式审核流程改为 `A` 接受、`R` 拒绝、`S` 跳过、`E` 局部修正。
- 生产推荐解码后端改为 `cuda-full`；推理后端仍默认使用 Torch。
- 37 分钟视频的审核数据生成耗时由 20 分 15.5 秒降至 13 分 32.5 秒。
- 37 分钟视频的最终成片耗时由约 28 分 46 秒降至 3 分 53 秒。

### Compatibility

- 继续支持旧版 `accepted_first_half` 与 `accepted_second_half` 审核决定。
- 不支持硬件解码或智能渲染时，可回退到 OpenCV 解码与传统全片渲染。

### Known limitations

- CUDA 解码路径目前面向 HEVC 输入并依赖 NVIDIA NVDEC/NVENC。
- TensorRT 在本轮完整视频测试中没有稳定收益，仍为实验后端。
- 许可证与商业分发条款尚待项目所有者确认。
