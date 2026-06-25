# 代码变更审计报告

**日期：** 2026-06-24  
**仓库：** `new_da_ma` / 远程 `ai-video-face-masking`  
**基准：** Git 提交 `a239036`（Initial commit）+ 工作区未提交变更  
**说明：** 本报告仅审计版本库内文件变更；`output/` 等被 `.gitignore` 忽略的运行产物另见附录。

---

## 1. 修改过的文件

### 1.1 已提交（commit `a239036`，2026-06-24 11:46 +0800）

今日首次提交包含 **93 个文件、14,682 行新增**，均为仓库初始入库。核心入口与业务模块均在此提交中首次加入，无「修改已有文件」的增量（全新仓库）。

### 1.2 未提交（工作区，截至报告生成时）

| 文件 | 变更量 | 变更摘要 |
|------|--------|----------|
| `confirm.py` | +46 / −4 | 新增 `parse_review_decisions()`，支持 Review UI 扁平化 `confirmed_events.json`（`bevt_*: accepted/rejected`）；未决策 Review 事件默认视为 accepted；输出 reject 计数 |
| `review_ui.py` | +50 / −3 | 支持从 `pending_events.json` 加载事件（pipeline 产出）；`load_decisions` 过滤模板元数据键；`media_paths` 支持 `review/previews/` 关键帧预览 |

---

## 2. 新增的文件

### 2.1 Git 已跟踪（commit `a239036`，共 93 个）

<details>
<summary>点击展开完整文件列表</summary>

```
.gitignore
README.md
appearance_embedder.py
audit_log.py
auto_tuning.py
batch.py
behavior_merge_layer.py
benchmark/__init__.py
benchmark/bootstrap_gt.py
benchmark/clips/DJI_20260511100755_0008_D_clips.json
benchmark/compare_stitching.py
benchmark/generate_clips.py
benchmark/ground_truth.py
benchmark/gt/DJI_20260511100755_0008_D_gt_bootstrap.json
benchmark/gt/DJI_20260511100755_0008_D_gt_template.json
benchmark/metrics.py
benchmark/pipelines/__init__.py
benchmark/pipelines/stitching_v1_frozen.py
benchmark/pipelines/stitching_v2_frozen.py
benchmark/report.py
benchmark/run_variant.py
compare_event_merge.py
confirm.py
core/__init__.py
core/event.py
core/io.py
detect.py
event_builder.py
event_contact_sheet.py
event_merge.py
event_quality.py
export.py
export_events.py
gap_analysis.py
identity_behavior_builder.py
identity_graph_matching.py
identity_stitching.py
low_conf_log.py
models/README.md
modules/__init__.py
modules/detect/__init__.py
modules/detect/__main__.py
modules/detect/detector.py
modules/event/__init__.py
modules/event/__main__.py
modules/event/builder.py
modules/pipeline/__init__.py
modules/pipeline/__main__.py
modules/pipeline/batch.py
modules/pipeline/runner.py
modules/render/__init__.py
modules/render/__main__.py
modules/render/renderer.py
modules/review/__init__.py
modules/review/__main__.py
modules/review/confirm.py
modules/review/export.py
modules/review/ui.py
modules/scoring/__init__.py
modules/scoring/__main__.py
modules/scoring/rules.py
modules/scoring/scorer.py
modules/track/__init__.py
modules/track/__main__.py
modules/track/tracker.py
pipeline.py
render.py
render_overlay.py
requirements.txt
review_stats.py
review_ui.py
rules.py
scripts/bootstrap_write.py
scripts/create_modules.py
scripts/e2e_apply_review.py
scripts/fix_compare.py
scripts/setup_benchmark.py
scripts/test_write.py
scripts/trace_evt_0006.py
timeline_debug_ui.py
timeline_generator.py
timeline_overlay.py
timeline_preview.py
tools/gen_timeline_ui.py
track_stitching.py
tracker.py
utils/__init__.py
utils/audit_log.py
utils/draw.py
utils/low_conf_log.py
utils/video_meta.py
validate_track_event.py
video_meta.py
```

</details>

### 2.2 本报告

| 文件 | 状态 |
|------|------|
| `audit_report.md` | 本次新增（未提交） |

### 2.3 运行产物（未纳入 Git，今日实际生成）

路径：`output/DJI_20260510110741_0008_D/`

| 类型 | 代表文件 |
|------|----------|
| 检测/跟踪 | `tracked_detections.json` |
| 身份/行为 | `identity_clusters.json`, `identity_graph.json`, `behavior_events.json`, `face_events.json` |
| 分段调试 | `segmentation_events.json`, `absence_segments.json`, `event_segment_map.json` |
| 自动打码 | `masked_draft.mp4`（≈4.4 GB） |
| 人工审核 | `review/pending_events.json`, `review/confirmed_events.json`, `review/previews/`（276 张） |
| 最终交付 | `final.mp4`（≈4.42 GB） |
| 日志 | `confirm_render.log`, `audit.log`, `review_report.json` 等 |

---

## 3. 删除的文件

### 3.1 Git 跟踪文件

**无。** 当前工作区与 `main` 分支相比，没有已跟踪文件的删除记录。

### 3.2 历史清理（非工作区删除）

今日曾尝试将大体积 `.mp4` 纳入版本库，后通过 orphan 分支重写历史移除；**最终 commit 未包含任何视频文件**。`.gitignore` 已加入 `*.mp4` / `*.MP4` 规则。

本地工作区中，原 pipeline 首次 commit 涉及的 5 个 DJI 视频及 `test456.mp4` 未保留在仓库根目录（仅存在于被丢弃的早期 Git 对象中）。

---

## 4. 哪些模块进入了生产 Pipeline

今日实际执行路径（`DJI_20260510110741_0008_D.MP4`）：

```
batch.py（可选批处理入口）
  └─ pipeline.py（单视频主入口，今日使用）
       ├─ [1/5] tracker.py          — YOLO 检测 + ByteTrack 跟踪
       ├─ [2/5] event_builder.py   — Presence 分段（Scheme C，调试产物）
       │         event_merge.py     — 分段保存（非最终 Event 单元）
       │         gap_analysis.py    — Track 分组辅助
       ├─ [3/5] identity_stitching.py
       │         appearance_embedder.py
       │         identity_graph_matching.py
       ├─ [4/5] identity_behavior_builder.py  — ★ 生产 Event 单元
       │         event_quality.py   — 质量评分
       ├─ [5/5] render.py            — Auto 档打码 → masked_draft.mp4
       │         export_events.py   — Review 包 → review/pending_events.json
       │         audit_log.py / low_conf_log.py
       └─ review_report.json 生成

人工审核：
  review_ui.py  — Streamlit Review UI

最终交付：
  confirm.py    — 合并 Auto + Review accepted → final.mp4
  render.py     — 全片重渲染
  video_meta.py — 元数据
  review_stats.py — 审核统计
```

**生产 Pipeline 判定标准：** 被 `pipeline.py` → `batch.py` 直接或间接调用，且产出可交付视频或审核包的模块。

| 层级 | 模块 | 角色 |
|------|------|------|
| 入口 | `pipeline.py`, `batch.py` | 夜间批处理 / 单视频 |
| 检测跟踪 | `tracker.py`, `export.py`（draw） | GPU 稀疏检测 + ByteTrack |
| 事件构建 | `identity_behavior_builder.py` | **生产 Event**（`behavior_events.json`） |
| 身份拼接 | `identity_stitching.py`, `appearance_embedder.py`, `identity_graph_matching.py` | Track → Identity 聚类 |
| 辅助 | `event_builder.py`, `event_merge.py`, `gap_analysis.py`, `event_quality.py` | 分段调试 + 行为 Event 内部逻辑 |
| 渲染 | `render.py` | ffmpeg + h264_nvenc 马赛克 |
| 审核导出 | `export_events.py`, `review_ui.py`, `review_stats.py` | Review 档导出与 UI |
| 交付 | `confirm.py` | 审核后 `final.mp4` |
| 基础设施 | `video_meta.py`, `audit_log.py`, `low_conf_log.py`, `rules.py` | 元数据 / 审计 / 规则 |

**未走 Timeline 分支：** 今日未调用 `timeline_generator.py` / `render_overlay.py` 路径，直接使用 `pipeline.py` + `confirm.py` 渲染链。

**`modules/` 包：** 平行的模块化重构（`modules/pipeline/runner.py` 等），**今日未使用**；生产仍走根目录 monolith 脚本。

---

## 5. 哪些模块只是实验代码

以下模块 **未进入今日生产运行路径**，用于验证、基准测试、调试或已被替代：

### 5.1 分阶段验证（Phase 1 / Phase 2）

| 模块 | 用途 |
|------|------|
| `detect.py` | Phase 1：仅 YOLO 检测验证 |
| `validate_track_event.py` | Phase 2：跟踪 + Event 验证（读 Phase 1 输出） |
| `event_contact_sheet.py` | Event 可视化 contact sheet / GIF |
| `auto_tuning.py` | 基于 `detection_summary.json` 的参数推荐 |

### 5.2 Timeline 替代渲染链（与今日 pipeline 并行，未使用）

| 模块 | 用途 |
|------|------|
| `timeline_generator.py` | 从 `confirmed_events.json` 生成 `timeline.json` |
| `timeline_preview.py` | Timeline bbox 叠加预览视频 |
| `timeline_overlay.py` | Timeline 马赛克/overlay 执行器 |
| `render_overlay.py` | Timeline 驱动渲染（overlay / preview / final） |
| `timeline_debug_ui.py` | Timeline 调试 Streamlit UI |

### 5.3 Benchmark / 算法对比

| 模块 | 用途 |
|------|------|
| `benchmark/*` | Stitching v1/v2  frozen pipeline 对比、GT、指标、报告 |
| `compare_event_merge.py` | **已废弃**（deprecated 提示） |
| `behavior_merge_layer.py` | 行为合并实验层（未被 `pipeline.py` 调用） |
| `track_stitching.py` | Track 拼接分析实验 |

### 5.4 模块化重构草案（未接线到 batch）

| 模块 | 用途 |
|------|------|
| `modules/pipeline/runner.py` | 模块化 orchestrator（无 identity stitching 完整链） |
| `modules/detect/*`, `modules/track/*`, `modules/event/*` | 拆分后的 detect/track/event |
| `modules/render/*`, `modules/review/*`, `modules/scoring/*` | 拆分后的 render/review/scoring |
| `core/*` | 模块化 Event 类型定义 |

### 5.5 脚本与工具

| 模块 | 用途 |
|------|------|
| `scripts/bootstrap_write.py` | 代码/bootstrap 生成 |
| `scripts/create_modules.py` | 模块脚手架生成 |
| `scripts/setup_benchmark.py` | Benchmark 环境搭建 |
| `scripts/e2e_apply_review.py` | E2E 自动 accept/reject（测试用） |
| `scripts/trace_evt_0006.py` | 单 Event 追踪调试 |
| `scripts/fix_compare.py` | Benchmark 报告修补 |
| `scripts/test_write.py` | 写入测试 |
| `tools/gen_timeline_ui.py` | Timeline UI 生成工具 |

---

## 附录 A：今日 Git 时间线

| 时间 | 事件 |
|------|------|
| 上午 | 初始化仓库，首次 commit（含大视频，后重写） |
| 11:46 | 干净历史 `a239036`：93 文件，无视频 |
| 下午 | 修改 `review_ui.py`、`confirm.py`（未提交） |
| 下午 | 运行 `pipeline.py` → `review_ui.py` → `confirm.py` 完成 `final.mp4` |

## 附录 B：待办建议

1. **提交** `confirm.py`、`review_ui.py` 今日修复，避免 Review → Render 链路丢失。
2. **统一** `confirmed_events.json` 格式（扁平 map vs 模板结构），防止元数据键污染。
3. **明确** `modules/` 与根目录脚本的关系：合并或标注其一为 deprecated。

---

*报告由自动化审计生成，基于 `git log` / `git diff` / `git status` 及今日实际运行命令链分析。*
