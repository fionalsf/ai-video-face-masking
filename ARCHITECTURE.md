# ARCHITECTURE.md

**项目：** AI Video Face Masking (`new_da_ma`)  
**文档目的：** 固定当前唯一 Production Pipeline，基于**代码实际 import / 调用关系**（非 README 描述）  
**生成日期：** 2026-06-24  
**约束：** 本文档仅描述现状，不代表后续重构计划  

---

## ① 当前唯一 Production Pipeline

生成 `final.mp4` 的完整路径如下。**不经过 Timeline 链**（`timeline_generator.py` / `render_overlay.py`）。

```
Video (.mp4)
    ↓
Detection + Tracking          [pipeline.py → tracker.py]
    ↓
Presence Segmentation         [pipeline.py → event_builder.py]  ※调试产物，非最终 Event
    ↓
Identity Stitching            [pipeline.py → identity_stitching.py]
    ↓
Behavior Events ★            [pipeline.py → identity_behavior_builder.py]
    ↓
Review                        [export_events.py → review_ui.py（人工）]
    ↓
Auto Render（草稿）           [pipeline.py → render.py → masked_draft.mp4]
    ↓
Confirm + Final Render        [confirm.py → render.py]
    ↓
final.mp4
```

**入口命令：**

```powershell
# 单视频
python pipeline.py -i "<video.mp4>" -o "<output_root>"

# 批处理（内部 subprocess 调用 pipeline.py）
python batch.py -i "<input_dir>" -o "<output_root>"

# 人工审核（有 Review 档事件时）
streamlit run review_ui.py -- --review-dir "<output_root>/<stem>/review"

# 最终交付
python confirm.py --output-dir "<output_root>/<stem>"
```

**输出根目录结构：** `<output_root>/<video_stem>/`（由 `safe_video_stem()` 决定）

---

## ②–⑤ Pipeline 分层说明

### Layer 0 — 输入

| 项目 | 内容 |
|------|------|
| **职责** | 提供源视频与模型权重 |
| **负责文件** | （无 Python 模块，纯输入） |
| **输入** | 源视频路径；`models/face.pt` |
| **输出** | — |

---

### Layer 1 — Detection + Tracking

| 项目 | 内容 |
|------|------|
| **职责** | YOLO-face 稀疏 GPU 检测（`interval` 帧采样）+ ByteTrack 分配 `track_id` |
| **负责文件** | `tracker.py`（`run_detect_track()`） |
| **调用方** | `pipeline.py` 第 [1/5] 步 |
| **输入** | 源视频；`models/face.pt`；`video_meta.get_video_meta()` 返回的 fps/frames/width/height |
| **输出（内存）** | 扁平检测列表：每条含 `frame`, `t`, `track_id`, `bbox`, `conf` |
| **输出（文件）** | `<out>/<stem>/tracked_detections.json` |

---

### Layer 2 — Presence Segmentation（Scheme C，调试层）

| 项目 | 内容 |
|------|------|
| **职责** | 按 track 做 presence/absence 分段；生成 Scheme C 调试产物。**代码注释标明 non-production** |
| **负责文件** | `event_builder.py`（`build_events()`）；`event_merge.py`（`save_segmentation_events()`）；`gap_analysis.py`（`save_gap_analysis_debug()`） |
| **调用方** | `pipeline.py` 第 [2/5] 步 |
| **输入** | Layer 1 内存中的 tracked detections；`event_gap`；fps；frames |
| **输出（文件）** | `segmentation_events.json`；`absence_segments.json`；`event_segment_map.json` |
| **说明** | 此层 Event **不是**最终打码 Event 单元；生产 Event 见 Layer 4 |

---

### Layer 3 — Identity Stitching

| 项目 | 内容 |
|------|------|
| **职责** | 跨 track 身份图匹配，将多条 track 聚合为 identity cluster |
| **负责文件** | `identity_stitching.py`（`run_identity_stitching()`） |
| **依赖库** | `appearance_embedder.py`；`identity_graph_matching.py`；`gap_analysis.py`（`group_by_track`） |
| **调用方** | `pipeline.py` 第 [3/5] 步 |
| **输入** | Layer 1 tracked detections |
| **输出（文件）** | `identity_clusters.json`；`identity_graph.json`；（可能）`track_graph.json` |

---

### Layer 4 — Behavior Events ★（生产 Event 单元）

| 项目 | 内容 |
|------|------|
| **职责** | 在 identity cluster 内按时间连续性切分 **behavior events**（`bevt_*`）；三档分类 Auto / Review / LowConf |
| **负责文件** | `identity_behavior_builder.py`（`build_identity_behavior_events()`、`behavior_event_to_face_event()`） |
| **依赖库** | `event_builder.py`（FaceEvent、tier）；`event_merge.py`；`event_quality.py`；`gap_analysis.py`；`rules.py` |
| **调用方** | `pipeline.py` 第 [4/5] 步 |
| **输入** | tracked detections；identity clusters；fps / 帧范围 / detect_interval |
| **输出（文件）** | `behavior_events.json`（**production_events 字段指向此文件**）；`face_events.json`（含 tier，供 Render / Review / Confirm 使用） |

---

### Layer 5 — Review

| 项目 | 内容 |
|------|------|
| **职责** | 导出 Review 档待审事件 + 关键帧预览；人工 Accept/Reject/Skip |
| **负责文件** | `export_events.py`（`export_review_pack()`）；`review_ui.py`；`review_stats.py` |
| **依赖库** | `export.py`（`draw_detections`）；`event_quality.py` |
| **调用方** | `pipeline.py` 第 [5/5] 步（导出）；人工运行 Streamlit（审核） |
| **输入** | 源视频；Review tier 的 `FaceEvent` 列表 |
| **输出（文件）** | `review/pending_events.json`；`review/previews/*.jpg`；`review/confirmed_events.json`；`review/review_report.json`；根目录 `review_report.json`（pipeline 汇总） |

---

### Layer 6 — Auto Render（草稿，Pipeline 内）

| 项目 | 内容 |
|------|------|
| **职责** | 对 **Auto tier** 事件从原视频打码，生成夜间草稿 |
| **负责文件** | `render.py`（`render_masked_output()`） |
| **调用方** | `pipeline.py` 第 [5/5] 步（Auto 分支） |
| **输入** | 源视频；Auto tier event dicts；video meta |
| **输出（文件）** | `masked_draft.mp4`；`audit.log`；`audit.json`（`audit_log.py`）；`low_conf_stats.json`（`low_conf_log.py`，LowConf 档） |

---

### Layer 7 — Confirm + Final Render

| 项目 | 内容 |
|------|------|
| **职责** | 合并 Auto + Review accepted 事件，**从原视频全片重渲染** 为交付成片 |
| **负责文件** | `confirm.py`；`render.py`（`events_to_render()` + `render_video()` + `mux_audio()`） |
| **调用方** | 人工在 Review 完成后执行 |
| **输入** | `face_events.json`；`review/confirmed_events.json`；`masked_draft.mp4`（仅检查存在性）；源视频 |
| **输出（文件）** | **`final.mp4`**；更新 `review_report.json`（`delivery_ready`, `morning_confirmed_at` 等） |
| **说明** | **不读取** `timeline.json`；**不调用** `render_overlay.py`；**不基于** `masked_draft.mp4` 叠加，而是重新打码 |

---

### Timeline 链（非 Production）

| 项目 | 内容 |
|------|------|
| **状态** | **未接入** 上述 Production Pipeline |
| **涉及文件** | `timeline_generator.py` → `timeline.json` → `render_overlay.py` / `timeline_overlay.py` |
| **设计用途** | Phase 2 验证链（track 级 `evt_*` + `confirmed_events.json`）的并行交付路径 |
| **与 Production 关系** | 互斥的 Render 执行器；当前 `final.mp4` 生成**不经过此链** |

---

## 关键模块关系（用户指定）

| 模块 | 与 Production 的关系 |
|------|----------------------|
| **`event_builder.py`** | Pipeline 第 2 步调用，产出 **non-production** 分段调试文件；同时作为 **库** 被 `identity_behavior_builder.py` 引用（FaceEvent、tier、IoU 等） |
| **`identity_behavior_builder.py`** | **生产 Event 唯一来源**（`bevt_*` → `face_events.json`），替代旧 track 级 Event 作为打码单元 |
| **`timeline_generator.py`** | **并行 Experimental 链**；基于 `event_builder.build_events()` + track 级 confirmed 决策；Production **不调用** |
| **`render_overlay.py`** | **并行 Experimental 链** 的 Render 执行器；只读 `timeline.json`；Production **不调用** |
| **`render.py`** | **Production 唯一 Render 引擎**；读 Event trajectory，不读 timeline；被 `pipeline.py` 和 `confirm.py` 各调用一次 |
| **`confirm.py`** | **Production 最终交付编排**；读 Review 决策 + `face_events.json`，调用 `render.py` 产出 `final.mp4` |

---

## 并行 Experimental Pipeline（存在但未用于 final.mp4）

```
Video
  ↓ detect.py                    Phase 1 检测验证
  ↓ validate_track_event.py      Phase 2 跟踪 + Event 验证
  ↓ review_ui.py / event_contact_sheet.py
  ↓ timeline_generator.py        → timeline.json
  ↓ timeline_preview.py / render_overlay.py
  ↓ final_mosaic.mp4（Experimental 交付路径）
```

---

## 全仓库文件分类

状态定义：

| 状态 | 含义 |
|------|------|
| **Production** | 当前 `final.mp4` 路径上的入口、层级主模块或直接产出模块 |
| **Utility** | 被 Production 调用的共享库 / 辅助输出 |
| **Experimental** | 验证链、Benchmark、模块化重构草案、脚本工具，不在 final.mp4 主路径 |
| **Deprecated** | 代码内明确标记废弃或启动即退出 |

### Python 源文件

| 文件 | 状态 | 原因 |
|------|------|------|
| `pipeline.py` | Production | 单视频 Production 主编排入口 |
| `batch.py` | Production | 批处理入口，subprocess 调用 `pipeline.py` |
| `confirm.py` | Production | Review 后最终 `final.mp4` 交付入口 |
| `review_ui.py` | Production | Review 档人工审核 UI |
| `tracker.py` | Production | Layer 1 Detection + Tracking |
| `identity_stitching.py` | Production | Layer 3 Identity Stitching |
| `identity_behavior_builder.py` | Production | Layer 4 生产 Behavior Events |
| `export_events.py` | Production | Layer 5 Review 包导出 |
| `render.py` | Production | Layer 6/7 Render 引擎 |
| `review_stats.py` | Production | Review UI 统计与 `review_report.json` 更新 |
| `audit_log.py` | Production | Auto 档审计日志输出 |
| `low_conf_log.py` | Production | LowConf 档统计输出 |
| `video_meta.py` | Utility | 视频元数据，Production 各层共用 |
| `event_builder.py` | Utility | Pipeline 调试分段 + `identity_behavior_builder` 共用库 |
| `event_merge.py` | Utility | 保存 segmentation 调试文件；behavior builder 内部工具 |
| `gap_analysis.py` | Utility | Track 分组 / absence 分析，多层共用 |
| `event_quality.py` | Utility | Event 质量评分，behavior builder / review UI 共用 |
| `rules.py` | Utility | `suggest_rule_hints()`，event/behavior builder 共用 |
| `export.py` | Utility | Phase 1 导出 + `draw_detections()` 供 review 预览 |
| `appearance_embedder.py` | Utility | Identity stitching 外观特征 |
| `identity_graph_matching.py` | Utility | Identity stitching 图匹配 |
| `detect.py` | Experimental | Phase 1 检测验证独立入口 |
| `validate_track_event.py` | Experimental | Phase 2 跟踪 + Event 验证独立入口 |
| `timeline_generator.py` | Experimental | Timeline 链 Event→timeline，Production 未调用 |
| `timeline_overlay.py` | Experimental | Timeline 渲染核心，供 render_overlay 使用 |
| `render_overlay.py` | Experimental | Timeline 链 Render 入口 |
| `timeline_preview.py` | Experimental | Timeline bbox 预览视频 |
| `timeline_debug_ui.py` | Experimental | Timeline 调试 Streamlit UI |
| `event_contact_sheet.py` | Experimental | Event 可视化 contact sheet / GIF |
| `auto_tuning.py` | Experimental | Phase 1 参数推荐 |
| `behavior_merge_layer.py` | Experimental | 行为合并实验层，Production 未调用 |
| `track_stitching.py` | Experimental | Track 拼接分析实验 |
| `compare_event_merge.py` | Deprecated | 文件头 `DEPRECATED`，启动即 `SystemExit` |
| `benchmark/__init__.py` | Experimental | Benchmark 包 |
| `benchmark/bootstrap_gt.py` | Experimental | GT bootstrap |
| `benchmark/compare_stitching.py` | Experimental | Stitching v1/v2 对比 |
| `benchmark/generate_clips.py` | Experimental | Clip 生成 |
| `benchmark/ground_truth.py` | Experimental | GT 加载 |
| `benchmark/metrics.py` | Experimental | 指标计算 |
| `benchmark/report.py` | Experimental | 报告渲染 |
| `benchmark/run_variant.py` | Experimental | 变体 pipeline 运行 |
| `benchmark/pipelines/__init__.py` | Experimental | Frozen pipeline 包 |
| `benchmark/pipelines/stitching_v1_frozen.py` | Experimental | 冻结 v1 pipeline |
| `benchmark/pipelines/stitching_v2_frozen.py` | Experimental | 冻结 v2 pipeline |
| `modules/__init__.py` | Experimental | 模块化重构包（未接入 Production） |
| `modules/pipeline/runner.py` | Experimental | 简化 orchestrator，无 identity/behavior 完整链 |
| `modules/pipeline/batch.py` | Experimental | 模块化批处理草案 |
| `modules/pipeline/__init__.py` | Experimental | 包初始化 |
| `modules/pipeline/__main__.py` | Experimental | CLI 入口 |
| `modules/detect/detector.py` | Experimental | 模块化检测器 |
| `modules/detect/__init__.py` | Experimental | 包初始化 |
| `modules/detect/__main__.py` | Experimental | CLI 入口 |
| `modules/track/tracker.py` | Experimental | 模块化 ByteTrack |
| `modules/track/__init__.py` | Experimental | 包初始化 |
| `modules/track/__main__.py` | Experimental | CLI 入口 |
| `modules/event/builder.py` | Experimental | 模块化 event builder |
| `modules/event/__init__.py` | Experimental | 包初始化 |
| `modules/event/__main__.py` | Experimental | CLI 入口 |
| `modules/scoring/scorer.py` | Experimental | 模块化 scoring |
| `modules/scoring/rules.py` | Experimental | 模块化 rules |
| `modules/scoring/__init__.py` | Experimental | 包初始化 |
| `modules/scoring/__main__.py` | Experimental | CLI 入口 |
| `modules/render/renderer.py` | Experimental | 模块化 render |
| `modules/render/__init__.py` | Experimental | 包初始化 |
| `modules/render/__main__.py` | Experimental | CLI 入口 |
| `modules/review/export.py` | Experimental | 模块化 review export |
| `modules/review/confirm.py` | Experimental | 模块化 confirm 草案 |
| `modules/review/ui.py` | Experimental | 模块化 review UI 草案 |
| `modules/review/__init__.py` | Experimental | 包初始化 |
| `modules/review/__main__.py` | Experimental | CLI 入口 |
| `core/__init__.py` | Experimental | 模块化 Event 类型包 |
| `core/event.py` | Experimental | 模块化 Event 定义 |
| `core/io.py` | Experimental | 模块化 IO |
| `utils/__init__.py` | Experimental | 供 `modules/` 使用的工具副本 |
| `utils/video_meta.py` | Experimental | `video_meta.py` 的 modules 侧副本 |
| `utils/audit_log.py` | Experimental | `audit_log.py` 的 modules 侧副本 |
| `utils/low_conf_log.py` | Experimental | `low_conf_log.py` 的 modules 侧副本 |
| `utils/draw.py` | Experimental | 绘制工具，modules/review 使用 |
| `scripts/bootstrap_write.py` | Experimental | Bootstrap 代码生成 |
| `scripts/create_modules.py` | Experimental | 模块脚手架生成 |
| `scripts/setup_benchmark.py` | Experimental | Benchmark 环境搭建 |
| `scripts/e2e_apply_review.py` | Experimental | E2E 自动 accept/reject 测试 |
| `scripts/trace_evt_0006.py` | Experimental | 单 Event 追踪调试 |
| `scripts/fix_compare.py` | Experimental | Benchmark 报告修补 |
| `scripts/test_write.py` | Experimental | 写入测试 |

### 配置、文档与数据

| 文件 | 状态 | 原因 |
|------|------|------|
| `requirements.txt` | Utility | Python 依赖清单 |
| `.gitignore` | Utility | Git 忽略规则 |
| `README.md` | Utility | 项目说明（含部分与 Production 现状不一致的并行路径描述） |
| `models/README.md` | Utility | 模型权重说明 |
| `audit_report.md` | Utility | 人工审计报告 |
| `ARCHITECTURE.md` | Utility | 本架构文档 |
| `benchmark/clips/*.json` | Experimental | Benchmark clip 数据 |
| `benchmark/gt/*.json` | Experimental | Benchmark GT 数据 |
| `tools/gen_timeline_ui.py` | Experimental | Timeline UI 生成工具（若存在） |

---

## Production 输出文件一览

`<output_root>/<video_stem>/` 下，一次完整 Production 运行（含 Confirm）典型产物：

| 文件 | 产生层 |
|------|--------|
| `tracked_detections.json` | Layer 1 |
| `segmentation_events.json` | Layer 2（调试） |
| `absence_segments.json` | Layer 2（调试） |
| `event_segment_map.json` | Layer 2（调试） |
| `identity_clusters.json` | Layer 3 |
| `identity_graph.json` | Layer 3 |
| `behavior_events.json` | Layer 4 ★ |
| `face_events.json` | Layer 4 |
| `masked_draft.mp4` | Layer 6 |
| `audit.log` / `audit.json` | Layer 6 |
| `low_conf_stats.json` | Layer 6 |
| `review/pending_events.json` | Layer 5 |
| `review/previews/` | Layer 5 |
| `review/confirmed_events.json` | Layer 5（人工） |
| `review_report.json` | Layer 5 / 7 |
| **`final.mp4`** | **Layer 7** |

**不在 Production 路径产出：** `timeline.json`、`detections.json`（Phase 1）、`event_summary.json`（Phase 2）

---

*本文档由代码静态调用分析生成，反映 2026-06-24 仓库现状。*
