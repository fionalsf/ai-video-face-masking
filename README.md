# new_da_ma — Event-Based 工厂人脸打码

夜间无人值守批处理 + 早上 Streamlit Review UI 确认。

## 安装

```powershell
cd "D:\work\制造业视频数据\new_da_ma"
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge ffmpeg -y
```

模型：`models/face.pt`（见 `models/README.md`）

## Auto Tuning — 参数自动推荐

根据 Phase-1 `detection_summary.json`（视频统计 + 检测密度 + confidence 直方图）推荐 pipeline 参数：

```powershell
python auto_tuning.py --detection-dir "output/detection/test456"
python auto_tuning.py --detection-dir "output/detection/test456" --write
```

输出字段：

| 字段 | 说明 |
|------|------|
| `recommended_conf` | YOLO 置信度阈值 |
| `recommended_event_gap` | Event 切分间隔（秒） |
| `recommended_padding` | `pre/post_padding_sec` 与对应帧数 |
| `recommended_mosaic_level` | 渲染马赛克强度 |

`--write` 会额外写入 `tuning_recommendations.json`（含 rationale）。

## 夜间批处理

```powershell
python batch.py -i "D:/视频/待处理" -o "D:/打码输出" --device 0
```

## 单视频

```powershell
python pipeline.py -i "D:/视频/test123.mp4" -o output --device 0
```

输出：`output/test123/`
- `masked_draft.mp4` — Auto（peak≥0.85）已打码
- `review/pending_events.json` — Review 档待确认
- `audit.log` / `low_conf_stats.json`
- `review_report.json`

## 早上 Review UI（Event 审核 v1）

```powershell
streamlit run review_ui.py -- --output-dir "output/detection/test456"
```

数据来源：`event_summary.json` + `event_contact_sheet/` + `event_gifs/`  
实时保存：`confirmed_events.json`（`{"evt_0001": "accepted", ...}`）

快捷键：**A** Accept · **R** Reject · **S** Skip · **←/→** 切换 Event

每次保存决策时同步更新 `review_report.json`（含 accept/reject/skip 比例与分析指标）。

## Review Statistics（审核统计）

审核过程中顶部栏实时显示 **Total / Accepted / Rejected / Skipped / Remaining** 及百分比。

审核结束后 `review_report.json` 示例：

```json
{
  "video": "test456.mp4",
  "total_events": 10,
  "accepted": 9,
  "rejected": 1,
  "skipped": 0,
  "accept_rate": 90.0,
  "reject_rate": 10.0,
  "skip_rate": 0.0,
  "review_finished_at": "2026-06-21 14:30:21",
  "analysis": {
    "accepted_avg_peak_confidence": 0.7526,
    "rejected_avg_peak_confidence": 0.8085,
    "accepted_avg_duration_sec": 0.9444,
    "rejected_avg_duration_sec": 2.0,
    "accepted_avg_frame_count": 6.11,
    "rejected_avg_frame_count": 12.0
  }
}
```

也可单独刷新统计：

```powershell
python -c "from review_ui import load_events, load_decisions; from review_stats import update_review_report; od='output/detection/test456'; e,v=load_events(od); update_review_report(od,v,e,load_decisions(od))"
```

## Timeline Generator（打码时间轴）

根据 `confirmed_events.json`（仅 Accepted）+ `tracked_detections.json` 生成连续 bbox 时间轴。**不调用 ffmpeg，不生成视频。**

```powershell
python timeline_generator.py --output-dir "output/detection/test456"
```

输出 `timeline.json`：每条 entry 含 `frame`、`timestamp`、`track_id`、`bbox`、`confidence`（可选）、`event_id`。检测帧之间的间隔使用线性插值；Event 边界额外扩展 **temporal padding**（默认 `detect_interval × 2` 帧，约 0.33s@30fps），首尾帧用首/末检测 bbox 填充，避免稀疏采样导致的 2~3 帧漏打码。

## Timeline Debug UI

```powershell
streamlit run timeline_debug_ui.py -- --output-dir "output/detection/test456"
```

支持 Overview / By Event / By Time / By Frame 查看，确认 Timeline 正确后再进入 Render 模块。

## Timeline Preview Overlay（整段视频 bbox 验证）

读取 `timeline.json`，**不打码**，在整段视频上绘制绿色 bbox + Event/Track/Frame/Timecode，用于验证轨迹是否正确。

```powershell
python timeline_preview.py --video test456.mp4 --timeline output/detection/test456/timeline.json
```

默认输出：`output/debug/timeline_preview.mp4`

验证要点：
- bbox 是否跟随人脸移动
- 是否存在漂移或帧间跳跃
- Event 起止时间是否正确
- Rejected Event 不应出现（timeline 仅含 Accepted）

Timeline 验证通过后，Render 模块将直接复用 `timeline.json`，不再重新计算 bbox。

## Render — Timeline 马赛克执行器

严格只读 `timeline.json`，**block pixelation 马赛克**（无 Gaussian blur）。Render 是执行器，不是智能模块。

| 模式 | 说明 | 默认输出 |
|------|------|----------|
| `overlay` | 仅绘制 bbox（调试） | `output/render/overlay.mp4` |
| `preview` | 马赛克，审核验证 | `output/render/preview_mosaic.mp4` |
| `final` | 马赛克，最终交付 | `output/render/final_mosaic.mp4` |

```powershell
# 调试：bbox overlay
python render_overlay.py --video test456.mp4 --timeline output/detection/test456/timeline.json --mode overlay

# 审核：马赛克预览
python render_overlay.py --video test456.mp4 --timeline output/detection/test456/timeline.json --mode preview

# 最终输出（隐私推荐 extreme）
python render_overlay.py --video test456.mp4 --timeline output/detection/test456/timeline.json --mode final --mosaic_level extreme
```

马赛克强度 `--mosaic_level`（preview/final 模式）：

| 级别 | downscale | 说明 |
|------|-----------|------|
| `low` | 0.3 | 轻度 |
| `medium` | 0.15 | 中等 |
| `high` | 0.08 | 默认 |
| `extreme` | 0.03 | 隐私推荐，人脸轮廓不可识别 |

算法：打码前 bbox 扩展 20%（`expand_ratio=0.2`），ROI 按 downscale 比例缩小再放大，**全程 `cv2.INTER_NEAREST`**（禁止 linear/cubic）。

调试开关（overlay 默认全开，preview/final 默认全关）：

```powershell
python render_overlay.py ... --mode preview --show_bbox --show_event_id
```

## 合并交付（旧 pipeline）

```powershell
python confirm.py --output-dir "output/test123"
```

输出 `final.mp4`

## 三档策略

| 档位 | peak_conf | 行为 |
|------|-----------|------|
| Auto | ≥ 0.85 | 自动打码 + audit.log |
| Review | 0.75 ~ 0.85 | 3 关键帧 + UI 确认 |
| LowConf | < 0.75 | 仅 low_conf_stats.json |

## 阶段一：Detection Validation（检测验证）

仅 YOLO-face 检测，不含 tracking / event / review / render。

```powershell
python detect.py --video "D:/视频/test123.mp4" --device 0
```

输出目录：`output/detection/test123/`

| 文件 | 说明 |
|------|------|
| `detections.json` | 所有有人脸的采样帧及 bbox |
| `detection_summary.json` | 统计 + confidence_histogram |
| `review_images/` | 每张检测截图，如 `frame_000660_conf_0.720.jpg` |

验证目标：真实人脸召回、手部/工具误检、各 confidence 区间误检率。

## 阶段二：Tracking + Event 验证

基于阶段一 `detections.json`，不重新检测、不打码、无 UI。

```powershell
python validate_track_event.py --detection-dir "output/detection/test123"
```

输出（同目录）：

| 文件 | 说明 |
|------|------|
| `track_summary.json` | Track 总数、每条 duration/detection 数、最长/最短 |
| `event_summary.json` | Event 总数、每条 duration/frame 数、最长/最短、压缩比 |
| `event_preview.json` | 每个 event 的 peak/avg conf + 中间帧截图路径 |
| `event_previews/` | 代表截图（middle frame + bbox） |

Event 时间范围在首尾检测点基础上扩展 temporal padding（`pre/post = detect_interval × 2` 帧，或默认 0.25s / 0.4s），`start_time ≥ 0`，不改变 trajectory 与 `event_gap` 切分逻辑。
| `tracked_detections.json` | ByteTrack 后的带 track_id 检测列表 |

## Event 可视化验证（Contact Sheet）

不修改 Detection / Tracking / Event Builder，用于验证每个 Event 是否代表连续人脸出现。

```powershell
python event_contact_sheet.py --detection-dir "output/detection/test456"
```

输出：`output/detection/test456/event_contact_sheet/` + `event_gifs/`
- `evt_0001.jpg` … 每个 Event 的时间序 contact sheet（≤9 帧全出，>9 帧均匀抽 9 张）
- `event_gifs/evt_XXXX.gif` — duration **> 2s** 的 Event 自动生成 **4fps** GIF（可调 `--gif-fps`）
- `summary.html` — 每 Event 一行元数据 + 缩略图，点击看原图

## 阶段三：Event Pipeline（夜间打码）
