# DRIFT_ANALYSIS.md

**分析对象：** `DJI_20260510153703_0003_D.MP4`  
**输出目录：** `output_production_DJI_20260510153703_0003_D/DJI_20260510153703_0003_D/`  
**成片：** `final.mp4`（由 `confirm.py` → `render.py` 生成）  
**分析日期：** 2026-06-24  
**约束：** 只读分析，未修改任何代码  

---

## 结论摘要

| 问题 | 结论 |
|------|------|
| **漂移首次出现在哪一层？** | **分类型**：① 检测帧上的 bbox 抖动 → **Detection/Tracking**；② 跨 track 合并处的大跳变 → **Identity Stitching**（体现在 `behavior_events.json`）；③ 非检测帧上的马赛克滑动/滞后 → **Render**（`render.py` 首次**合成**中间帧 bbox） |
| Detection bbox 是否已漂移？ | **是**。稀疏检测帧（interval=5）上已有抖动，相邻采样帧中心平均位移 **11.8px**，最大 **110px**，IoU 最低 **0.021** |
| Track 是否漂移？ | **检测帧上等同 Detection**；ByteTrack **不输出**中间帧 bbox，**不做**帧间平滑 |
| Behavior Event 是否 bbox 插值？ | **否**。`trajectory` 与 `tracked_detections.json` **逐点完全一致**（3315/3315，0 偏差） |
| Render 是否平滑/插值？ | **有线性插值，无平滑**。`events_to_render()` 在稀疏轨迹点之间线性插值；约 **85.8%** 遮罩帧为插值合成帧 |

---

## 1. 数据流与检查顺序

```
tracked_detections.json          ← Layer 1: Detection + Tracking
        ↓（原样拷贝，无插值）
behavior_events.json             ← Layer 4: identity_behavior_builder
        ↓（face_events.json 同 trajectory）
confirm.py → render.py           ← Layer 7: events_to_render() + render_video()
        ↓
final.mp4
```

**未经过：** `timeline_generator.py`（无 `timeline.json`）

---

## 2. Layer 1 — `tracked_detections.json`（Detection + Tracking）

**负责模块：** `tracker.py`（`gpu_sparse_detect` + `cpu_byte_track`）

### 行为（代码）

- YOLO 每 **5 帧**采样一次（`vid_stride=interval=5`）
- ByteTrack **仅在采样帧**更新 bbox
- **不生成** frame+1…+4 的 bbox
- **不做** bbox 时间平滑

### 实测（本视频）

| 指标 | 值 |
|------|-----|
| 检测记录总数 | 3315 |
| Track 数 | 231 |
| 有检测的帧数 | 2304 / 22646（**10.2%**） |
| 相邻 interval=5 采样帧中心位移 | 平均 **11.77px**，最大 **110.1px** |
| 相邻 interval=5 采样帧 IoU | 平均 **0.759**，最小 **0.021** |
| 非 5 帧间隔的 gap（track 断裂/重捕获） | 99 处，最大位移 **470.7px**（gap=90 帧） |

### 判断

**Detection bbox 在采样帧上已经存在抖动与跳变**，这是漂移的**最早数据源**。  
Track 层不额外修正 bbox，仅在采样帧关联 `track_id`；因此 **Track 漂移 ≡ 检测帧上的 Detection 漂移**。

---

## 3. Layer 4 — `behavior_events.json`（Identity Behavior Builder）

**负责模块：** `identity_behavior_builder.py`

### 行为（代码）

- `trajectory` 由 tracked detection **chunk 直接组装**（`chunk` → `traj` 列表）
- **无**帧间 bbox 插值、**无**平滑滤波
- Identity stitching 可将 **多个 track_id** 并入同一 behavior event

### 实测（本视频）

| 指标 | 值 |
|------|-----|
| Behavior events | 171 |
| Trajectory 点数 | 3315（与 tracked 记录数相同） |
| 与 tracked_detections bbox 一致性 | **3315/3315 完全匹配**，max diff **0** |
| 含多个 track_id 的 event | **43** |
| track_id 切换边界的 bbox 跳变 | **63** 处，平均 **117.9px**，最大 **788.8px** |

### 典型跳变案例：`bevt_0140`

Identity 合并 track 1211 → 1226 → 1229 → 1233，边界处 bbox 突变：

| 帧 | track_id | bbox |
|----|----------|------|
| 17070 | 1211 | [102.5, 207.9, 217.0, 328.7] |
| 17210 | 1226 | [428.3, 315.0, 1217.5, 1075.8]（Δcenter ≈ **789px**） |
| 17230 | 1226 | [587.3, 280.5, 1383.3, 1049.0] |
| 17315 | 1229 | [1572.5, 476.1, 1708.9, 567.6] |
| 17335 | 1233 | [1674.2, 702.9, 1894.0, 861.6] |

### 判断

**Behavior Event 层未做 bbox 插值**，但 **Identity Stitching 合并多 track 时引入空间跳变**，写入 `behavior_events.json` 的 trajectory。  
这是 **大幅度“瞬移”式漂移** 的来源，早于 Render，晚于单 track 内的 Detection 抖动。

---

## 4. Layer 7 — `render.py` 输入 bbox（Confirm → Final Render）

**负责模块：** `confirm.py` → `render.py`（`events_to_render` + `render_video`）

### 行为（代码）

`events_to_render()`（`render.py` 第 37–70 行）：

1. 在 trajectory **已有检测帧**写入原始 bbox  
2. 在相邻检测帧之间 **线性插值**（`b0 + (b1-b0) * t`）  
3. 首帧前/末帧后各 **extend_frames=3** 帧 hold 边界 bbox  
4. **无** Kalman/指数平滑/IoU 约束  

`render_video()` 仅按 `render` dict 逐帧画 mask，**不再修改** bbox。

### Confirm 输入

- 读 `face_events.json`（trajectory 与 `behavior_events.json` 相同）
- Auto 26 + Review accepted 103 = **129** 个打码 event
- 日志：遮罩帧 **14193** 帧

### 实测（插值占比，基于 129 个打码 event 模拟 `events_to_render`）

| 指标 | 值 |
|------|-----|
| 打码帧位总数（event 合并后） | 22223 frame-slots |
| **纯插值帧**（无对应检测点） | **19062（85.8%）** |
| 示例 event `bevt_0171` | 249 检测点 → 1262 打码帧，其中 **1013** 帧为插值 |

### 判断

**Render 是首个为“非检测帧”合成 bbox 的层。**  
若人脸运动非线性（加速/转向），线性插值会导致马赛克 **滞后或偏离真实位置**——这是 `final.mp4` 中 **连续滑动式漂移** 的主要来源。

Render **有插值、无平滑**。

---

## 5. `final.mp4` 层

**文件：** `final.mp4`（与 `confirm.py` 输出的 `final.mp4` 相同）

- 马赛克位置 **完全由** `render.py` 的 `render` dict 决定  
- **不再**做检测、跟踪、Event 构建或二次 bbox 修正  
- 可见漂移 = Detection 抖动 + Identity 跳变 + Render 线性插值的 **叠加结果**

---

## 6. 四层检查表（用户要求）

| 检查项 | Detection<br>`tracked_detections.json` | Behavior Events<br>`behavior_events.json` | Render 输入<br>`render.py` | `final.mp4` |
|--------|----------------------------------------|-------------------------------------------|----------------------------|-------------|
| bbox 是否已漂移？ | **是**（采样帧抖动） | **是**（继承 Detection + stitch 跳变） | **是**（叠加线性插值） | **是**（最终可见） |
| 是否插值？ | **否**（仅 10.2% 帧有 bbox） | **否**（原样拷贝） | **是**（线性，85.8% 遮罩帧） | — |
| 是否平滑？ | **否** | **否** | **否** | — |
| 漂移首次出现 | **采样帧 bbox 抖动：此层** | **多 track 合并跳变：此层** | **中间帧合成 bbox：此层** | 表现层 |

---

## 7. 漂移首次出现 — 分层判定

```
Video
  ↓
Detection + Tracking (tracker.py)
  │  ★ 首次：采样帧 YOLO bbox 抖动（avg 11.8px / 5帧）
  │  ★ 首次：track 断裂大跳变（最大 470px）
  ↓
Identity Stitching (identity_stitching.py)
  │  ★ 首次：跨 track 合并边界跳变（最大 789px，写入 behavior trajectory）
  ↓
Behavior Events (identity_behavior_builder.py)
  │  无插值；trajectory = tracked 拷贝
  ↓
Review (export_events.py / review_ui.py)
  │  不影响 bbox
  ↓
Timeline                          ❌ 未经过
  ↓
Render (render.py)
  │  ★ 首次：非检测帧 bbox 合成（线性插值，85.8% 遮罩帧）
  ↓
Confirm (confirm.py)
  ↓
final.mp4
```

### 对用户可见的「马赛克漂移」

| 漂移形态 | 首次引入层 |
|----------|------------|
| 每 ~5 帧轻微抖动 | **Detection / Tracking** |
| 马赛克突然跳到另一位置 | **Identity Stitching**（合并 track 边界） |
| 两检测帧之间马赛克匀速滑动但与脸不同步 | **Render**（线性插值） |

---

## 8. 关键代码引用（只读，未修改）

### `tracker.py` — 稀疏检测，无中间帧 bbox

```python
# vid_stride=interval → 每 5 帧一个检测点
results = model.predict(..., vid_stride=interval, ...)
# ByteTrack 仅在 sparse_dets 的 frame_idx 上运行
for frame_idx in sorted(sparse_dets):
    tracks = tracker.update(res, img=None)
```

### `identity_behavior_builder.py` — trajectory 原样拷贝

```python
traj = [
    {"t": d["t"], "frame": d["frame"], "bbox": d["bbox"], ...}
    for d in chunk  # chunk 来自 tracked detections
]
```

### `render.py` — 线性插值（无平滑）

```python
for f in range(f0 + 1, f1):
    t = (f - f0) / gap
    render[f].append((b0 + (b1 - b0) * t).tolist())
```

---

## 9. 分析限制

- 未对 `final.mp4` 逐帧跑人脸检测做 ground-truth 对比（Run Only / 无代码修改约束下，仅做 JSON + 代码路径分析）
- `masked_draft.mp4`（Auto-only 渲染）与 `final.mp4`（Confirm 全 event 重渲染）使用 **同一** `events_to_render` 逻辑，漂移机制相同
- 本报告数据来自 `output_production_DJI_20260510153703_0003_D` 一次 Production 运行

---

*本文档为只读漂移定位分析，不包含修复建议或代码变更。*
