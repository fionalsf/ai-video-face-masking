# EVENTS_TO_RENDER_ANALYSIS.md

**分析对象：** `render.py` → `events_to_render()`  
**约束：** 只读分析，未修改任何代码  
**关联调用：** `render_masked_output()`（`pipeline.py`）、`confirm.py` 直接调用  

---

## 函数签名与返回值

```37:41:d:\work\制造业视频数据\new_da_ma\render.py
def events_to_render(
    events: list[dict],
    total_frames: int,
    extend_frames: int = 3,
) -> dict[int, list[list[float]]]:
```

| 参数 | 来源 | 默认值 | 是否 CLI 可调 |
|------|------|--------|---------------|
| `events` | `face_events.json` / Auto+Review event dicts | — | — |
| `total_frames` | `video_meta.get_video_meta()` → `meta["frames"]` | — | — |
| `extend_frames` | 函数默认 | **3** | **否**（`pipeline.py` / `confirm.py` 均未传入） |

**返回值：** `render: dict[int, list[list[float]]]`  
键 = 视频帧号，值 = 该帧所有待打码 bbox 列表（多 event 重叠时一帧可有多个 bbox）。

---

## 逐步执行流程

对 **每一个 event** 的 `trajectory` 依次执行以下四步，最后合并到全局 `render` dict。

```
for ev in events:
    traj = ev.get("trajectory") or []
    if not traj: continue

    Step A  写入原始检测点
    Step B  相邻检测点之间线性插值
    Step C  首检测帧前 temporal hold
    Step D  末检测帧后 temporal hold
```

---

### Step A — 写入 trajectory 原始检测点

**代码：** 第 49–50 行

```49:50:d:\work\制造业视频数据\new_da_ma\render.py
        for i, pt in enumerate(traj):
            render[pt["frame"]].append(list(pt["bbox"]))
```

**行为：**

- 遍历 `trajectory` 中每个点 `pt`
- 在 `render[pt["frame"]]` 追加 **原始 bbox**（不做任何变换）
- 这些点来自上游稀疏检测（Production 默认每 5 帧一个）

**此步不生成新 bbox**，只登记已有检测帧。

---

### Step B — 相邻检测点之间线性插值

**代码：** 第 51–59 行

```51:59:d:\work\制造业视频数据\new_da_ma\render.py
        for i in range(len(traj) - 1):
            f0, b0 = traj[i]["frame"], np.asarray(traj[i]["bbox"], dtype=np.float64)
            f1, b1 = traj[i + 1]["frame"], np.asarray(traj[i + 1]["bbox"], dtype=np.float64)
            gap = f1 - f0
            if gap <= 1:
                continue
            for f in range(f0 + 1, f1):
                t = (f - f0) / gap
                render[f].append((b0 + (b1 - b0) * t).tolist())
```

#### 1. 什么时候开始插值？

满足 **同时** 以下条件时开始：

| 条件 | 含义 |
|------|------|
| `len(traj) >= 2` | 至少两个检测点 |
| 进入内层循环 `for i in range(len(traj) - 1)` | 取相邻点对 `(traj[i], traj[i+1])` |
| `gap = f1 - f0 > 1` | 两检测帧 **不连续**（中间至少缺 1 帧） |

若 `gap <= 1`（两检测点在同一帧或相邻帧），**跳过插值**（`continue`）。

Production 默认 `detect_interval=5` → 相邻 trajectory 点通常 `gap=5` → **几乎总是触发插值**。

#### 2. 插值跨度是多少？

对每一对 `(f0, b0)` 与 `(f1, b1)`：

- **帧跨度：** `f = f0+1, f0+2, …, f1-1`（**不含** `f0` 和 `f1` 本身）
- **插值帧数：** `gap - 1` 帧  
  - 例：`f0=100, f1=105` → `gap=5` → 插值帧 **101, 102, 103, 104**（共 4 帧）
- **bbox 跨度：** 从 `b0` 四维坐标 `[x1,y1,x2,y2]` 线性过渡到 `b1`，四个分量独立插值

**全视频尺度（本仓库实测，DJI_20260510153703）：**

- trajectory 检测点：3315 个
- 插值后打码帧位：约 **85.8%** 为无检测点的合成帧

#### 3. 为什么是线性插值？

**代码层面：** 第 58–59 行使用标准线性公式：

```python
t = (f - f0) / gap          # t ∈ (0, 1)，均匀步进
bbox(f) = b0 + (b1 - b0) * t
```

**设计文档层面：**

- `events_to_render()` **内无注释** 解释为何选线性
- 同仓库 `timeline_generator.py` 的 `interpolate_event_trajectory()` 也使用帧间线性插值（分段策略更复杂，但段内仍线性）
- README 对 Timeline 链描述为「检测帧之间的间隔使用**线性插值**」（`README.md` 第 109 行）

**可推断原因（非代码明示）：**

1. 实现最简单：NumPy 向量运算一行完成  
2. 计算量小：逐 event、逐相邻对、逐中间帧  
3. 无需额外状态（无 Kalman、无速度模型）  

**未采用：** 样条、匀速/匀加速模型、IoU 约束、人脸 landmark 跟踪。

---

### Step C — 首检测帧前 temporal hold（padding）

**代码：** 第 60–63 行

```60:63:d:\work\制造业视频数据\new_da_ma\render.py
        f_first = traj[0]["frame"]
        b_first = traj[0]["bbox"]
        for f in range(max(0, f_first - extend_frames), f_first):
            render[f].append(list(b_first))
```

**行为：**

- 在 **第一个检测帧 `f_first` 之前**，向前延伸 `extend_frames` 帧  
- 这些帧使用 **与首检测点相同的 bbox**（hold，非插值）

**帧范围：** `[max(0, f_first - 3), f_first)` → 最多 **3 帧**

---

### Step D — 末检测帧后 temporal hold（padding）

**代码：** 第 64–68 行

```64:68:d:\work\制造业视频数据\new_da_ma\render.py
        f_last = traj[-1]["frame"]
        b_last = traj[-1]["bbox"]
        end = total_frames if total_frames > 0 else f_last + extend_frames + 1
        for f in range(f_last + 1, min(end, f_last + 1 + extend_frames)):
            render[f].append(list(b_last))
```

**行为：**

- 在 **最后一个检测帧 `f_last` 之后**，向后延伸最多 `extend_frames` 帧  
- 使用 **与末检测点相同的 bbox**（hold）

**帧范围：** `(f_last, min(total_frames, f_last + 3)]` → 最多 **3 帧**

注意：`min(end, f_last + 1 + extend_frames)` 中 `f_last + 1 + extend_frames` = `f_last + 4`，range 上界不含 → 实际为 `f_last+1, f_last+2, f_last+3`。

---

### Step E — 多 event 合并

函数对 **每个 event** 重复 A–D，写入 **同一个** `render` dict。

- 同一帧多个 event 重叠 → `render[f]` 为 **list**，含多个 bbox  
- **无** bbox 合并、IoU 去重、优先级规则  

---

## 后续：`render_video()` 中的 expand（不属于 `events_to_render`，但影响成片）

`events_to_render()` 返回后，`render_video()` 绘制 mask 时对每个 bbox 做 **空间扩展**：

```107:108:d:\work\制造业视频数据\new_da_ma\render.py
    expand: float = 0.18,
```

```155:158:d:\work\制造业视频数据\new_da_ma\render.py
                ex1 = int(round((x1 - bw * expand) * sx))
                ey1 = int(round((y1 - bh * expand) * sy))
                ex2 = int(round((x2 + bw * expand) * sx))
                ey2 = int(round((y2 + bh * expand) * sy))
```

- 宽高各向外扩 `bbox_size × expand`  
- **不改变** `render` dict 中存储的 bbox 坐标，只扩大 mask 绘制区域  

---

## 4. 为什么 padding 是 3 帧？

| 事实 | 说明 |
|------|------|
| 代码默认值 | `extend_frames: int = 3`（第 40 行） |
| 调用方 | `pipeline.py` / `confirm.py` 调用 `events_to_render(events, meta["frames"])` **未覆盖**此参数 |
| 代码注释 | **无** 解释为何是 3 |
| 时间意义 @30fps | 3 帧 ≈ **0.1 秒** |

**同仓库其他参考（非 `events_to_render` 直接引用）：**

- README 描述 Timeline 链 temporal padding 为 `detect_interval × 2` 帧（默认 interval=5 → **10 帧**）  
- `events_to_render` 的 **3 帧** 比 Timeline 链 padding **更短**，二者 **不一致**

**可推断意图（非代码明示）：**

- 在 event 起止边界少量补帧，减少「检测首帧才出现马赛克」的闪烁  
- 3 是硬编码常数，**非**从 `detect_interval` 或 fps 动态计算  

---

## 5. 为什么 expand 是 18%？

| 事实 | 说明 |
|------|------|
| `render_video()` 默认 | `expand: float = 0.18`（第 107 行） |
| `pipeline.py` CLI 默认 | `--expand` default **0.18**（第 41 行） |
| `confirm.py` CLI 默认 | `--expand` default **0.18**（第 21 行） |
| 代码注释 | **无** 解释 0.18 的由来 |

**同仓库对比：**

| 模块 | expand 默认值 |
|------|---------------|
| `render.py` / Production CLI | **0.18（18%）** |
| `timeline_overlay.py` | `DEFAULT_EXPAND_RATIO = **0.2**（20%）` |
| README（Timeline 渲染描述） | 「bbox 扩展 **20%**」 |

Production Render 用 **18%**，Timeline Render 用 **20%**，存在 **2 个百分点差异**，代码无说明。

**可推断意图（非代码明示）：**

- 马赛克略大于检测框，覆盖检测误差和脸部边缘  
- 18% 为经验默认，与 CLI `--expand` 暴露为可调参数  

---

## 6. 哪些参数会导致马赛克漂移？

### 6.1 `events_to_render()` 直接相关

| 参数 | 默认值 | 漂移影响 | 机制 |
|------|--------|----------|------|
| **`extend_frames`** | 3 | **边界漂移** | 首/末检测帧外 hold 固定 bbox；若人脸在边界外仍在移动，马赛克位置错误 |
| **（输入）trajectory 稀疏度** | interval=5 | **主要漂移** | 两检测点之间全靠线性插值；人脸非匀速运动时插值轨迹偏离真实位置 |
| **（输入）trajectory bbox 跳变** | 上游 identity stitch | **跳变漂移** | `b0→b1` 跨度大时，插值仍走直线，中间帧偏差更大 |
| **线性插值本身** | 固定算法 | **连续漂移** | 匀速假设 vs 实际加速/转向 |
| **`gap <= 1` 跳过规则** | — | 边缘情况 | 相邻检测帧无插值，马赛克仅出现在采样帧 |

`extend_frames` **不可通过 Production CLI 调整**（未暴露）。

### 6.2 `render_video()` 相关（绘制阶段）

| 参数 | 默认值 | 漂移影响 | 机制 |
|------|--------|----------|------|
| **`expand`** | 0.18 | **边缘视觉漂移** | 不改变 bbox 中心，但 mask 更大；脸快速移动时边缘可能露出或过度遮挡 |
| **`mosaic_block`** | 22 | 非位置漂移 | 影响马赛克颗粒大小，不影响 bbox 轨迹 |
| **`encoder` / `bitrate`** | auto / 12M | 无位置漂移 | 仅编码质量 |

`expand` **可通过** `--expand` CLI 调整（`pipeline.py` / `confirm.py`）。

### 6.3 上游输入（非 `events_to_render` 参数，但决定插值端点）

| 上游因素 | 影响 |
|----------|------|
| **`detect_interval=5`** | 决定 trajectory 密度 → 直接决定插值跨度 `gap-1`（通常 4 帧） |
| **YOLO 检测 jitter** | 改变 `b0`/`b1` 端点 → 插值整条线段偏移 |
| **Identity stitching 多 track 合并** | 相邻 trajectory 点跨 track 时 bbox 大跳 → 插值跨越大跳变 |

### 6.4 漂移类型 ↔ 参数对照

| 可见现象 | 主要责任参数/层 |
|----------|----------------|
| 两检测帧之间马赛克「滑动」但与脸不同步 | trajectory 稀疏度 + **线性插值**（Step B） |
| Event 开始/结束处马赛克多挡或少挡几帧 | **`extend_frames=3`**（Step C/D） |
| 马赛克突然跳到远处 | 上游 trajectory 跳变（非 events_to_render 参数） |
| 马赛克比脸大一圈、边缘闪动 | **`expand=0.18`**（render_video） |
| 同帧多个马赛克重叠 | 多 event 写入同一 `render[f]`（设计行为） |

---

## 7. 调用链确认

### `pipeline.py`（Auto 草稿）

```204:204:d:\work\制造业视频数据\new_da_ma\render.py
    render = events_to_render(events, meta["frames"])
```

仅传 2 个参数 → `extend_frames` **恒为 3**。

### `confirm.py`（final.mp4）

```112:112:d:\work\制造业视频数据\new_da_ma\confirm.py
        render = events_to_render(all_mask_events, meta["frames"])
```

同样 → `extend_frames` **恒为 3**。

---

## 8. 总结表

| 问题 | 答案 |
|------|------|
| 何时插值？ | 相邻 trajectory 点 `gap > 1` 时，对 `f0+1 … f1-1` 插值 |
| 插值跨度？ | 每对检测点之间 `gap-1` 帧；Production 通常 gap=5 → **4 帧/段** |
| 为何线性？ | 代码无说明；实现为 `b0+(b1-b0)*t`；同仓库 Timeline 链亦用线性 |
| 为何 padding 3 帧？ | 硬编码 `extend_frames=3`；代码无说明；≈0.1s@30fps |
| 为何 expand 18%？ | 硬编码 + CLI 默认；代码无说明；Timeline 链用 20% 略不一致 |
| 漂移相关参数 | trajectory 稀疏度、`extend_frames`、线性插值算法、`expand`、上游 bbox 跳变 |

---

*本文档为只读函数分析，不包含修复建议或代码变更。*
