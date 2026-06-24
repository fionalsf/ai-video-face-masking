# 模型权重

将 YOLO-face 权重放在此目录，命名为 `face.pt`。

推荐 [akanametov/yolo-face](https://github.com/akanametov/yolo-face/releases) 的 YOLOv11-face（PowerShell）：

```powershell
Invoke-WebRequest -Uri "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt" -OutFile models/face.pt
```

权重文件不入库，需手动下载。

## ArcFace（Identity Stitching，可选）

Cross-track identity stitching 优先使用 ArcFace ONNX embedding。将模型放在：

`models/arcface_r50.onnx`

未提供时自动回退到 HSV histogram（精度较低）。可选依赖：

```powershell
pip install onnxruntime
```
