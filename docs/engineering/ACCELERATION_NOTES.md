# V4 Acceleration Notes

## Current status

- Stable production backend remains `torch`.
- ONNX export works: `models/face.pt` -> `models/face.onnx`.
- ONNX Runtime CUDA can be activated with `onnxruntime-gpu==1.20.2` and the CUDA/cuDNN DLLs bundled with PyTorch.
- TensorRT Python runtime is installed with `tensorrt-cu12==10.13.3.9`.
- `trtexec` is still not available, but ONNX Runtime can use `TensorrtExecutionProvider` through the pip TensorRT libraries.

## Why ONNX is not the default

Benchmarks on `DJI_20260509094235_0002_D.MP4`, `imgsz=1280`, `interval=2`:

- `torch`, batch 4, 40 sparse frames: about 5.2s.
- `onnx`, batch 4, 40 sparse frames: about 7.5s.
- `onnx`, batch 1, 40 sparse frames: about 7.9s.
- `tensorrt` FP32, batch 4, 40 sparse frames, `conf=0.23`: about 3.4s after engine cache warmup.
- `tensorrt` FP16, batch 4, 40 sparse frames, `conf=0.23`: about 2.5s after engine cache warmup.

The ONNX backend is real CUDA, not CPU fallback, but it is slower than the current PyTorch backend on this GPU/runtime combination. It also produced slightly fewer boxes in the small benchmark, so it should stay explicit-test only until detection parity is audited.

TensorRT is faster after its engine cache has been built, but first engine construction is slow:

- FP32 first build: about 4 minutes on the 8-frame benchmark.
- FP16 first build: several minutes.
- A partial full-video FP32 test showed about 11 sparse FPS, which was not enough to justify running the whole 37-minute review-only job in that configuration.
- A full-video FP16 review-only run with `--infer-backend tensorrt --conf 0.23 --detect-batch 4` was stopped after it stalled:
  - Sparse detection completed in about 58m24s (`predict=3266s`), far slower than the torch baseline.
  - It produced 11,050 raw boxes across 8,435 sparse frames.
  - After detection, OpenCV emitted repeated 6 MB allocation failures while reading frames.
  - The process reached behavior/identity output files but never produced `review_report.json`.
- `conf=0.23` gave similar small-sample recall to the torch `conf=0.25` baseline; this needs real review-quality validation before production use.

## Runtime guardrails

- `--infer-backend torch` is the production path.
- `--infer-backend auto` intentionally keeps the torch backend.
- `--infer-backend onnx` must be selected explicitly.
- `--infer-backend tensorrt` must be selected explicitly.
- TensorRT uses FP32 by default. Set `FACE_MASK_TRT_FP16=True` to use FP16.
- ONNX/TensorRT sessions verify active providers and call `disable_fallback()` when available, so CUDA failures do not silently turn into slow CPU runs.

## Useful commands

Export ONNX:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python export_onnx_model.py --model models/face.pt --output models/face.onnx --imgsz 1280 --opset 12 --dynamic
```

Benchmark backend:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python benchmark_infer_backend.py -i "D:\path\to\input.MP4" --backend torch --samples 40 --interval 2 --batch 4 --imgsz 1280 --conf 0.25
python benchmark_infer_backend.py -i "D:\path\to\input.MP4" --backend onnx --samples 40 --interval 2 --batch 4 --imgsz 1280 --conf 0.25
$env:FACE_MASK_TRT_FP16='True'
python benchmark_infer_backend.py -i "D:\path\to\input.MP4" --backend tensorrt --samples 40 --interval 2 --batch 4 --imgsz 1280 --conf 0.23
```

## TensorRT next step

TensorRT runtime libraries install successfully after network access to `https://pypi.nvidia.com/` is available:

```powershell
python -m pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-cu12==10.13.3.9
```

The remaining work is not installation; it is quality/performance validation on real review-only runs.
