# ModelZoo

ModelZoo contains model export, inference, and benchmarking examples.

## Models

| Model | Task | Description |
| --- | --- | --- |
| [Vehicle Classifier](Vehicle_Classifier/) | Image classification | Automotive color, direction, and vehicle-type recognition using a bilinear ResNet-18 model. |
| [UNet](UNet/) | Semantic segmentation | UNet export instructions using MMSegmentation and a Cityscapes checkpoint. |
| [YOLOv10](YOLOv10/) | Object detection | ONNX-based YOLOv10 inference and performance benchmarking with TVM Relax on CUDA. |

Each model directory provides its own setup and usage instructions. Exported
ONNX model files are excluded from Git and should be downloaded or generated
locally.
