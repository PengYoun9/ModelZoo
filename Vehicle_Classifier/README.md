# Vehicle_classifier

## DownLoad Weights 
Download the weights from: https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification

## Run Inference
```shell
python3 inference.py -w vehicle_classifier.pth -i test_images/test_car0.png
```

## Get Onnx Model
```shell
python3 inference.py -w vehicle_classifier.pth -e vehicle_classifier.onnx
```

## Optimize Onnx Model
```bash
onnxsim vehicle_classifier.onnx vehicle_classifier_opt.onnx
```


## Reference
https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification