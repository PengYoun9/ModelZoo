# UNet

## DownLoad Weights
Download the weights from: https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth

## Export Model to Onnx
```shell
pip3 install mmsegmentation==0.30.0

git clone -b v0.30.0 https://github.com/open-mmlab/mmsegmentation.git

cd mmsegmentation

python3 tools/pytorch2onnx.py configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py --checkpoint ../fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --output-file unet.onnx --opset-version 11
```


## Reference
https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet