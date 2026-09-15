<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# YOLOv10

This directory contains a TVM Relax CUDA inference and performance benchmark
for an ONNX-exported YOLOv10 model.

Export a model to ONNX, then run:

```bash
export TVM_HOME=/path/to/tvm
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/.local/python"
python3 YOLOv10/inference.py /path/to/yolov10n.onnx
```

For a dynamic ONNX model, set the spatial input size with `--input-size`. The
benchmark duration is controlled by `--warmup`, `--number`, and `--repeat`:

```bash
python3 YOLOv10/inference.py /path/to/yolov10n.onnx \
    --input-size 640 --warmup 10 --number 10 --repeat 5
```
