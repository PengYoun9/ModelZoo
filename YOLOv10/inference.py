# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Compile and benchmark an ONNX-exported YOLOv10 model with TVM CUDA."""

import argparse
from pathlib import Path
import re

import numpy as np
import onnx

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the YOLOv10 ONNX model")
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="height and width used for dynamic model dimensions (default: 640)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup runs")
    parser.add_argument(
        "--number", type=int, default=10, help="inferences averaged in each measurement"
    )
    parser.add_argument("--repeat", type=int, default=5, help="number of measurements")
    return parser.parse_args()


def resolve_shape(shape: list[int | str | None], input_size: int) -> tuple[int, ...]:
    """Resolve the conventional dynamic NCHW dimensions used by YOLO exports."""
    if len(shape) != 4:
        raise ValueError(f"expected a rank-4 NCHW input, but got {shape}")
    defaults = (1, 3, input_size, input_size)
    return tuple(
        dim if isinstance(dim, int) and dim > 0 else defaults[i] for i, dim in enumerate(shape)
    )


def flatten_outputs(value) -> list[np.ndarray]:
    if isinstance(value, (tuple, list)):
        return [item for element in value for item in flatten_outputs(element)]
    if hasattr(value, "numpy"):
        return [value.numpy()]
    return [np.asarray(value)]


def sanitize_tensor_names(model: onnx.ModelProto) -> None:
    """Replace ONNX tensor names that cannot be emitted as CUDA identifiers."""
    names = set()
    names.update(value.name for value in model.graph.input)
    names.update(value.name for value in model.graph.output)
    names.update(value.name for value in model.graph.value_info)
    names.update(value.name for value in model.graph.initializer)
    for node in model.graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)

    rename = {}
    used = {name for name in names if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)}
    for name in sorted(names):
        if name in used:
            continue
        candidate = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if not candidate or candidate[0].isdigit():
            candidate = f"tensor_{candidate}"
        base = candidate
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        rename[name] = candidate
        used.add(candidate)

    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value.name = rename.get(value.name, value.name)
    for value in model.graph.initializer:
        value.name = rename.get(value.name, value.name)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = rename.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = rename.get(name, name)


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"model does not exist: {args.model}")
    if args.input_size <= 0 or args.warmup < 0 or args.number <= 0 or args.repeat <= 0:
        raise ValueError(
            "input-size, number, and repeat must be positive; warmup cannot be negative"
        )

    device = tvm.cuda(0)
    if not device.exist:
        raise RuntimeError("CUDA device 0 is unavailable or TVM was built without CUDA support")
    target = tvm.target.Target.from_device(device)

    model = onnx.load(args.model, load_external_data=True)
    onnx.checker.check_model(model)
    sanitize_tensor_names(model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in initializer_names]
    if len(graph_inputs) != 1:
        raise ValueError(f"expected one YOLO image input, found {len(graph_inputs)}")

    input_info = graph_inputs[0]
    input_shape = resolve_shape(
        [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in input_info.type.tensor_type.shape.dim
        ],
        args.input_size,
    )
    input_name = input_info.name
    rng = np.random.default_rng(0)
    input_data = rng.random(input_shape, dtype=np.float32)

    print(f"Model: {args.model}")
    print(f"Input: {input_name}, shape={input_shape}, dtype=float32")
    print(f"Target: {target}")

    mod = from_onnx(
        model,
        shape_dict={input_name: input_shape},
        keep_params_in_input=False,
    )
    mod = relax.transform.DecomposeOpsForInference()(mod)

    with tvm.transform.PassContext(opt_level=3):
        executable = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(executable, device)

    gpu_input = tvm.runtime.tensor(input_data, device)
    for _ in range(args.warmup):
        vm["main"](gpu_input)
    device.sync()

    outputs = flatten_outputs(vm["main"](gpu_input))
    for index, output in enumerate(outputs):
        print(f"Output[{index}]: shape={output.shape}, dtype={output.dtype}")

    vm.set_input("main", gpu_input)
    benchmark = vm.time_evaluator(
        "invoke_stateful",
        device,
        number=args.number,
        repeat=args.repeat,
    )("main")
    latency_ms = np.asarray(benchmark.results) * 1e3
    mean_ms = float(np.mean(latency_ms))
    print("Performance:")
    print(f"  samples (ms): {np.array2string(latency_ms, precision=3)}")
    print(f"  mean (ms):    {mean_ms:.3f}")
    print(f"  median (ms):  {np.median(latency_ms):.3f}")
    print(f"  min (ms):     {np.min(latency_ms):.3f}")
    print(f"  max (ms):     {np.max(latency_ms):.3f}")
    print(f"  throughput:   {1000.0 / mean_ms:.2f} images/s")


if __name__ == "__main__":
    main()
