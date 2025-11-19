import os
import shutil

import typer

from collections import defaultdict

import pytimeloop.timeloopfe.v4 as tl

from .trace import _trace
from .tilings import _tilings, Tiling

_use_cached_timeloop_results = False
_apply_bandwidth_limits = True

def model(cache: bool = True, bandwidth_limits: bool = True):
    global _use_cached_timeloop_results
    _use_cached_timeloop_results = cache

    global _apply_bandwidth_limits
    _apply_bandwidth_limits = bandwidth_limits

    _model()

def _model():
    trace = _trace("facebook/opt-125m", 128)
    tilings = _tilings()

    cycles_per_layer = defaultdict(int)
    latency_per_layer = defaultdict(int)

    for dimensions, value in trace.matmuls_grouped.items():
        # print(dimensions, value)

        tiling_key = _select_tiling_key(tilings, dimensions)

        if tiling_key is None:
            raise RuntimeError(f"no tiling for dimensions {dimensions}")

        processing_elements = tilings[tiling_key].processing_elements[0]

        cycles = _run(tiling_key, processing_elements, dimensions)

        for i in range(len(value) // 12):
            # print(tiling_key, value[i])

            cycles_per_layer[(tiling_key, value[i])] += cycles
            latency_per_layer[(tiling_key, value[i])] += _cycles_to_latency_us(cycles)

    latency_layer_norm_ms = 0.74
    latency_softmax_ms = 0.28
    latency_add_ms = 0.02

    processing_elements_layer_norm = 128 # TODO verify
    processing_elements_softmax = 128 # TODO verify
    processing_elements_add = 128 # TODO verify

    for (event, shapes), value in trace.static_grouped.items():
        if event == "aten::add":
            if [len(s) for s in shapes] == [3, 3, 0]:
                latency_per_layer[("layer_add", value[0])] = (shapes[0][1] * shapes[0][2]) * latency_add_ms / processing_elements_add
                continue
            if [len(s) for s in shapes] == [2, 2, 0]:
                latency_per_layer[("layer_add", value[0])] = (shapes[0][0] * shapes[0][1]) * latency_add_ms / processing_elements_add
                continue
        if event == "aten::layer_norm":
            if [len(s) for s in shapes] == [3, 0, 1, 1, 0, 0]:
                latency_per_layer[("layer_norm", value[0])] = (shapes[0][1]) * latency_layer_norm_ms / processing_elements_layer_norm
                continue
            if [len(s) for s in shapes] == [2, 0, 1, 1, 0, 0]:
                latency_per_layer[("layer_norm", value[0])] = (shapes[0][0] * shapes[0][1]) * latency_layer_norm_ms / processing_elements_layer_norm
                continue
        if event == "aten::softmax":
            latency_per_layer[("softmax", value[0])] = (shapes[0][1] * shapes[0][2] * shapes[0][3]) * latency_softmax_ms / processing_elements_softmax # 12 times per layer
            continue

        raise RuntimeError(f"unhandled static event {event}")

    for (key, value) in sorted(latency_per_layer.items(), key=lambda x: x[0][1]):
        print(f"{key[0]},{value:.4f}")

def _run(layer, processing_elements, dimensions):
    # layer = "qkv_with_linear"
    # processing_elements = 96
    # dimensions = (512, 768, 768)

    spec = tl.Specification.from_yaml_files(
        "config/intmac.yaml",
        "config/architecture.yaml",
        "config/problem.yaml",
        "config/variables.yaml",
        f"config/mappings/128/{layer}.yaml"
    )

    spec.architecture.find("PE").spatial.meshX = processing_elements

    if not _apply_bandwidth_limits:
        spec.architecture.find("DRAM").attributes.shared_bandwidth = None
        spec.architecture.find("DRAM").attributes.read_bandwidth = None
        spec.architecture.find("DRAM").attributes.write_bandwidth = None
        spec.architecture.find("Buffer").attributes.shared_bandwidth = None
        spec.architecture.find("Buffer").attributes.read_bandwidth = None
        spec.architecture.find("Buffer").attributes.write_bandwidth = None


    spec.problem.instance['M'] = dimensions[0]
    # spec.problem.instance['N'] = dimensions[1]
    spec.problem.instance['N'] = dimensions[2]
    # spec.problem.instance['K'] = dimensions[2] # m x n x shared
    spec.problem.instance['K'] = dimensions[1] # m x shared x n

    out_path = f"out/{layer}"

    if os.path.exists(out_path) and _use_cached_timeloop_results:
        stats = tl.parse_timeloop_output(spec=spec, output_dir=out_path, prefix="timeloop-model")
    else:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        stats = tl.call_model(spec, output_dir=out_path)

    return stats.cycles

def _select_tiling_key(tilings, dimensions):
    for key, tiling in tilings.items():
        assert type(tiling) is Tiling

        # if (tiling.op_a[0], tiling.op_b[1], tiling.op_a[1]) == dimensions: # m x n x shared
        if (tiling.op_a[0], tiling.op_a[1], tiling.op_b[1]) == dimensions: # m x shared x n
            return key

    return None

def _cycles_to_latency_ms(cycles, clock_speed=150e6):
    return  (cycles / clock_speed) * 1e3

def _cycles_to_latency_us(cycles, clock_speed=150e6):
    return  (cycles / clock_speed) * 1e6