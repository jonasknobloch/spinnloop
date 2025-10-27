import os
import typer

from collections import defaultdict

import pytimeloop.timeloopfe.v4 as tl

from .trace import _trace
from .tilings import _tilings, Tiling


def model():
    _model()

def _model():
    trace = _trace("facebook/opt-125m", 512)
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
            latency_per_layer[(tiling_key, value[i])] += _cycles_to_latency_ms(cycles)

    latency_layer_norm_ms = 0.74 * 1e-3
    latency_softmax_ms = 0.28 * 1e-3
    latency_add_ms = 0.02 * 1e-3

    for (event, shapes), value in trace.static_grouped.items():
        if event == "aten::add":
            if [len(s) for s in shapes] == [3, 3, 0]:
                # latency_per_layer[("layer_add", value[0])] = (shapes[0][1] * shapes[0][2]) * latency_add_ms
                latency_per_layer[("layer_add", value[0])] = (shapes[0][1]) * latency_add_ms # TODO per embedding vector or per element
                continue
            if [len(s) for s in shapes] == [2, 2, 0]:
                # latency_per_layer[("layer_add", value[0])] = (shapes[0][0] * shapes[0][1]) * latency_add_ms
                latency_per_layer[("layer_add", value[0])] = (shapes[0][0]) * latency_add_ms
                continue
        if event == "aten::layer_norm":
            if [len(s) for s in shapes] == [3, 0, 1, 1, 0, 0]:
                # latency_per_layer[("layer_norm", value[0])] = (shapes[0][1] * shapes[0][2]) * latency_layer_norm_ms
                latency_per_layer[("layer_norm", value[0])] = (shapes[0][1]) * latency_layer_norm_ms
                continue
            if [len(s) for s in shapes] == [2, 0, 1, 1, 0, 0]:
                # latency_per_layer[("layer_norm", value[0])] = (shapes[0][0] * shapes[0][1]) * latency_layer_norm_ms
                latency_per_layer[("layer_norm", value[0])] = (shapes[0][0]) * latency_layer_norm_ms
                continue
        if event == "aten::softmax":
            # latency_per_layer[("softmax", value[0])] = (shapes[0][1] * shapes[0][2] * shapes[0][3]) * latency_softmax_ms # 12 times per layer
            latency_per_layer[("softmax", value[0])] = (shapes[0][1] * shapes[0][2]) * latency_softmax_ms # 12 times per layer
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
        f"config/tilings/{layer}.yaml"
    )

    spec.architecture.find("PE").spatial.meshX = processing_elements

    spec.problem.instance['M'] = dimensions[0]
    # spec.problem.instance['N'] = dimensions[1]
    spec.problem.instance['N'] = dimensions[2]
    # spec.problem.instance['K'] = dimensions[2] # m x n x shared
    spec.problem.instance['K'] = dimensions[1] # m x shared x n

    if os.path.exists(f"out/{layer}"):
        stats = tl.parse_timeloop_output(spec=spec, output_dir=f"out/{layer}", prefix="timeloop-model")
    else:
        stats = tl.call_model(spec, output_dir=f"out/{layer}")

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