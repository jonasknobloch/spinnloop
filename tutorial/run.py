#!/usr/bin/env python3
import os

import pytimeloop.timeloopfe.v4 as tl

os.environ["TIMELOOP_ENABLE_TRACING"] = "1"
# os.environ["TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION"] = "1"
# os.environ["TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION"] = "1"


def run():
    spec = tl.Specification.from_yaml_files(
        "architecture12.yaml",
        "problem.yaml",
        "mapping3.yaml",
        # "mapper.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    baz = spec.architecture.nodes[1].spatial.meshX
    l = spec.problem.instance['L']
    d = spec.problem.instance['D']
    e = spec.problem.instance['E']

    # spec.architecture.find()

    # spec.mapspace.template = 'uber'

    tl.call_model(spec, output_dir="out")
    # tl.call_mapper(spec, output_dir="out_mapping")


def qkv_with_linear():
    spec = tl.Specification.from_yaml_files(
        "arch/96.yaml",
        "shapes/gemm/qkv_with_linear.yaml",
        "tilings/qkv_with_linear.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    spec.architecture.find("PE").spatial.meshX = 96

    tl.call_model(spec, output_dir="out/qkv_with_linear")

def mlp_linear_1():
    spec = tl.Specification.from_yaml_files(
        "arch/96.yaml",
        "shapes/gemm/mlp_linear_1.yaml",
        "tilings/mlp_linear_1.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    spec.architecture.find("PE").spatial.meshX = 128

    tl.call_model(spec, output_dir="out/mlp_linear_1")

def mlp_linear_2():
    spec = tl.Specification.from_yaml_files(
        "arch/96.yaml",
        "shapes/gemm/mlp_linear_2.yaml",
        "tilings/mlp_linear_2.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    spec.architecture.find("PE").spatial.meshX = 128

    tl.call_model(spec, output_dir="out/mlp_linear_2")

def bmm1():
    spec = tl.Specification.from_yaml_files(
        "arch/96.yaml",
        "shapes/gemm/bmm1.yaml",
        "tilings/bmm1.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    spec.architecture.find("PE").spatial.meshX = 8

    tl.call_model(spec, output_dir="out/bmm1")

def bmm2():
    spec = tl.Specification.from_yaml_files(
        "arch/96.yaml",
        "shapes/gemm/bmm2.yaml",
        "tilings/bmm2.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    spec.architecture.find("PE").spatial.meshX = 8

    tl.call_model(spec, output_dir="out/bmm2")

def foo():
    # (512, 768, 768) 48  # qkv_with_linear
    # (512, 64, 512) 144  # bmm1
    # (512, 512, 64) 144  # bmm2
    # (512, 768, 3072) 12 # mlp_linear_1
    # (512, 3072, 768) 12 # mlp_linear_2

    executions_qkv_with_linear = 48
    executions_mlp_linear_1 = 144
    executions_mlp_linear_2 = 144
    executions_bmm1 = 12
    executions_bmm2 = 12

    # cycles_qkv_with_linear = 312606720
    # cycles_mlp_linear_1 = 1248854016
    # cycles_mlp_linear_2 = 1235877888
    # cycles_bmm1 = 18087936
    # cycles_bmm2 = 18317312

    cycles_qkv_with_linear = 245760
    cycles_mlp_linear_1 = 737280
    cycles_mlp_linear_2 = 737280
    cycles_bmm1 = 163840
    cycles_bmm2 = 163840

    # cycles_qkv_with_linear = 49152
    # cycles_mlp_linear_1 = 147456
    # cycles_mlp_linear_2 = 147456
    # cycles_bmm1 = 32768
    # cycles_bmm2 = 32768

    # cycles / clock_freq
    # 10^6 for microseconds
    # 10^3 for milliseconds

    total_cycles = (
            (cycles_qkv_with_linear * executions_qkv_with_linear)
            + (cycles_mlp_linear_1 * executions_mlp_linear_1)
            + (cycles_mlp_linear_2 * executions_mlp_linear_2)
            + (cycles_bmm1 * executions_bmm1)
            + (cycles_bmm2 * executions_bmm2)
    )

    clock_hz = 150e6  # 150 MHz

    total_us = (total_cycles / clock_hz) * 1e6
    total_ms = (total_cycles / clock_hz) * 1e3

    print(f"Total cycles: {total_cycles:,}")
    print(f"Total time: {total_us:.2f} µs")
    print(f"Total time: {total_ms:.2f} ms")

def bar():
    # ('aten::layer_norm', ((1, 512, 768), (), (768,), (768,), (), ())) 12
    # ('aten::softmax', ((1, 12, 512, 512), (), ())) 12
    # ('aten::add', ((1, 512, 768), (1, 512, 768), ())) 12
    # ('aten::layer_norm', ((512, 768), (), (768,), (768,), (), ())) 12
    # ('aten::add', ((512, 768), (512, 768), ())) 12

    latency_layer_norm_us = 0.74
    latency_softmax_us = 0.28
    latency_add_us = 0.02

    total_us = (
        (latency_layer_norm_us * 24 * 512 * 768) # AxB
        + (latency_softmax_us * 12 * 512 * 512)
        + (latency_add_us * 24 * 512 * 768)
    )

    total_ms = total_us * 1e-3

    print(f"Total static: {total_us:.2f} µs")
    print(f"Total static: {total_ms:.2f} ms")


qkv_with_linear()
# mlp_linear_1()
# mlp_linear_2()
# bmm1()
# bmm2()

# foo()
# bar()

# run()