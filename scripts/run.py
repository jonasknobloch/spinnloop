#!/usr/bin/env python3
import os

import pytimeloop.timeloopfe.v4 as tl

# os.environ["TIMELOOP_ENABLE_TRACING"] = "1"
# os.environ["TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION"] = "1"
# os.environ["TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION"] = "1"

def run(layer, processing_elements, dimensions):
    spec = tl.Specification.from_yaml_files(
        "../config/intmac.yaml",
        "../config/architecture.yaml",
        "../config/problem.yaml",
        "../config/variables.yaml",
        f"../config/tilings/{layer}.yaml"
    )

    spec.architecture.find("PE").spatial.meshX = processing_elements

    spec.problem.instance['M'] = dimensions[0]
    spec.problem.instance['N'] = dimensions[1]
    spec.problem.instance['K'] = dimensions[2]

    stats = tl.call_model(spec, output_dir=f"../out/{layer}")

    return stats.cycles

run("qkv_with_linear", 96, (512, 768, 768))
run("mlp_linear_1", 128, (512, 3072, 768))
run("mlp_linear_2", 128, (512, 768, 3072))
run("bmm1", 8, (512, 512, 64))
run("bmm2", 8, (512, 64, 512))
