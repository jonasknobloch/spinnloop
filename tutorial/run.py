#!/usr/bin/env python3

import pytimeloop.timeloopfe.v4 as tl

def run():
    spec = tl.Specification.from_yaml_files(
        "architecture12.yaml",
        "problem.yaml",
        # "mapping.yaml",
        "mapper.yaml",
        "variables.yaml",
        "intmac.yaml",
    )

    # tl.call_model(spec, output_dir="out")
    tl.call_mapper(spec, output_dir="out_mapping")

run()


