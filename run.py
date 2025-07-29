import os

import pytimeloop.timeloopfe.v4 as tl

def run():
    spec = tl.Specification.from_yaml_files(
        "spinnaker2/arch.yaml",
        "spinnaker2/components.yaml",
        "spinnaker2/problem.yaml",
        "spinnaker2/mapper.yaml",
        "spinnaker2/variables.yaml",
    )

    tl.call_mapper(spec, output_dir="out")

ARCH_PATH = f"{os.curdir}/transformer_gemmini/arch/system_gemmini.yaml"
COMPONENTS_PATH = f"{os.curdir}/transformer_gemmini/arch/components/*.yaml"
PROBLEM_PATH = f"{os.curdir}/transformer_gemmini/layers/FF1_layer.yaml"
MAPPER_PATH = f"{os.curdir}/transformer_gemmini/mapper/mapper.yaml"
CONSTRAINTS_PATH = f"{os.curdir}/transformer_gemmini/constraints/constraints.yaml"
VARIABLES_PATH = f"{os.curdir}/transformer_gemmini/mapper/variables.yaml"

def run_mapper():
    # spec = tl.Specification.from_yaml_files("transformer_gemmini/arch/system_gemmini.yaml", "transformer_gemmini/layers/FF1_layer.yaml")
    spec = tl.Specification.from_yaml_files(
        ARCH_PATH,
        COMPONENTS_PATH,
        MAPPER_PATH,
        PROBLEM_PATH,
        CONSTRAINTS_PATH,
        VARIABLES_PATH
    )
    spec.mapspace.template = 'uber' #'ruby'
    tl.call_mapper(spec, output_dir="out2")

run()