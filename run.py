import pytimeloop.timeloopfe.v4 as tl

def run():
    spec = tl.Specification.from_yaml_files("inputs/arch.yaml", "inputs/map.yaml", "inputs/prob.yaml")
    tl.call_model(spec, output_dir="out")

run()