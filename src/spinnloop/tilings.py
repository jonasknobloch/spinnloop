import os
import csv
import re
import numpy as np

class Tiling:
    def __init__(
            self,
            op_a,
            op_b,
            out,
            op_a_size,
            op_b_size,
            out_size,
            op_a_num,
            op_b_num,
            out_num,
            processing_elements):

        for k, v in locals().items():
            if v == self:
                continue

            if type(v) == tuple and len(v) == 2:
                continue

            raise TypeError(f"{k} must be a tuple")

        self.op_a = op_a
        self.op_b = op_b
        self.out = out
        self.op_a_size = op_a_size
        self.op_b_size = op_b_size
        self.out_size = out_size
        self.op_a_num = op_a_num
        self.op_b_num = op_b_num
        self.out_num = out_num
        self.processing_elements = processing_elements # (processing_elements, parts)

    def validate(self, buffer_capacity):
        return self._shared_dimension(), self._memory_available(buffer_capacity), self._tiles_match_processing_elements()

    def _shared_dimension(self):
        return self.op_a[1] == self.op_b[0]

    def _memory_available(self, buffer_capacity):
        return ((self.op_a_size[0] * self.op_a_size[1]) + (self.op_b_size[0] * self.op_b_size[1])) <= buffer_capacity

    def _tiles_match_processing_elements(self):
        return self.out_num[0] * self.out_num[1] == self.processing_elements[0] * self.processing_elements[1]

    def loop_bounds(self, processing_elements, parts):
        mal_x = 16
        mal_y = 4

        m = self.op_a[0]
        n = self.op_a[1]
        k = self.op_b[1]

        bounds = np.array([
            [1, 1, parts],
            [self.out_num[0], 1, int(self.out_num[1]/parts)],
            [int(self.out_size[0]/mal_y), n, int(self.out_size[1]/mal_x)],
            [mal_y, 1, mal_x]
        ])

        prod_cols = np.prod(bounds, axis=0).tolist()
        prod_rows = np.prod(bounds, axis=1).tolist()

        if prod_cols != [m, n, k]:
            raise RuntimeError(f"invalid loop bounds: expected {[m, n, k]} but got {prod_cols}")

        if prod_rows[1] != processing_elements:
            raise RuntimeError(f"unexpected spatial mapping: expected {processing_elements} but got {prod_rows[1]}")

        return bounds.tolist()

def tilings():
    layers = _tilings()

    print(layers["q_proj"].validate(96*1024))
    print(layers["k_proj"].validate(96*1024))
    print(layers["v_proj"].validate(96*1024))
    print(layers["qk_bmm"].validate(96*1024)) # TODO still 12 times?
    print(layers["pv_bmm"].validate(96*1024)) # TODO still 12 times?
    print(layers["fc1"].validate(96*1024))
    print(layers["fc2"].validate(96*1024))

    _generate_mappings(layers, "config/mappings/128")

def _tilings():
    layers = _from_csv("data/2025-11-18__opt125m__tiling_prefill/tiling_prefill_token=128.csv")

    return layers

def _from_csv(file_path):
    layers = dict()

    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if not row['layer_type'].startswith('opt_'):
                continue

            try:
                def get_dims(key_v, key_h):
                    return int(row[key_v]), int(row[key_h])

                op_a_size = get_dims('input_tile_size_vertical', 'input_tile_size_horizontal')
                op_b_size = get_dims('weights_tile_size_vertical', 'weights_tile_size_horizontal')
                out_size  = get_dims('output_tile_size_vertical', 'output_tile_size_horizontal')

                op_a_num = get_dims('input_tile_num_vertical', 'input_tile_num_horizontal')
                op_b_num = get_dims('weights_tile_num_vertical', 'weights_tile_num_horizontal')
                out_num  = get_dims('output_tile_num_vertical', 'output_tile_num_horizontal')

                # if row['layer_type'] == 'opt_bmm':
                #   out_num = (out_num[0] * 12, out_num[1]) # TODO fix in csv

                op_a = (op_a_size[0] * op_a_num[0], op_a_size[1] * op_a_num[1])
                op_b = (op_b_size[0] * op_b_num[0], op_b_size[1] * op_b_num[1])
                out  = (out_size[0] * out_num[0],   out_size[1] * out_num[1])

                pes = (int(row['num_worker_pes_per_part']), int(row['num_parts']))

                # layer_name = row['layer_name'].replace('decoder_layer_0_', '')
                layer_name = re.sub(r'^decoder_layer_\d+_', '', row['layer_name'])

                layers[layer_name] = Tiling(
                    op_a,
                    op_b,
                    out,
                    op_a_size,
                    op_b_size,
                    out_size,
                    op_a_num,
                    op_b_num,
                    out_num,
                    pes
                )
            except ValueError as e:
                print(f"skipping row {row.get('layer_name', 'unknown')}: {e}")

    return layers

def _generate_mappings(layers, output_dir="mappings"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mal_x = 16
    mal_y = 4

    # M (rows), K (shared), N (columns)

    for name, tiling in layers.items():
        parts = tiling.processing_elements[1]

        factors_accelerator = {
            "M": mal_y,
            "K": 1,
            "N": mal_x
        }

        factors_sram = {
            "M": tiling.out_size[0] // mal_y,
            "K": tiling.op_a_size[1],
            "N": tiling.out_size[1] // mal_x
        }

        factors_pe = {
            "M": tiling.out_num[0],
            "K": 1,
            "N": tiling.out_num[1] // parts
        }

        factors_dram = {
            "M": 1,
            "K": 1,
            "N": parts
        }

        def fmt(f):
            return f"[N={f['N']}, M={f['M']}, K={f['K']}]"

        yaml_content = f"""mapping:
  - factors: {fmt(factors_dram)}
    permutation: [K, M, N]
    target: DRAM
    type: temporal
  - factors: {fmt(factors_pe)}
    permutation: [K, M, N]
    target: PE
    type: spatial
  - factors: {fmt(factors_sram)}
    permutation: [K, M, N]
    target: Buffer
    type: temporal
  - factors: {fmt(factors_accelerator)}
    permutation: [N, M, K]
    target: Accelerator
    type: spatial
    split: 1
"""

        filename = f"{output_dir}/{name}.yaml"

        with open(filename, "w") as f:
            f.write(yaml_content)

        print(f"generated: {filename}")
