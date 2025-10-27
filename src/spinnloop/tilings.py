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
            out_num):

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

    def validate(self, buffer_capacity, processing_elements, parts):
        return self._shared_dimension(), self._memory_available(buffer_capacity), self._tiles_match_processing_elements(processing_elements, parts)

    def _shared_dimension(self):
        return self.op_a[1] == self.op_b[0]

    def _memory_available(self, buffer_capacity):
        return ((self.op_a_size[0] * self.op_a_size[1]) + (self.op_b_size[0] * self.op_b_size[1])) <= buffer_capacity

    def _tiles_match_processing_elements(self, processing_elements, parts):
        return self.out_num[0] * self.out_num[1] == processing_elements * parts

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

def tilings(
        # trace: Annotated[str, typer.Argument(help="")] = "",
):
    # TODO loop over trace

    layers = dict()

    layers["qkv_with_linear"] = Tiling(
        (512, 768),
        (768, 768),
        (512, 768),
        (64, 768),
        (768, 64),
        (64, 64),
        (8, 1),
        (1, 12),
        (8, 12),
    )

    layers["mlp_linear_1"] = Tiling(
        (512, 768),
        (768, 3072),
        (512, 3072),
        (64, 768),
        (768, 64),
        (64, 64),
        (8, 1),
        (1, 48),
        (8, 48),
    )

    layers["mlp_linear_2"] = Tiling(
        (512, 3072),
        (3072, 768),
        (512, 768),
        (16, 3072),
        (3072, 16),
        (16, 16),
        (32, 1), # was 12
        (1, 48),
        (32, 48), # was 12
    )

    layers["bmm1"] = Tiling(
        (512, 64),
        (64, 512),
        (512, 512),
        (64, 64),
        (64, 512),
        (64, 512),
        (8, 1),
        (1, 1),
        (8, 1),
    )

    layers["bmm2"] = Tiling(
        (512, 512),
        (512, 64),
        (512, 64),
        (64, 512),
        (512, 64),
        (64, 64),
        (8, 1),
        (1, 1),
        (8, 1),
    )

    print(layers["qkv_with_linear"].validate(96*1024, 96, 1))
    print(layers["mlp_linear_1"].validate(96*1024, 128, 3))
    print(layers["mlp_linear_2"].validate(96*1024, 128, 12))
    print(layers["bmm1"].validate(96*1024, 8, 1)) # 12 times
    print(layers["bmm2"].validate(96*1024, 8, 1)) # 12 times

    print(layers["qkv_with_linear"].loop_bounds(96, 1))
    print(layers["mlp_linear_1"].loop_bounds(128, 3))
    print(layers["mlp_linear_2"].loop_bounds(128, 12))
    print(layers["bmm1"].loop_bounds(8, 1)) # 12 times
    print(layers["bmm2"].loop_bounds(8, 1)) # 12 times

