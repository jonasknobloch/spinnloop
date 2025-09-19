import torch
from enum import Enum

# from torch.profiler import profile, ProfilerActivity
# from transformers import OPTModel
#
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)


class Key(str, Enum):
    ADDMM = "aten::addmm"
    BMM = "aten::bmm"
    LINEAR = "aten::linear"
    MATMUL = "aten::matmul"
    # MM = "aten::mm"


#
# model = OPTModel.from_pretrained("facebook/opt-125m").eval()
# x = torch.randint(0, model.config.vocab_size, (1, 1024))
#
MM_KEYS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::einsum", "aten::linear")
#
# with profile(
#         activities=[ProfilerActivity.CPU],  # add ProfilerActivity.CUDA if running on GPU
#         record_shapes=True,
# ) as prof:
#     with torch.inference_mode():
#         _ = model(input_ids=x)
#
# # group_by_input_shape keeps distinct (op, shape) entries
# # events = [
# #     e for e in prof.key_averages(group_by_input_shape=True)
# #     if any(k in e.key for k in MM_KEYS)
# # ]
#
# events = prof.key_averages(group_by_input_shape=True)
#
# for i, e in enumerate(events, 1):
#     shapes = getattr(e, "input_shapes", None)
#     print(f"{i:03d} {e.key:>14}  count={e.count}  shapes={shapes}")

from torch.profiler import profile, ProfilerActivity, record_function
from transformers import OPTModel
from transformers.models.opt.modeling_opt import OPTAttention


def install_layer_scopes(model):
    layers = model.decoder.layers

    for idx, layer in enumerate(layers):
        # --- Layer scope ---
        def pre_layer(mod, inputs, idx=idx):
            rf = record_function(f"L{idx}:layer")
            mod.__dict__["_rf_layer"] = rf
            rf.__enter__()                 # enter range
            # DO NOT return anything

        def post_layer(mod, inputs, output, idx=idx):
            rf = mod.__dict__.pop("_rf_layer", None)
            if rf is not None:
                rf.__exit__(None, None, None)

        layer.register_forward_pre_hook(pre_layer)
        layer.register_forward_hook(post_layer)

        # --- Attention scope (optional) ---
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, OPTAttention):
            def pre_attn(mod, inputs, idx=idx):
                rf = record_function(f"L{idx}:attn")
                mod.__dict__["_rf_attn"] = rf
                rf.__enter__()

            def post_attn(mod, inputs, output, idx=idx):
                rf = mod.__dict__.pop("_rf_attn", None)
                if rf is not None:
                    rf.__exit__(None, None, None)

            attn.register_forward_pre_hook(pre_attn)
            attn.register_forward_hook(post_attn)

        # --- FFN scope (optional) ---
        for name in ("fc1", "fc2"):
            sub = getattr(layer, name, None)
            if sub is None:
                continue

            def pre_ffn(mod, inputs, idx=idx, name=name):
                rf = record_function(f"L{idx}:ffn.{name}")
                mod.__dict__[f"_rf_ffn_{name}"] = rf
                rf.__enter__()

            def post_ffn(mod, inputs, output, idx=idx, name=name):
                rf = mod.__dict__.pop(f"_rf_ffn_{name}", None)
                if rf is not None:
                    rf.__exit__(None, None, None)

            sub.register_forward_pre_hook(pre_ffn)
            sub.register_forward_hook(post_ffn)


class Problem:
    matmuls = []

    events_handled = []
    events_unhandled = []
    events_layers = []

    def eval_event(self, event, idx):
        name = event.name

        if name[0] == "L":
            self.events_layers.append((idx, event))
            return # skip layer annotations

        shapes = getattr(evt, "input_shapes", None)

        if name == Key.MATMUL:
            self.events_handled.append((idx, event))
            return # lowered to batched matmul

        if name == Key.BMM:
            # TODO verify last handled event is matmul with matching shape
            # 1781 aten::bmm  shapes=[[12, 1024, 64], [12, 64, 1024]]

            assert len(shapes) == 2
            assert len(shapes[0]) == 3 # 12x operand (sequence_length, head_dim)
            assert len(shapes[1]) == 3 # 12x operand (head_dim, sequence_length)

            # TODO option to combine into larger matmuls

            for _ in range(shapes[0][0]):
                self.matmuls.append((idx, shapes[0][1], shapes[0][2], shapes[1][2]))

            self.events_handled.append((idx, event))
            return

        if name == Key.LINEAR:
            self.events_handled.append((idx, event))
            return # lowered to add matmul

        if name == Key.ADDMM:
            # TODO verify last handled event is linear with matching shape
            # 2064 aten::addmm  shapes=[[768], [1024, 3072], [3072, 768], [], []]

            assert len(shapes) == 5
            assert len(shapes[0]) == 1 # bias vector
            assert len(shapes[1]) == 2 # operand (sequence_length x hidden_dim)
            assert len(shapes[2]) == 2 # operand (hidden_dim x model_dim)
            assert len(shapes[3]) == 0
            assert len(shapes[4]) == 0

            self.matmuls.append((idx, shapes[1][0], shapes[1][1], shapes[2][1]))

            self.events_handled.append((idx, event))
            return

        self.events_unhandled.append((idx, event))

        return;

    def collect_gemms(self):
        idx_lower = 0
        idx_upper = 0

        for (idx, event) in self.events_layers:
            if event.name == "L0:layer":
                idx_lower = idx
            if event.name == "L1:layer":
                idx_upper = idx
                break

        for (idx, m, n, k) in self.matmuls:
            if idx < idx_lower:
                continue
            if idx > idx_upper:
                break

            print(idx, m, n, k)

model = OPTModel.from_pretrained("facebook/opt-125m").eval()
install_layer_scopes(model)

x = torch.randint(0, model.config.vocab_size, (1, 1024))

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with torch.inference_mode():
        _ = model(input_ids=x)

p = Problem()

# You’ll now see explicit layer markers
for i, evt in enumerate(prof.events()):
    p.eval_event(evt, i)

    # print(f"{i:03d} {evt.name}  shapes={shapes}")
    # if evt.name.startswith("UserAnnotation#"):
    #     print(evt.name)

p.collect_gemms()

# events = prof.key_averages()

# for i, e in enumerate(events, 1):
#     shapes = getattr(e, "input_shapes", None)
#   print(f"{i:03d} {e.key:>14}  count={e.count}  shapes={shapes}")

# TODO add tooling to filter events
# remove linear followed by addmm with same shape
# remove matmul followed by bmm with same shape
# transform inputs into GEMMs
# generate directory with yaml files
# LayerNorm, bias add, reshape/view/transpose/as_strided, masking, softmax, GELU/ReLU -> static timings