import os
import json

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
    SOFTMAX = "aten::softmax"
    SOFTMAX_INTERNAL = "aten::_softmax"
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

# Generic helper to register a profiler scope around any module
def _register_scope(obj, attr: str | None = None, tag: str = ""):
    module = getattr(obj, attr, obj) if attr else obj
    if module is None:
        return
    def _pre(mod, inputs):
        rf = record_function(tag)
        rf.__enter__()
        # store by tag to support nested/recursive calls safely
        m = mod.__dict__.setdefault("_rf_map", {})
        m[tag] = rf
    def _post(mod, inputs, output):
        m = mod.__dict__.get("_rf_map", None)
        rf = None if m is None else m.pop(tag, None)
        if rf is not None:
            rf.__exit__(None, None, None)
    module.register_forward_pre_hook(_pre)
    module.register_forward_hook(_post)


def install_layer_scopes(model):
    layers = model.decoder.layers

    # --- Global scopes outside decoder layers ---
    _register_scope(model.decoder, "embed_tokens", "G:embed_tokens")
    _register_scope(model.decoder, "embed_positions", "G:embed_positions")
    _register_scope(model.decoder, "final_layer_norm", "G:final_layer_norm")
    _register_scope(model, "lm_head", "G:lm_head")

    for idx, layer in enumerate(layers):
        # --- Layer scope ---
        _register_scope(layer, None, f"L{idx}:layer")

        # --- Attention scope (optional) ---
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, OPTAttention):
            _register_scope(attn, None, f"L{idx}:attn")

            # Projections inside attention
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is None:
                    continue
                _register_scope(proj, None, f"L{idx}:attn.{proj_name}")

        # --- Per-layer norms ---
        for norm_name in ("self_attn_layer_norm", "final_layer_norm"):
            norm = getattr(layer, norm_name, None)
            if norm is None:
                continue
            _register_scope(norm, None, f"L{idx}:{norm_name}")

        # --- FFN scope (optional) ---
        for name in ("fc1", "fc2"):
            sub = getattr(layer, name, None)
            if sub is None:
                continue
            _register_scope(sub, None, f"L{idx}:ffn.{name}")


class Problem:
    matmuls = []
    softmaxes = []
    trace = []

    events_handled = []
    events_unhandled = []
    events_layers = []
    current_layer = None

    def eval_event(self, event, idx):
        name = event.name

        if name[0] == "L":
            self.events_layers.append((idx, event))
            self.current_layer = name
            # trace layer marker
            self.trace.append((self.current_layer, name, None))
            return  # skip layer annotations

        shapes = getattr(event, "input_shapes", None)

        if name == Key.SOFTMAX or name == Key.SOFTMAX_INTERNAL:
            # Expect a single tensor input; non-tensor args (dim, dtype) are not recorded in input_shapes
            inp_shape = None
            if shapes is not None and len(shapes) >= 1:
                inp_shape = shapes[0]
            self.softmaxes.append((idx, inp_shape))
            self.events_handled.append((idx, event))
            self.trace.append((self.current_layer, name, inp_shape))
            return

        if name == Key.MATMUL:
            self.events_handled.append((idx, event))
            self.trace.append((self.current_layer, name, shapes))
            return  # lowered to batched matmul

        if name == Key.BMM:
            # TODO verify last handled event is matmul with matching shape
            # 1781 aten::bmm  shapes=[[12, 1024, 64], [12, 64, 1024]]

            assert len(shapes) == 2
            assert len(shapes[0]) == 3  # 12x operand (sequence_length, head_dim)
            assert len(shapes[1]) == 3  # 12x operand (head_dim, sequence_length)

            # TODO option to combine into larger matmuls

            for _ in range(shapes[0][0]):
                self.matmuls.append((idx, shapes[0][1], shapes[0][2], shapes[1][2]))

            self.events_handled.append((idx, event))
            self.trace.append((self.current_layer, name, shapes))
            return

        if name == Key.LINEAR:
            self.events_handled.append((idx, event))
            self.trace.append((self.current_layer, name, shapes))
            return  # lowered to add matmul

        if name == Key.ADDMM:
            # TODO verify last handled event is linear with matching shape
            # 2064 aten::addmm  shapes=[[768], [1024, 3072], [3072, 768], [], []]

            assert len(shapes) == 5
            assert len(shapes[0]) == 1  # bias vector
            assert len(shapes[1]) == 2  # operand (sequence_length x hidden_dim)
            assert len(shapes[2]) == 2  # operand (hidden_dim x model_dim)
            assert len(shapes[3]) == 0
            assert len(shapes[4]) == 0

            self.matmuls.append((idx, shapes[1][0], shapes[1][1], shapes[2][1]))

            self.events_handled.append((idx, event))
            self.trace.append((self.current_layer, name, shapes))
            return

        self.events_unhandled.append((idx, event))
        self.trace.append((self.current_layer, name, shapes))
        return

    def serialize(self, start = "", end = ""):
        out_dir = "../layer_shapes/opt"

        os.makedirs(out_dir, exist_ok=True)

        # write sequential trace
        trace_path = os.path.join(out_dir, "trace.tsv")
        with open(trace_path, "w") as f:
            f.write("layer\tkey\tinput\n")
            for (layer, key, inp) in self.trace:
                try:
                    inp_str = json.dumps(inp)
                except Exception:
                    inp_str = repr(inp)
                f.write(f"{layer}\t{key}\t{inp_str}\n")

        template = """{{include_text('../problem_base.yaml')}}
problem:
  <<<: *problem_base
  instance: {{C: {C}, M: {M}, P: {P}}}
"""
        idx_lower = -1
        idx_upper = -1

        if start != "" or end != "":
            for (idx, event) in self.events_layers:
                if event.name == start:
                    idx_lower = idx
                if event.name == end:
                    idx_upper = idx
                    break

        for (i, (idx, m, n, k)) in enumerate(self.matmuls):
            if idx_lower > -1 and idx < idx_lower:
                continue
            if idx_upper > -1 and idx > idx_upper:
                break

            cfg = {
                "C": m,
                "M": n,
                "P": k,
            }

            content = template.format(**cfg)
            name = os.path.join(out_dir, f"{i:03d}.yaml")
            with open(name, "w") as f:
                f.write(content)


model = OPTModel.from_pretrained("facebook/opt-125m").eval()
install_layer_scopes(model)

x = torch.randint(0, model.config.vocab_size, (1, 512))

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

p.serialize("L0:layer", "L11:layer")

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
