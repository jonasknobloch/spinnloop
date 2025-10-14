import typer
from typing_extensions import Annotated

import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity, record_function
from transformers import OPTModel

from enum import Enum
from collections import defaultdict

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def trace(
        model: Annotated[str, typer.Argument(help="")] = "facebook/opt-125m",
        sequence_length: Annotated[int, typer.Argument(help="")] = 512,
        batch_attn_heads: Annotated[bool, typer.Argument(help="")] = True,
):
    model = OPTModel.from_pretrained(model).eval()

    _scope_model(model)

    x = torch.randint(0, model.config.vocab_size, (1, sequence_length))

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.inference_mode():
            _ = model(input_ids=x)

    matmuls = []
    static = []
    names = {}

    start = False

    for i, evt in enumerate(prof.events()):
        if evt.name[:2] == "L0":
            start = True

        if not start:
            continue

        if start and evt.name[0] == "G":
            break

        matmuls = matmuls + _extract_matmuls(evt, i)
        static = static + _extract_static(evt, i)
        names[evt.name] = names.get(evt.name, 0) + 1

    matmuls_grouped = _group_matmuls(matmuls)
    static_grouped = _group_matmuls(static)
    names_sorted = dict(sorted(names.items(), key=lambda x: x[1], reverse=True))

    for key, value in matmuls_grouped.items():
        print(key, len(value))

    for key, value in static_grouped.items():
        print(key, len(value))

    # for key, value in names_sorted.items():
    #     print(key, value)


def _scope_module(obj, attr: str | None = None, tag: str = ""):
    if not tag:
        raise ValueError("Profiler scope must have a non-empty tag")

    module = getattr(obj, attr, None) if attr else obj

    if module is None:
        raise AttributeError(f"Attribute {attr!r} not found on {type(obj).__name__}")

    if not isinstance(module, nn.Module):
        raise TypeError(f"{type(module).__name__} is not an 'nn.Module'")

    def _pre(mod, inputs):
        rf = record_function(tag)
        rf.__enter__()
        m = mod.__dict__.setdefault("_rf_map", {})
        s = m.setdefault(tag, [])
        s.append(rf)

    def _post(mod, inputs, output):
        m = mod.__dict__.get("_rf_map", None)
        s = m.get(tag)

        if not s:
            raise RuntimeError(f"No open profiler scope for tag {tag!r}")

        rf = s.pop()
        rf.__exit__(None, None, None)

        if not s:
            m.pop(tag, None)

        if not m:
            mod.__dict__.pop("_rf_map", None)

    module.register_forward_pre_hook(_pre)
    module.register_forward_hook(_post)


def _scope_model(model):
    # print(model)

    _scope_module(model.decoder, "embed_tokens", "G:embed_tokens")
    _scope_module(model.decoder, "embed_positions", "G:embed_positions")

    for idx, layer in enumerate(model.decoder.layers):
        _scope_module(layer, "self_attn_layer_norm", f"L{idx}:self_attn_layer_norm")

        attn = getattr(layer, "self_attn", None)

        if attn is None:
            raise AttributeError(f"Decoder layer has not attribute 'self_attn'")

        for proj_attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            _scope_module(attn, proj_attr, f"L{idx}:self_attn.{proj_attr}")

        _scope_module(layer, "final_layer_norm", f"L{idx}:final_layer_norm")
        _scope_module(layer, "fc1", f"L{idx}:fc1")
        _scope_module(layer, "activation_fn", f"L{idx}:activation_fn")
        _scope_module(layer, "fc2", f"L{idx}:fc2")

    _scope_module(model.decoder, "final_layer_norm", "G:final_layer_norm")

    # TODO scope lm_head for OPTForCausalLM

class Key(str, Enum):
    ADDMM = "aten::addmm"
    BMM = "aten::bmm"
    LINEAR = "aten::linear"
    MATMUL = "aten::matmul"
    MM = "aten::mm" # not used in OPT

    ADD = "aten::add"
    LAYER_NORM = "aten::layer_norm"
    SOFTMAX = "aten::softmax"

def _extract_matmuls(event, idx):
    name = event.name

    shapes = getattr(event, "input_shapes", None)

    matmuls = []

    if name == Key.MATMUL:
        return matmuls # lowered to batched matmul

    if name == Key.BMM:
        assert len(shapes) == 2

        assert len(shapes[0]) == 3  # 12x operand (sequence_length, head_dim)
        assert len(shapes[1]) == 3  # 12x operand (head_dim, sequence_length)

        # TODO option to combine into larger matmuls

        for _ in range(shapes[0][0]):
            matmuls.append((idx, shapes[0][1], shapes[0][2], shapes[1][2]))

        return matmuls

    if name == Key.LINEAR:
        return matmuls # lowered to add matmul

    if name == Key.ADDMM:
        assert len(shapes) == 5

        assert len(shapes[0]) == 1  # bias vector
        assert len(shapes[1]) == 2  # operand (sequence_length x hidden_dim)
        assert len(shapes[2]) == 2  # operand (hidden_dim x model_dim)
        assert len(shapes[3]) == 0
        assert len(shapes[4]) == 0

        matmuls.append((idx, shapes[1][0], shapes[1][1], shapes[2][1]))

        return matmuls

    if name == Key.MM:
        raise NotImplementedError(f"Handler not implemented for event {event.name!r}")

    # print(f"Skipping event {event.name!r}")

    return matmuls

def _group_matmuls(data):
    grouped = defaultdict(list)

    for first, *rest in data:
        grouped[tuple(rest)].append(first)

    return dict(grouped)

def _extract_static(event, idx):
    name = event.name

    shapes = getattr(event, "input_shapes", None)

    if name == Key.ADD:
        # [[1, 512, 768], [1, 512, 768], []]
        # [[512, 768], [512, 768], []]
        assert [len(s) for s in shapes] in [[3, 3, 0], [2, 2, 0]]
        return [(idx, name, tuple(tuple(s) for s in shapes))]

    if name == Key.LAYER_NORM:
        # [[1, 512, 768], [], [768], [768], [], []]
        # [[512, 768], [], [768], [768], [], []]
        assert [len(s) for s in shapes] in [[3, 0, 1, 1, 0, 0], [2, 0, 1, 1, 0, 0] ]
        return [(idx, name, tuple(tuple(s) for s in shapes))]

    if name == Key.SOFTMAX:
        assert [len(s) for s in shapes] == [4, 0, 0] # [[1, 12, 512, 512], [], []]
        return [(idx, name, tuple(tuple(s) for s in shapes))]

    return []