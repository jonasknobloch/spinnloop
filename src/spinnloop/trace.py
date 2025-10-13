import typer
from typing_extensions import Annotated

import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity, record_function
from transformers import OPTModel

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

    for i, evt in enumerate(prof.events()):
        print(i, evt)

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
    print(model)

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
