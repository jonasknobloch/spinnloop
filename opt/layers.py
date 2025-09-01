from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing import List, Tuple

model_name = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

print(model.__class__.__name__)
print(model.model)
print(model.config)

# --- Generic TorchDispatch op tracker (logs everything) ---
from typing import Any

def _collect_tensors(obj, n=6):
    # Flatten nested (args, kwargs) and collect up to n tensors
    stack = [obj]
    out = []
    while stack and len(out) < n:
        x = stack.pop(0)
        if isinstance(x, (list, tuple)):
            stack[:0] = list(x)
        elif isinstance(x, dict):
            stack[:0] = list(x.values())
        elif torch.is_tensor(x):
            out.append(x)
    return out

class OpTracker(TorchDispatchMode):
    """Track *all* ops seen by TorchDispatch.

    Records op name, a few input tensor shapes/dtypes/devices, and a few output shapes.
    """
    def __init__(self, max_tensors: int = 6):
        super().__init__()
        self.max_tensors = max_tensors
        # Each event: (name, in_shapes, out_shapes)
        self.events: List[Tuple[str, list, list]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Derive a readable op name
        name = None
        if hasattr(func, 'name'):
            try:
                name = func.name()
            except Exception:
                name = str(func)
        if name is None and hasattr(func, 'overloadpacket'):
            name = getattr(func.overloadpacket, '__name__', str(func))
        if name is None:
            name = str(func)

        # Collect a handful of input tensor shapes for context
        in_tensors = _collect_tensors((args, kwargs), self.max_tensors)
        in_shapes = [tuple(t.shape) for t in in_tensors]

        # Call the actual op
        out = func(*args, **kwargs)

        # Collect a handful of output tensor shapes
        out_shapes = []
        if torch.is_tensor(out):
            out_shapes = [tuple(out.shape)]
        elif isinstance(out, (list, tuple)):
            out_shapes = [tuple(t.shape) for t in out if torch.is_tensor(t)][: self.max_tensors]
        elif isinstance(out, dict):
            out_shapes = [tuple(t.shape) for t in out.values() if torch.is_tensor(t)][: self.max_tensors]

        self.events.append((name, in_shapes, out_shapes))
        return out

    def report(self, top_k: int = 50) -> str:
        lines = ["== Ops observed in order =="]
        for i, (name, in_shapes, out_shapes) in enumerate(self.events, 1):
            lines.append(f"{i:03d}. {name} | in {in_shapes} -> out {out_shapes}")
        # Aggregate counts
        counts = {}
        for name, _, _ in self.events:
            counts[name] = counts.get(name, 0) + 1
        lines.append("\n-- Top ops --")
        for name, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]:
            lines.append(f"{name}: {cnt}")
        return "\n".join(lines)

    def to_tsv(self, path: str = "ops.tsv"):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["name", "in", "out"])
            for name, in_shapes, out_shapes in self.events:
                writer.writerow([name, str(in_shapes), str(out_shapes)])


# --- Demo: track all matmul-like ops for a single forward pass ---
sample_text = "Hello world"
inputs = tok(sample_text, return_tensors="pt")
with OpTracker() as ot:
    with torch.inference_mode():
        _ = model(**inputs)
print(ot.report())
ot.to_tsv("ops.tsv")
print("Saved ops.tsv")