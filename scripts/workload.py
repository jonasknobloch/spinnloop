from transformers import GPT2Model
from pytorch2timeloop import convert_model_with_sample_input
from pathlib import Path
import torch

model = GPT2Model.from_pretrained("gpt2")

x = torch.randint(0, model.config.vocab_size, (1, 1024))

# Convert the model using a sample input
convert_model_with_sample_input(
    model=model,
    sample_input=x,
    batch_size=1,
    model_name="gpt2",
    save_dir=Path("./timeloop_output"),
    exception_module_names=[]
)
