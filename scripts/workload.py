from transformers import OPTModel, AutoTokenizer
from pytorch2timeloop import convert_model_with_sample_input
from pathlib import Path
import torch

# Load the model and tokenizer
model = OPTModel.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model.eval()

x = torch.randint(0, model.config.vocab_size, (1, 1024))

# Convert the model using a sample input
convert_model_with_sample_input(
    model=model,
    sample_input=x,
    batch_size=1,
    model_name="opt-125m",
    save_dir=Path("./timeloop_output"),
    exception_module_names=[]
)
