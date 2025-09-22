from transformers import OPTModel, AutoTokenizer
from pytorch2timeloop import convert_model_with_sample_input
from pathlib import Path
import torch

# Load the model and tokenizer
model = OPTModel.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model.eval()

# Prepare a sample input
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
sample_input = (inputs["input_ids"],)

# Convert the model using a sample input
convert_model_with_sample_input(
    model=model,
    sample_input=sample_input,
    batch_size=1,
    model_name="opt-125m",
    save_dir=Path("./timeloop_output"),
    exception_module_names=[]
)
