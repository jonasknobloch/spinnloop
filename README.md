# spinnloop

```bash
docker build -t jonasknobloch/timeloop .
docker run -it -v .:/opt/project jonasknobloch/timeloop /bin/bash
```




OctopuScheduler can only generate uniform workloads, meaning each available PE has the same workload.
Timeloop can only geenrate uniform mappings, meaning the workload is distributed uniformly across available compute.


CLI tool to generate worklaods, and simukate wokloads on multiple Chips.

Just use layer shapes from gpt2

But what about non matmul operations

SINGLE CPU only run -> identify transformer blocks

spinnloop


```bash
git submodule add https://github.com/Accelergy-Project/pytorch2timeloop-converter vendor/pytorch2timeloop-converter
git submodule update --init --recursive

conda create --prefix ./env python=3.11
conda activate ./env

pip install -r vendor/pytorch2timeloop-converter/requirements.txt
pip install -e ./vendor/pytorch2timeloop-converter

pip install pandas
pip install openpyxl
 ```

2025-07-04__time_measurements__large_matmul__3072-512-768__only_1_dram_transfer_per_weight_tile__and__input_tile
