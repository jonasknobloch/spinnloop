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


