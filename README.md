# spinnloop

```bash
docker build -t jonasknobloch/timeloop .
docker run -it -v .:/opt/project jonasknobloch/timeloop /bin/bash
docker run -it --entrypoint /bin/bash -v .:/opt/project jonasknobloch/timeloop
```

```bash
alias spinnloop="docker run -it --entrypoint spinnloop -v .:/opt/project jonasknobloch/timeloop"
```

```bash
git submodule add https://github.com/Accelergy-Project/pytorch2timeloop-converter vendor/pytorch2timeloop-converter
git submodule update --init --recursive

conda create --prefix ./env python=3.11
conda activate ./env

pip install -r vendor/pytorch2timeloop-converter/requirements.txt
pip install -e ./vendor/pytorch2timeloop-converter

pip install pandas
pip install openpyxl
pip install plotly
pip install typer
 ```

```bash
pip install hatchling
pip install -e .
```

| layer           | M   | N    | K    |
|-----------------|-----|------|------|
| qkv_with_linear | 512 | 768  | 768  |
| mlp_linear_1    | 512 | 3072 | 768  |
| mlp_linear_2    | 512 | 768  | 3072 |
| bmm1            | 512 | 512  | 64   |
| bmm2            | 512 | 64   | 512  |
