import re
import pandas as pd
from datetime import timedelta
import plotly.express as px
import plotly.figure_factory as ff

# xls = pd.read_excel("../data/time_measurements__01__qkv_+_att-out_linear__512x768_768x768_512x768.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/time_measurements__02__mlp_linear_1__512x768_768x3072_512x3072.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/time_measurements__03__mlp_linear_2__512x3072_3072x768_512x768.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/time_measurements__04__bmm_1__512x64_64x512_512x512.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/time_measurements__04__bmm_2__512x512_512x64_512x64.xlsx", sheet_name=None)

xls = pd.read_excel("../data/2025-11-18__opt125m__measurements_prefill/time_measurements_full_opt_model_prefill_64context.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/2025-11-18__opt125m__measurements_prefill/time_measurements_full_opt_model_prefill_128context.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/2025-11-18__opt125m__measurements_prefill/time_measurements_full_opt_model_prefill_256context.xlsx", sheet_name=None)
# xls = pd.read_excel("../data/2025-11-18__opt125m__measurements_prefill/time_measurements_full_opt_model_prefill_512context.xlsx", sheet_name=None)

# print(xls.keys())

intervals = []

_avg_over_layers = True
_decoder_layer = 0
_worker_idx = 0

def _order(result):
    order = [
        "q_proj",
        "k_proj",
        "v_proj",
        "qk_bmm",
        "pv_bmm",
        "out_proj",
        "fc1",
        "fc2",
    ]

    order_map = {k: i for i, k in enumerate(order)}

    result["Layer name"] = result["Layer name"].str.replace(
        r"^decoder_layer_avg_", "", regex=True
    )

    result = (
        result.assign(_order=result["Layer name"].map(order_map))
        .sort_values("_order")
        .drop(columns="_order")
    )

    return result

all = []

for name, df in xls.items():
    m = re.findall(r'\d+', name)

    if len(m) != 1:
        continue

    # if name != f"WORKER #{_worker_idx}":
        # continue

    def estimate_cycles(decoder_layer, log_idx):
        if _avg_over_layers:
            foo = df[df["Layer name"].str.contains(f"decoder_layer_")]
        else:
            foo = df[df["Layer name"].str.contains(f"decoder_layer_{decoder_layer}_")]

        bar = foo[foo["Layer type"].isin(["OPTLinearSLayer", "OPTBMMSLayer"])]
        baz = bar[bar["log idx"] == log_idx]

        if _avg_over_layers:
            baz = baz.copy()

            baz["Layer name"] = baz["Layer name"].str.replace(
                r"decoder_layer_\d+_",
                "decoder_layer_avg_",
                regex=True,
            )

        result = (
            baz.groupby("Layer name").agg({
                "Duration [µs]": "sum",
                "Layer name": "count"
            })
            .rename(columns={"Layer name": "Row count"})
            .reset_index()
        )

        if _avg_over_layers:
            result["Duration [µs]"] /= 12

        result["Cycles"] = result["Duration [µs]"] * 150

        result = _order(result)

        return result

    print("\n### LOAD INPUT\n")
    foo = estimate_cycles(_decoder_layer, 5)  # load input
    print(foo)

    print("\n### LOAD WEIGHTS\n")
    bar = estimate_cycles(_decoder_layer, 7)  # load weights
    print(bar)

    print("\n### EXECUTE\n")
    baz = estimate_cycles(_decoder_layer, 8)  # execute MLA
    print(baz)

    # print("\n### STORE OUTPUT\n")
    # boo = estimate_cycles(_decoder_layer, 11)  # execute MLA
    # print(baz)

    print("\n### ALL \n")

    all_df = pd.concat([foo, bar, baz])
    # all_df = pd.concat([baz])

    result = (
        all_df
        .groupby("Layer name", as_index=False)
        .agg({
            "Duration [µs]": "sum",
            "Cycles": "sum",
            "Row count": "sum",
        })
    )

    result = _order(result)

    all.append(result)

    print(result)

avg = (
    pd.concat(all)
    .groupby("Layer name", as_index=False)
    .mean()
)

avg = _order(avg)

print(avg)

    # column_header = "Measurement Position"
    # column_header = "log idx"

    # input_transfer = df[df[column_header] == 5]["Duration [µs]"].iat[0] # DRAM only workers 0 - 31
    # input_reordering = df[df[column_header] == 6]["Duration [µs]"].iat[0]
    # weight_transfer = df[df[column_header] == 7]["Duration [µs]"].iat[0]  # 0, 32, 64, 96 DRAM; SRAM otherwise
    # mla_execution = df[df[column_header] == 8]["Duration [µs]"].iat[0]
    # output_transfer = df[df[column_header] == 9]["Duration [µs]"].iat[0]  # DRAM all workers

    # input_start = df[df[column_header] == 5]["Start [µs]"].iat[0]
    # reorder_start = df[df[column_header] == 6]["Start [µs]"].iat[0]
    # weight_start = df[df[column_header] == 7]["Start [µs]"].iat[0]
    # mla_start = df[df[column_header] == 8]["Start [µs]"].iat[0]
    # output_start = df[df[column_header] == 9]["Start [µs]"].iat[0]

    # input_end = df[df[column_header] == 5]["End [µs]"].iat[0]
    # reorder_end = df[df[column_header] == 6]["End [µs]"].iat[0]
    # weight_end = df[df[column_header] == 7]["End [µs]"].iat[0]
    # mla_end = df[df[column_header] == 8]["End [µs]"].iat[0]
    # output_end = df[df[column_header] == 9]["End [µs]"].iat[0]

    # print(f"{input_transfer/1000}ms [{input_transfer*150:6}] input_transfer")
    # print(f"{input_reordering/1000}ms [{input_reordering*150:6}] input_reordering")
    # print(f"{weight_transfer/1000}ms [{weight_transfer*150:6}] weight_transfer")
    # print(f"{mla_execution/1000}ms [{mla_execution*150:6}] mla_execution")
    # print(f"{output_transfer/1000}ms [{output_transfer*150:6}] output_transfer")

    # if int(m[0]) > 31:
    #     input_transfer = 0
    #     input_start = 0
    #     input_end = 0

    # intervals += [
        # dict(Task=m[0], Start=int(input_start), Finish=int(input_end), Delta=int(input_transfer), Type="input_transfer"),
        # dict(Task=m[0], Start=int(reorder_start), Finish=int(reorder_end), Delta=int(input_reordering), Type="input_reordering"),
        # dict(Task=m[0], Start=int(weight_start), Finish=int(weight_end), Delta=int(weight_transfer), Type="weight_transfer"),
        # dict(Task=m[0], Start=int(mla_start), Finish=int(mla_end), Delta=int(mla_execution), Type="mla_execution"),
        # dict(Task=m[0], Start=int(output_start), Finish=int(output_end), Delta=int(output_transfer), Type="output_transfer"),
    # ]

    # intervals.append([
    #     (int(input_start), int(input_end)),
    #     (int(weight_start), int(weight_end)),
    #     (int(mla_start), int(mla_end)),
    #     (int(output_start), int(output_end)),
    # ])

# df = pd.DataFrame(intervals)

# colors = {
    # "input_transfer": "rgb(255, 0, 0)",
    # "input_reordering": "rgb(0, 255, 255)",
    # "weight_transfer": "rgb(0, 255, 0)",
    # "mla_execution": "rgb(0, 0, 255)",
    # "output_transfer": "rgb(255, 255, 0)",
# }

# fig = ff.create_gantt(df, colors=colors, index_col="Type", group_tasks=True)

# fig.layout.xaxis.type = "linear"
# f = fig.full_figure_for_development(warn=False)
# fig['layout']['xaxis'].update({'type': None})

# fig.update_layout(
#     height=30 * 128
# )

# fig.show()
# fig.write_html("../out/stats.html", include_plotlyjs="cdn", auto_open=False)
