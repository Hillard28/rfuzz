"""
Code for testing rfuzz performance
"""

import time
import polars as pl
from rfuzz import ratio, partial_ratio
import polars_ds as pds
from rapidfuzz import fuzz

replications = 10000
df = pl.concat(
    [
        pl.DataFrame(
            {
                "str1": [
                    "john smith",
                    "smith john",
                    "john  smith",
                    "john albertson",
                    "albert johnson",
                    "mary willis",
                    "john",
                    None,
                    "john smith",
                    None,
                ],
                "str2": [
                    "john smith",
                    "john smith",
                    "john smith",
                    "john smith",
                    "john smith",
                    "john smith",
                    "john smith",
                    "john smith",
                    None,
                    None,
                ],
            }
        )
        for i in range(0, replications)
    ]
)

times = {
    "ratio": 0,
    "partial ratio": 0,
    "rf ratio": 0,
    "apply rf ratio": 0,
    "apply rf partial ratio": 0,
}
tests = 5
for i in range(0, tests):
    start_time = time.time()
    df = df.with_columns(ratio("str1", "str2").alias("ratio"))
    times["ratio"] += time.time() - start_time

    start_time = time.time()
    df = df.with_columns(partial_ratio("str1", "str2").alias("partial_ratio"))
    times["partial ratio"] += time.time() - start_time

    start_time = time.time()
    df = df.with_columns(pds.str_leven("str1", "str2", return_sim=True).alias("rf_ratio"))
    times["rf ratio"] += time.time() - start_time

    start_time = time.time()
    df = df.with_columns(
        pl.struct("str1", "str2")
        .map_elements(lambda x: fuzz.ratio(x["str1"], x["str2"]), return_dtype=pl.Float64)
        .alias("apply_rf_ratio")
    )
    times["apply rf ratio"] += time.time() - start_time

    start_time = time.time()
    df = df.with_columns(
        pl.struct("str1", "str2")
        .map_elements(
            lambda x: fuzz.partial_ratio(x["str1"], x["str2"]), return_dtype=pl.Float64
        )
        .alias("apply_rf_partial_ratio")
    )
    times["apply rf partial ratio"] += time.time() - start_time
df = df.with_columns(
    (pl.col("apply_rf_ratio") / 100).name.keep(),
    (pl.col("apply_rf_partial_ratio") / 100).name.keep()
)

times["ratio"] /= tests
times["partial ratio"] /= tests
times["rf ratio"] /= tests
times["apply rf ratio"] /= tests
times["apply rf partial ratio"] /= tests

print(times)

print(df.head(10))
