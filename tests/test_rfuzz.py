import polars as pl
from rfuzz import ratio, partial_ratio

df = pl.DataFrame(
    {
        "str1": ["john smith", "smith john", "john albertson", "albert johnson", "mary willis", "john"],
        "str2": ["john smith", "john smith", "john smith", "john smith", "john smith", "john smith"],
    }
)

df.with_columns(ratio("str1", "str2").alias("gram"))
df.with_columns(partial_ratio("str1", "str2").alias("gram"))
