import polars as pl
from rfuzz import ratio

df = pl.DataFrame(
    {
        "str1": ["john smith", "smith john", "john albertson", "albert johnson", "mary willis"],
        "str2": ["john smith", "john smith", "john smith", "john smith", "john smith"],
    }
)

result = df.with_columns(ratio("str1", "str2").alias("gram"))
