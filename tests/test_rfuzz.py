import polars as pl
from rfuzz import ratio, partial_ratio

df = pl.DataFrame(
    {
        "str1": [
            "john smith",
            "smith john",
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
            None,
            None,
        ],
    }
)

print(
    df.with_columns(
        ratio("str1", "str2").alias("ratio"),
        partial_ratio("str1", "str2").alias("partial_ratio"),
    )
)
