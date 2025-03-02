from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

def gram(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="gram",
        is_elementwise=True,
    )