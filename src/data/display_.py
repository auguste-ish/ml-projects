from IPython.display import display_html
import pandas as pd


def display_dfs(
    dfs: list[pd.DataFrame], 
    titles: list | None = None,
    axis: int = 1,
):
    titles = titles or [""] * len(dfs)
    layout = "block" if axis == 0 else "inline-block"
    html = "".join(
        f'<div style="display:{layout};margin-right:20px;margin-bottom:10px;vertical-align:top">'
        f"<h4>{t}</h4>"
        f"{df.style.set_table_attributes('')._repr_html_()}</div>" # type: ignore
        for t, df in zip(titles, dfs)
    )
    display_html(html, raw=True)


__all__ = [
    "display_dfs",
]
