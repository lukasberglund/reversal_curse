from typing import Optional


def filter_df(
    df,
    model: Optional[str] = "davinci",
    num_re: Optional[int] = 50,
    num_rg: Optional[int] = 300,
    num_ug: Optional[int] = 300,
    num_ce: Optional[int] = 0,
    num_ugp: Optional[int] = 0,
    num_rgp: Optional[int] = 0,
    num_rep: Optional[int] = 0,
    owt: Optional[float] = 0,
):
    """
    Filter a dataframe based on the assistant config parameters.
    All parameters are set to their default values.
    So if you want to get a sweep with variable num_re, you need to set num_re=None.
    """
    if model is not None:
        df = df[df["model"] == model]
    if num_re is not None:
        df = df[df["num_re"] == num_re]
    if num_rg is not None:
        df = df[df["num_rg"] == num_rg]
    if num_ug is not None:
        df = df[df["num_ug"] == num_ug]
    if num_ug is None or num_rg is None:
        df = df[df["num_ug"] == df["num_rg"]]
    if num_ce is not None:
        df = df[df["num_ce"] == num_ce]
    if num_ugp is not None:
        df = df[df["num_ugp"] == num_ugp]
    if num_rgp is not None:
        df = df[df["num_rgp"] == num_rgp]
    if num_ugp is None or num_rgp is None:
        df = df[df["num_ugp"] == df["num_rgp"]]
    if num_rep is not None:
        df = df[df["num_rep"] == num_rep]
    if owt is not None:
        df = df[df["owt"] == owt]
    return df
