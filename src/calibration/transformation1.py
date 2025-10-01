def read_mesor(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
        print(lines)
    # # header
    # lat = parse_header_value(lines, "location.latitude")
    # lon = parse_header_value(lines, "location.longitude")
    # alt = parse_header_value(lines, "location.altitude")
    # # find data start
    # try:
    #     start_idx = lines.index("#begindata") + 1
    # except ValueError:
    #     raise RuntimeError("MESOR file missing #begindata")
    # # load table (tab or whitespace separated)
    # df = pd.read_csv(
    #     pd.compat.StringIO("\n".join(lines[start_idx:])),
    #     sep=r"\s+|\t+", engine="python", header=None,
    #     names=["datetime","GHI","DNI","DHI","tempC","mbar"]
    # )
    # # to datetime with local tz, then UTC
    # df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d%H:%M:%S")
    # df = df.set_index("datetime")
    # df.index = df.index.tz_localize(tz_str, nonexistent='shift_forward', ambiguous='NaT') \
    #                    .tz_convert("UTC")
    # # physics fields
    # df["pressure_Pa"] = (df["mbar"] * 100.0).astype(float)
    # df["tempC"] = df["tempC"].astype(float)
    # meta = dict(lat=float(lat), lon=float(lon), alt=float(alt))
    # return df, meta

read_mesor('../../data/raw/PSA_timeSeries_Metas.csv')