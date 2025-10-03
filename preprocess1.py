from pathlib import Path
import pandas as pd
from io import StringIO

def extract_after_begindata(input_csv: str, output_csv: str | None = None) -> str:
    """
    Find the line starting with '#begindata' (case-insensitive) in a CSV-like file,
    and save everything after that line into a new CSV file.
    Returns the output file path.
    """
    in_path = Path(input_csv)
    out_path = Path(output_csv) if output_csv else in_path.with_name(in_path.stem + "_after_begindata.csv")

    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    marker_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith("#begindata"):
            marker_idx = i
            break

    if marker_idx is None:
        raise RuntimeError("No line starting with '#begindata' found (case-insensitive).")

    # Write everything after the marker exactly as-is
    data_lines = lines[marker_idx + 1 :]
    with out_path.open("w", encoding="utf-8", newline="") as out:
        out.writelines(data_lines)

    print(f"Found '#begindata' at line {marker_idx}. Wrote {len(data_lines)} data lines to {out_path}")
    return str(out_path)


def extract_begindata_with_columns(input_csv: str, output_csv: str | None = None):
    """
    Extract rows after '#begindata' and save only the first five columns
    with names: GHI, DNI, DHI, temperature, pressure.
    """
    in_path = Path(input_csv)
    out_path = Path(output_csv) if output_csv else in_path.with_name(in_path.stem + "_after.csv")

    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find '#begindata' line (case-insensitive, leading spaces allowed)
    marker_idx = next(
        (i for i, line in enumerate(lines) if line.lstrip().lower().startswith("#begindata")),
        None
    )
    if marker_idx is None:
        raise RuntimeError("No line starting with '#begindata' (case-insensitive) was found.")

    # Read everything after the marker; infer delimiter (comma/tab/space) automatically
    text = "".join(lines[marker_idx + 1 :])
    df = pd.read_csv(StringIO(text), header=None, sep=None, engine="python")

    if df.shape[1] < 6:
        raise ValueError(f"Expected at least 5 columns after '#begindata', but found {df.shape[1]}.")

    df = df.iloc[:, :6]
    df.columns = ["Date", "GHI", "DNI", "DHI", "temperature", "pressure"]
    df.to_csv(out_path, index=False)
    return str(out_path)



if __name__ == "__main__":
    # Example usage:
    # extract_after_begindata("PSA_timeSeries_Metas.csv")
    extract_begindata_with_columns("PSA_timeSeries_Metas.csv")
