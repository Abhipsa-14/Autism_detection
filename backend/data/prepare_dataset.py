"""
Merge UCI Autism Screening datasets (Adult + Adolescent + Child) into one
clean CSV, adding an `age_group` feature.

Input : data/raw/{adult,adolescent,child}/*.arff   (from download_datasets.py)
Output: data/autism_merged.csv

Notes
-----
* The raw ARFF column for jaundice is misspelled `jundice`; we normalise to `jaundice`.
* Target `Class/ASD` is YES/NO in the ARFF -> mapped to 1/0.
* `?` is the ARFF missing-value marker.
* This merged file is the new training source. The original single-cohort
  `autism.csv` (adult only) is left untouched for backwards-compatibility.
"""
import re
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "autism_merged.csv"

ARFF_FILES = {
    "adult": RAW_DIR / "adult" / "Autism-Adult-Data.arff",
    "adolescent": RAW_DIR / "adolescent" / "Autism-Adolescent-Data.arff",
    "child": RAW_DIR / "child" / "Autism-Child-Data.arff",
}


def parse_arff(path: Path):
    """Return (attribute_names, list_of_row_dicts) from an ARFF file."""
    attributes = []
    rows = []
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                # @attribute <name> <type/spec>
                m = re.match(r"@attribute\s+([^\s]+)\s+", line, re.IGNORECASE)
                if m:
                    attributes.append(m.group(1))
            elif lower.startswith("@data"):
                in_data = True
            elif in_data and not line.startswith("%"):
                # CSV split that respects single-quoted values
                values = next(csv.reader([line], quotechar="'", skipinitialspace=True))
                if len(values) != len(attributes):
                    # skip malformed rows
                    continue
                rows.append(dict(zip(attributes, values)))
    return attributes, rows


def clean_value(v: str):
    v = v.strip().strip("'").strip()
    if v in ("?", "", "NaN", "nan"):
        return None
    return v


def main():
    all_rows = []
    header = None

    for cohort, path in ARFF_FILES.items():
        if not path.exists():
            print(f"[WARN] missing {path}; run download_datasets.py first")
            continue
        attrs, rows = parse_arff(path)
        print(f"[{cohort:<10}] parsed {len(rows)} rows, {len(attrs)} attributes")

        for r in rows:
            clean = {k: clean_value(v) for k, v in r.items()}
            # Normalise the misspelled jaundice column
            if "jundice" in clean:
                clean["jaundice"] = clean.pop("jundice")
            clean["age_group"] = cohort
            all_rows.append(clean)

    if not all_rows:
        raise SystemExit("No rows parsed. Did the download succeed?")

    # Unified, ordered output columns
    out_cols = [
        "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
        "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
        "age", "gender", "ethnicity", "jaundice", "austim",
        "contry_of_res", "used_app_before", "result", "relation",
        "age_group", "Class/ASD",
    ]

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in out_cols})

    print(f"\nMerged {len(all_rows)} total rows -> {OUT_PATH}")

    # Quick class + cohort summary
    from collections import Counter
    cls = Counter(r.get("Class/ASD") for r in all_rows)
    grp = Counter(r.get("age_group") for r in all_rows)
    print(f"   Class/ASD: {dict(cls)}")
    print(f"   age_group: {dict(grp)}")


if __name__ == "__main__":
    main()
