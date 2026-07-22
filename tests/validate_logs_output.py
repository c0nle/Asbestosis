import os
import re
import glob
import pandas as pd

LOG_DIR = "/home/rwth1954/Asbestosis/logs/train_test/output"  # adjust if needed
PATTERN = os.path.join(LOG_DIR, "cv_*.txt")

def parse_log(filepath):
    with open(filepath, "r", errors="replace") as f:
        text = f.read()

    result = {
        "file": os.path.basename(filepath),
        "model": None,
        "primary_label": None,
        "fold": None,
        "outcome": None,
        "detail": None,
        "macro_auc_test": None,
        "occupational_disease_auc": None,
        "mixed_shapes_auc": None,
    }

    # --- Header: model, fold ---
    header = re.search(r"model=([\w]+)\s+fold=(\d+)", text)
    if header:
        result["model"] = header.group(1)
        result["fold"] = int(header.group(2))

    # --- Primary label from sampler line ---
    primary_match = re.search(r"Train sampler: primary_balanced \(([\w_]+)\)", text)
    if primary_match:
        result["primary_label"] = primary_match.group(1)

    # --- Outcome detection (order matters: most specific first) ---

    # 1. Disk quota / pretrained weights warning (not fatal by itself)
    quota_warn = "Disk quota exceeded" in text

    # 2. Skipped: degenerate primary label
    degen_primary = re.search(
        r"Primary label '([\w_]+)' is degenerate in train for fold \d+", text
    )
    if degen_primary:
        result["outcome"] = "skipped"
        result["primary_label"] = degen_primary.group(1)
        result["detail"] = f"Primary label '{degen_primary.group(1)}' is degenerate (single class in train)"
        return result

    # 3. Error: RuntimeError (shape mismatch etc.)
    runtime_err = re.search(r"RuntimeError: (.+)", text)
    if runtime_err:
        result["outcome"] = "error"
        err_msg = runtime_err.group(1).strip()
        result["detail"] = f"RuntimeError: {err_msg}"
        if quota_warn:
            result["detail"] += " [+ disk quota: random init used]"
        return result

    # 4. Error: Traceback present
    traceback = re.search(r"Traceback \(most recent call last\)", text)
    if traceback:
        # Try to get last exception line
        last_exc = re.findall(r"(\w+Error|\w+Exception): (.+)", text)
        if last_exc:
            exc_type, exc_msg = last_exc[-1]
            result["outcome"] = "error"
            result["detail"] = f"{exc_type}: {exc_msg.strip()}"
        else:
            result["outcome"] = "error"
            result["detail"] = "Unknown exception (see traceback)"
        if quota_warn:
            result["detail"] += " [+ disk quota: random init used]"
        return result

    # 5. Successful: look for final_test macro line
    final_test = re.search(
        r"final_test macro:.*?macro_auc=([\d.]+)", text
    )
    if final_test:
        result["outcome"] = "success"
        result["macro_auc_test"] = float(final_test.group(1))
        if quota_warn:
            result["detail"] = "Warning: disk quota exceeded, random init used (no pretrained weights)"

        # Focus label AUCs
        occ = re.search(r"occupational_disease auc=([\d.]+)", text)
        if occ:
            result["occupational_disease_auc"] = float(occ.group(1))
        mix = re.search(r"mixed_shapes auc=([\d.]+)", text)
        if mix:
            result["mixed_shapes_auc"] = float(mix.group(1))
        return result

    # 6. Unknown / incomplete
    result["outcome"] = "unknown/incomplete"
    result["detail"] = "No final_test result and no clear error found"
    if quota_warn:
        result["detail"] += "; disk quota warning present"
    return result


# --- Find and parse all matching files ---
files = sorted(glob.glob(PATTERN))

# Also check working dir as fallback
if not files:
    print(f"No log files found in {LOG_DIR}, checking current directory...")
    files = sorted(glob.glob("cv_*.txt"))

print(f"Found {len(files)} log file(s) matching pattern.")

rows = [parse_log(f) for f in files]
df = pd.DataFrame(rows)

# Reorder columns
cols = ["file","model","fold","primary_label","outcome","detail",
        "macro_auc_test","occupational_disease_auc","mixed_shapes_auc"]
df = df[[c for c in cols if c in df.columns]]

print(df.to_string(index=False))
out_path = os.path.join(LOG_DIR, "cv_log_summary.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")