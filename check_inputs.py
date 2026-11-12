# =============================================================================
# check_inputs.py  --  run BEFORE kaggle_merged_training.py (takes ~1 minute)
# Checks that every Kaggle input is the correct dataset, plus GPUs and internet.
#   !python $(find /kaggle/input -name check_inputs.py | head -1)
# =============================================================================
import os, re, glob, subprocess, urllib.request
import pandas as pd

ROOT = os.environ.get("MERGED_INPUT_ROOT", "/kaggle/input")
MAX_FILES = 300_000           # stop walking huge inputs (e.g. a DICOM copy) early
results = []


def check(name, ok, detail, fix=""):
    results.append((name, ok, detail, fix))


def print_run_mode():
    """Say whether this is a background commit (safe) or an interactive session."""
    mode = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")
    bar = "#" * 78
    if mode == "Batch":
        msg = ("RUN MODE: BACKGROUND COMMIT (Save & Run All)\n"
               "Safe to close the laptop. Output is kept under this version's Output tab.")
    elif mode == "Interactive":
        msg = ("RUN MODE: INTERACTIVE SESSION\n"
               "Output is LOST when the session stops. Use this for the quick test only;\n"
               "for the full run use Save Version -> Save & Run All (Commit).")
    else:
        msg = "RUN MODE: not on Kaggle (local run)"
    print(f"{bar}\n{msg}\n{bar}\n", flush=True)


print_run_mode()

# ---- one walk over all inputs ---------------------------------------------------
files = []
for top in sorted(glob.glob(os.path.join(ROOT, "*"))):
    n = 0
    for root, dirs, fs in os.walk(top, followlinks=True):
        dirs[:] = [d for d in dirs if d not in ("FLAIR", "T1w", "T1wCE", "T2w")]  # RSNA DICOM folders
        files += [os.path.join(root, f) for f in fs]; n += len(fs)
        if n > MAX_FILES: break
names = [os.path.basename(f) for f in files]
print(f"scanned {len(files):,} files under {ROOT}\n")


def find(pattern):
    rx = re.compile(pattern, re.I)
    return [f for f, b in zip(files, names) if rx.search(b)]


# ---- 1. GPUs --------------------------------------------------------------------
try:
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:
    gpus = []
check("GPU T4 x2", len(gpus) >= 2, f"{len(gpus)} GPU(s): " + "; ".join(g.split(" (")[0] for g in gpus),
      "Settings -> Accelerator -> GPU T4 x2")

# ---- 2. Internet (needed for ImageNet weights) ------------------------------------
try:
    urllib.request.urlopen("https://huggingface.co", timeout=10); net = True
except Exception:
    net = False
check("Internet", net, "on" if net else "off", "Settings -> Internet -> On (phone-verified account)")

# ---- 3. RSNA-MICCAI labels ----------------------------------------------------------
lab = find(r"^train_labels\.csv$")
if lab:
    d = pd.read_csv(lab[0]); ok = {"BraTS21ID", "MGMT_value"} <= set(d.columns) and len(d) >= 580
    check("RSNA-MICCAI competition", ok, f"train_labels.csv, {len(d)} patients", "add the competition input")
else:
    check("RSNA-MICCAI competition", False, "train_labels.csv not found",
          "add kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification")

# ---- 4. BraTS 2021 images -------------------------------------------------------------
tar = find(r"BraTS2021_Training_Data\.tar$")
segs = find(r"^BraTS2021_\d+_seg\.nii(\.gz)?$")
if tar:
    gb = os.path.getsize(tar[0]) / 1e9
    check("BraTS 2021 Task 1", gb > 5, f"tar found ({gb:.1f} GB)", "use dschettler8845/brats-2021-task1")
else:
    check("BraTS 2021 Task 1", len(segs) >= 1200, f"{len(segs)} patient folders with masks",
          "add kaggle.com/datasets/dschettler8845/brats-2021-task1")

# ---- 5. UPenn-GBM (NIfTI copy) ---------------------------------------------------------
up_img = find(r"^UPENN-GBM-\d+_11_FLAIR\.nii(\.gz)?$")
up_seg = find(r"^UPENN-GBM-\d+_11_.*segm\.nii(\.gz)?$")
up_dcm = [f for f in files if "UPENN-GBM" in f and f.lower().endswith(".dcm")]
if up_dcm and not up_img:
    check("UPenn-GBM", False, "WRONG COPY: DICOM files (.dcm) found, no NIfTI",
          "remove it; add kaggle.com/datasets/sanjanatg26/upenn-gbm")
else:
    subs = {re.match(r"(UPENN-GBM-\d+_11)", os.path.basename(f)).group(1) for f in up_img}
    check("UPenn-GBM (NIfTI)", len(subs) >= 550 and len(up_seg) >= 550,
          f"{len(subs)} patients with FLAIR, {len(up_seg)} tumour masks",
          "add kaggle.com/datasets/sanjanatg26/upenn-gbm")

# ---- 6. UCSF-PDGM --------------------------------------------------------------------
uc_seg = find(r"^UCSF-PDGM-\d+_tumor_segmentation\.nii(\.gz)?$")
uc_t1 = find(r"^UCSF-PDGM-\d+_T1\.nii(\.gz)?$")
uc_meta = find(r"UCSF-PDGM.*metadata.*\.csv$")
idx_ok = False
if uc_meta:
    m = pd.read_csv(uc_meta[0]); idx_ok = any("mgmt" in c.lower() and "index" in c.lower() for c in m.columns)
check("UCSF-PDGM", len(uc_seg) >= 450 and len(uc_t1) >= 450 and idx_ok,
      f"{len(uc_seg)} tumour masks, {len(uc_t1)} T1 scans, metadata "
      + ("with MGMT index" if idx_ok else "MISSING or no MGMT index"),
      "add kaggle.com/datasets/rukiyeaydn/ucsf-pdgm-filtered")

# ---- 7. your code dataset (mgmt-merged-code) -------------------------------------------
script = find(r"^kaggle_merged_training\.py$")
new_ver = bool(script) and 'SCRIPT_VERSION = "v9' in open(script[0]).read()
check("Training script", new_ver, "found, latest version" if new_ver else
      ("found but OLD version" if script else "not found"),
      "upload the 'Kaggle upload' folder as dataset mgmt-merged-code (New Version if it exists)")
figs = find(r"^paper_figures\.py$")
check("Figures script", bool(figs), "found" if figs else "not found",
      "upload the latest mgmt-merged-code.zip (it contains paper_figures.py)")
done = {os.path.basename(f) for f in files
        if f.endswith(".npz") and f.replace("\\", "/").split("/")[-3:-1] == ["outputs_merged", "jobs"]}
single = {d for d in done if d.startswith("ablsingle_")}
check("Proposed single-slice models (run 3)", len(single) >= 80,
      f"{len(single)} found (80 expected)",
      "add your dataset mgmt-merged-run3 (outputs_merged_run3.zip) as an input, "
      "otherwise the proposed model is retrained too (~1.5 h extra)")
ul = find(r"^upenn_mgmt_labels\.csv$")
check("UPenn MGMT labels", bool(ul) and len(pd.read_csv(ul[0])) == 291,
      f"{len(pd.read_csv(ul[0]))} labelled patients" if ul else "not found",
      "upload upenn_mgmt_labels.csv into mgmt-merged-code")
oof = find(r"^oof_predictions\.csv$")
check("Paper predictions (0.669)", bool(oof) and len(pd.read_csv(oof[0])) == 574,
      f"{len(pd.read_csv(oof[0]))} patients" if oof else "not found (comparison vs paper will be skipped)",
      "upload oof_predictions.csv into mgmt-merged-code")

# ---- report ---------------------------------------------------------------------------
w = max(len(r[0]) for r in results)
for name, ok, detail, fix in results:
    print(f"{'OK  ' if ok else 'FAIL'}  {name:<{w}}  {detail}" + ("" if ok else f"\n      -> fix: {fix}"))
bad = [r for r in results if not r[1]]
print("\n" + ("ALL GOOD - run kaggle_merged_training.py" if not bad
              else f"{len(bad)} problem(s) - fix them, then run this check again"))
