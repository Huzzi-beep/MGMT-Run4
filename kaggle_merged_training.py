#!/usr/bin/env python3
# =============================================================================
# kaggle_merged_training.py
# -----------------------------------------------------------------------------
# Multi-cohort training of the proposed tumour-focused 2.5D EfficientNet, using
# BOTH T4 GPUs at the same time.
#
# WHAT IT DOES
#   Design 1 (main result, like-for-like with the paper):
#     The 574 matched RSNA-MICCAI + BraTS 2021 patients keep EXACTLY the same
#     5-fold splits as the paper (StratifiedKFold, seeds 42 and 202). In every
#     fold, the training set is  BraTS train-fold  +  ALL UPenn-GBM / UCSF-PDGM
#     patients.  The held-out BraTS fold is scored as before, so the new AUC is
#     measured on the SAME 574 patients as the 0.669 headline and can be compared
#     with a DeLong test.
#   Control (RUN_CONTROL): EfficientNet-B0 on BraTS only, same code. Isolates the
#     effect of the extra data from any code differences.
#   Design 2 (RUN_EXTERNAL): leave-one-site-out. Train on BraTS + the other extra
#     site, test on the held-out site that no model has seen.
#
# HOW BOTH GPUs ARE USED
#   Every (backbone x modality x seed x fold) model is an independent job. Two
#   worker processes, one pinned to each T4, pull jobs from a shared queue, so
#   both GPUs train different models simultaneously (~2x throughput). For small
#   backbones like EfficientNet-B0/B2 this is faster and more stable than
#   splitting one model across GPUs (DataParallel/DDP).
#   Watch `!nvidia-smi` in another cell: both GPUs should show ~90%+ use.
#
# LEAK PROTECTION (UPenn and UCSF patients also appear in BraTS 2021)
#   * UCSF-PDGM metadata carries a BraTS21 ID column -> exact matches removed.
#   * Image-similarity audit of every extra patient vs the 574 evaluation
#     patients; anything above DEDUP_THRESHOLD is removed from training AND from
#     external testing. A false positive only costs one training case, so the
#     threshold is deliberately conservative. See dedup_report.csv.
#
# KAGGLE SETUP
#   Accelerator: GPU T4 x2.  Internet: ON (for ImageNet weights).
#   Inputs (Add Input):
#     1. RSNA-MICCAI Brain Tumor Radiogenomic Classification (competition)
#     2. BraTS 2021 Task 1 dataset (tar or extracted folders)
#     3. UPenn-GBM subset  -> <SID>_{FLAIR,T1,T1GD,T2,SEG/segm}.nii.gz
#                             + upenn_mgmt_labels.csv  (from upenn_1_fetch_labels.py)
#     4. UCSF-PDGM subset  -> UCSF-PDGM-xxxx_{FLAIR,T1,T1c,T2,tumor_segmentation}.nii.gz
#                             + UCSF-PDGM-metadata*.csv (from ucsf_subset_on_pc.py)
#     5. (optional) your earlier OOF files: oof_predictions.csv (0.669 headline),
#        oof_improve_*.csv, oof_ablation_nocrop.csv -> used for comparisons and
#        the combined ensemble.
#     6. (optional, to resume) the output of a previous run of this script.
#   Either extra site may be missing; the script uses whatever it finds.
#
# RUN
#   1) Set QUICK_TEST = True, run once (~15 min) to check inputs and GPUs.
#   2) Set QUICK_TEST = False, then "Save Version -> Save & Run All".
#      !python kaggle_merged_training.py
#   If the session stops early, attach this run's output as an input and run
#   again: finished jobs are skipped.
#
# OUTPUT  /kaggle/working/outputs_merged/
#   results_table.csv/.png     main table (AUC, 95% CI, dAUC, DeLong p, MCC, folds)
#   external_table.csv         leave-one-site-out results
#   oof_merged_*.csv           per-patient OOF predictions (for paper figures)
#   cohort_summary.json        patients per site, dedup counts
#   dedup_report.csv           similarity audit details
#   results.json               everything above in one file
#   jobs/*.npz                 per-model results (resume state)
# =============================================================================
import os, sys, re, glob, time, json, random, shutil, tarfile, fnmatch, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")   # no update check on import
try:
    sys.stdout.reconfigure(line_buffering=True)           # show log lines immediately
except Exception:
    pass

# ------------------------------- CONFIG --------------------------------------
SCRIPT_VERSION = "v9 (2026-10-11: single-slice input for every experiment)"
QUICK_TEST  = False      # True -> tiny run (2 epochs, 2 modalities, 1 seed) to check setup

MODALITIES  = ["FLAIR", "T1w", "T1wCE", "T2w"]
SUFFIX      = {"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"}
IMG_SIZE    = 224
N_SLICES    = 3
SLICE_GAP   = 4          # voxels at BraTS 1 mm spacing; rescaled for other spacings
CROP_MARGIN = 15
N_FOLDS     = 5
BATCH       = 32
LR          = 3e-4
WD          = 1e-5
BAD_IDS     = [109, 123, 709]
SEEDS       = [42, 202]  # same seeds = same folds as the paper
EPOCHS      = 20

BACKBONES         = ["efficientnet_b2", "efficientnet_b0"]   # longest first
EXTRA_SITES       = ["UPENN", "UCSF"]
EXTRA_GRADE4_ONLY = True       # UCSF-PDGM has grade 2-4; RSNA/BraTS is grade 4
# UCSF calls MGMT "positive" from index >= 1, but on the 245 UCSF patients that are
# also in RSNA-MICCAI, "index >= 2" reproduces the RSNA label exactly (100%).
# Relabel UCSF with the RSNA definition so both cohorts mean the same thing.
UCSF_MGMT_INDEX_CUTOFF = 2     # None -> use UCSF's own positive/negative status
RUN_CONTROL       = True       # B0 on BraTS only with this exact code
RUN_EXTERNAL      = True       # leave-one-site-out (needs >= 1 extra site)
EXTERNAL_BACKBONE = "efficientnet_b0"
# Standard baselines, trained under the IDENTICAL merged protocol as the proposed
# model (same 574 evaluation patients and folds, same extra cohorts, same 2.5D
# tumour inputs, epochs and augmentation). Report every baseline that is run.
RUN_BASELINES      = True
# Reported baselines = the architectures requested by the publishing author
# (supervisor's instruction, 2026-10-10). VGG-16 and MobileNet-v3-Large were also
# trained in run 2 (jobs base_vgg16_bn_* / base_mobilenetv3_large_100_* in
# outputs_merged_run2.zip) but are not part of the requested comparison set.
BASELINE_BACKBONES = ["densenet121", "resnet18", "mobilenetv2_100", "simple_cnn"]   # longest first
BASELINE_LABELS    = {"simple_cnn": "Simple CNN (4-layer, from scratch)", "vgg16_bn": "VGG-16",
                      "resnet18": "ResNet-18", "densenet121": "DenseNet-121",
                      "mobilenetv2_100": "MobileNet-v2", "mobilenetv3_large_100": "MobileNet-v3-Large"}
# Ablation of the proposed B0+B2 ensemble under the same merged protocol. Each
# variant removes ONE component; "−multi-cohort", "−ensemble", "−skill weighting"
# and "−multimodal fusion" are computed from existing runs without training.
RUN_CONTROL_B2   = True        # BraTS-only B2 (for the "−multi-cohort" ablation row)
RUN_ABLATION     = True
ABLATIONS        = ["center", "nocrop", "single", "randinit"]
ABLATION_LABELS  = {"randinit": "− ImageNet pretraining (random init)",
                    "single":   "− 2.5D context (single slice)",
                    "nocrop":   "− Tumour-focus crop (whole slice, tumour slice)",
                    "center":   "− Tumour guidance (centre slice, whole image)"}
CACHE_VARIANTS   = {"nocrop": dict(use_crop=False, slice_mode="tumor"),
                    "center": dict(use_crop=False, slice_mode="center")}
# Grad-CAM: a few B2 models are retrained with their weights saved, then
# heat-maps are drawn on held-out fold-1 patients.
RUN_GRADCAM      = True
GRADCAM_BACKBONE = "efficientnet_b2"
GRADCAM_MODS     = ["FLAIR", "T1wCE"]
DEDUP_THRESHOLD   = 0.95       # cosine similarity of FLAIR+T2 tumour crops
TIME_BUDGET_H     = 11.0       # stop taking new jobs after this (Kaggle cap is 12 h)
# 0 = images are augmented inside each GPU process. Extra loader processes are
# forked from a CUDA process with live threads, which can freeze silently on
# Python 3.13 -- the augmentation is cheap (~1 s/epoch), so they are not worth it.
LOADER_WORKERS    = 0
STALL_MINUTES     = 45         # watchdog: stop workers if no model finishes for this long
PROGRESS_MINUTES  = 5          # print a progress line this often during training

# Environment overrides (used only for local testing; leave alone on Kaggle)
ON_KAGGLE  = os.path.isdir("/kaggle")
INPUT_ROOT = os.environ.get("MERGED_INPUT_ROOT", "/kaggle/input" if ON_KAGGLE else ".")
WORK       = os.environ.get("MERGED_WORK", "/kaggle/working" if ON_KAGGLE else "./merged_work")
DEVICE     = os.environ.get("MERGED_DEVICE", "cuda")
PRETRAINED = os.environ.get("MERGED_PRETRAINED", "1") == "1"
LOADER_WORKERS = int(os.environ.get("MERGED_LOADER_WORKERS", LOADER_WORKERS))
if os.environ.get("MERGED_QUICK") == "1":
    QUICK_TEST = True

if QUICK_TEST:
    EPOCHS, MODALITIES, SEEDS = 2, ["FLAIR", "T2w"], [42]
    BACKBONES, RUN_CONTROL = ["efficientnet_b0"], False
    BASELINE_BACKBONES = ["resnet18", "simple_cnn"]
    ABLATIONS, RUN_CONTROL_B2 = ["single", "randinit"], False
    TIME_BUDGET_H = 3.0   # image preparation alone can take ~45 min on a fresh session

# Input configuration of the proposed model: the largest-tumour slice only (one
# slice, repeated to the 3 input channels). Every experiment -- proposed model,
# baselines, controls, external validation, ablation, Grad-CAM -- uses it.
# Job names carry an "SS" suffix; the proposed ensemble reuses the run-3 models
# trained with this input ("ablsingle_*" jobs).
SINGLE_SLICE = True
if SINGLE_SLICE:
    ABLATIONS = [a for a in ABLATIONS if a != "single"]
    EXTERNAL_BACKBONES = list(BACKBONES)        # external test of the full B0 + B2 ensemble
else:
    EXTERNAL_BACKBONES = [EXTERNAL_BACKBONE]

OUT_DIR    = os.path.join(WORK, "outputs_merged" + ("_quick" if QUICK_TEST else ""))
JOB_DIR    = os.path.join(OUT_DIR, "jobs")
WEIGHT_DIR = os.path.join(OUT_DIR, "cam_weights")
CACHE_ROOT = os.path.join(WORK, "cache_merged")
TMP        = os.path.join(WORK, "tmp_extract") if not ON_KAGGLE else "/kaggle/temp/brats"


def seed_all(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def short(backbone):
    return backbone.replace("efficientnet_", "")


def make_model(name, pretrained):
    """timm backbone with a 1-logit head, or the basic from-scratch CNN baseline."""
    import torch.nn as nn, timm
    if name == "simple_cnn":
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))
    return timm.create_model(name, pretrained=pretrained, num_classes=1, in_chans=3)


def job_name(kind, backbone, mod, seed, fold):
    if SINGLE_SLICE:
        # the proposed models already exist from run 3 under "ablsingle"
        kind = "ablsingle" if kind == "cv" else f"{kind}SS"
    return f"{kind}_{short(backbone)}_{mod}_s{seed}_f{fold}_e{EPOCHS}"


# ------------------------------- discovery -----------------------------------
_ALL_FILES = None


# RSNA DICOM series folders (~350,000 .dcm files). Only train_labels.csv is used
# from RSNA, so these folders are skipped -- walking them alone takes minutes.
SKIP_DIRS = {"FLAIR", "T1w", "T1wCE", "T2w"}


def all_input_files():
    """One walk over the inputs, reused by every search."""
    global _ALL_FILES
    if _ALL_FILES is None:
        t0 = time.time(); _ALL_FILES = []
        print("scanning inputs ...")
        for root, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and not os.path.abspath(os.path.join(root, d)).startswith(os.path.abspath(WORK))]
            _ALL_FILES += [os.path.join(root, f) for f in files]
        print(f"scanned {len(_ALL_FILES):,} input files in {time.time()-t0:.0f}s")
    return _ALL_FILES


def find_files(pattern):
    return sorted(f for f in all_input_files() if fnmatch.fnmatch(os.path.basename(f), pattern))


def list_inputs():
    print("Inputs under", INPUT_ROOT)
    for d in sorted(glob.glob(os.path.join(INPUT_ROOT, "*"))):
        print("   ", d)


def _sid(name):
    m = re.search(r"BraTS2021_(\d+)", name); return int(m.group(1)) if m else None


def _brats_key(name):
    n = name.lower()
    for k in ["_seg.nii.gz", "_flair.nii.gz", "_t1ce.nii.gz", "_t1.nii.gz", "_t2.nii.gz"]:
        if n.endswith(k): return k.strip("_").replace(".nii.gz", "")
    return None


def find_brats_source():
    tar = find_files("BraTS2021_Training_Data.tar") or find_files("*Training*Data*.tar")
    if not tar:
        tar = [t for t in find_files("*.tar") if "brats" in t.lower() or "training" in t.lower()]
    if tar:
        extra = find_files("BraTS2021_00*.tar")
        print("BraTS source: TAR ->", tar[0], f"(+{len(extra)} shards)")
        return ("tar", tar[0], extra)
    dirs = {}
    for s in find_files("BraTS2021_*_seg.nii.gz"):
        sid = _sid(os.path.basename(s))
        if sid is not None: dirs.setdefault(sid, os.path.dirname(s))
    if dirs:
        print(f"BraTS source: FOLDERS -> {len(dirs)} patients")
        return ("dirs", dirs, [])
    return (None, None, [])


# ------------------------------- image cache ---------------------------------
def robust_norm(x):
    x = x.astype(np.float32); nz = x[x > 0]
    if nz.size == 0: return np.zeros_like(x)
    lo, hi = np.percentile(nz, [1, 99]); x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)


def _reorient(img, axcodes):
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    cur, tgt = io_orientation(img.affine), axcodes2ornt(axcodes)
    if np.array_equal(cur, tgt): return img
    return img.as_reoriented(ornt_transform(cur, tgt))


def build_stack(files, out_dir, name, reorient_to=None, use_crop=True, slice_mode="tumor"):
    """Tumour-focused 2.5D stack per modality -- the paper's _build_one.
    files: {"seg","flair","t1","t1ce","t2"} -> path.
    reorient_to=None keeps BraTS volumes byte-identical to the paper; extra
    cohorts are reoriented to the BraTS axis codes and their slice gap is
    rescaled to the same physical distance. Returns tumour voxel count or None.
    Ablation variants: use_crop=False keeps the whole slice; slice_mode="center"
    takes the geometric centre slice instead of the largest-tumour slice."""
    import nibabel as nib, cv2
    need = ["seg"] + [SUFFIX[m] for m in MODALITIES]
    if any(k not in files for k in need): return None
    try:
        segi = nib.load(files["seg"])
        if reorient_to is not None: segi = _reorient(segi, reorient_to)
        tumor = np.asarray(segi.get_fdata()) > 0
        if tumor.ndim != 3 or tumor.sum() == 0: return None
        gap = SLICE_GAP
        if reorient_to is not None:
            gap = max(1, int(round(SLICE_GAP * 1.0 / float(segi.header.get_zooms()[2]))))
        z_tumor = int(np.argmax(tumor.sum(axis=(0, 1))))
        ys, xs = np.where(tumor.any(axis=2))
        y0, y1 = max(0, ys.min() - CROP_MARGIN), ys.max() + CROP_MARGIN
        x0, x1 = max(0, xs.min() - CROP_MARGIN), xs.max() + CROP_MARGIN
        off = [(i - N_SLICES // 2) * gap for i in range(N_SLICES)]
        stacks = {}
        for mod in MODALITIES:
            img = nib.load(files[SUFFIX[mod]])
            if reorient_to is not None: img = _reorient(img, reorient_to)
            vol = np.asarray(img.get_fdata())
            if vol.ndim == 4: vol = vol[..., 0]
            if vol.shape != tumor.shape: return None
            d = vol.shape[2]; chans = []
            base_z = d // 2 if slice_mode == "center" else z_tumor
            for o in off:
                zi = min(max(0, base_z + o), d - 1)
                sl = vol[y0:y1, x0:x1, zi] if use_crop else vol[:, :, zi]
                chans.append(cv2.resize(robust_norm(sl), (IMG_SIZE, IMG_SIZE)))
            stacks[mod] = (np.stack(chans, -1) * 255).astype(np.uint8)
        for mod, arr in stacks.items():
            np.save(os.path.join(out_dir, f"{mod}_{name}.npy"), arr)
        return int(tumor.sum())
    except Exception as e:
        print(f"   skip {name}: {type(e).__name__}: {e}")
        return None


def brats_axcodes(source):
    """Axis codes of the BraTS volumes; extra cohorts are aligned to these."""
    import nibabel as nib
    kind, data, _ = source
    try:
        if kind == "dirs":
            folder = next(iter(data.values()))
            seg = glob.glob(os.path.join(folder, "*_seg.nii.gz"))[0]
            return tuple(nib.aff2axcodes(nib.load(seg).affine))
        if kind == "tar":
            os.makedirs(TMP, exist_ok=True)
            with tarfile.open(data) as tf:
                for m in tf:
                    if m.isfile() and m.name.lower().endswith("_seg.nii.gz"):
                        dst = os.path.join(TMP, "axref_seg.nii.gz")
                        with tf.extractfile(m) as src, open(dst, "wb") as out:
                            shutil.copyfileobj(src, out)
                        codes = tuple(nib.aff2axcodes(nib.load(dst).affine))
                        os.remove(dst); return codes
    except Exception as e:
        print("   could not read BraTS orientation:", e)
    print("   WARNING: assuming BraTS orientation ('L','P','S')")
    return ("L", "P", "S")


def reuse_cache_from_inputs(dst, names):
    """Copy a cache folder from a previous run's output (attached as input)."""
    if os.path.isfile(os.path.join(dst, "done.flag")): return
    for flag in find_files("done.flag"):
        if os.path.basename(os.path.dirname(flag)) in names:
            print(f"reusing cache from input: {os.path.dirname(flag)}")
            shutil.copytree(os.path.dirname(flag), dst, dirs_exist_ok=True); return


def build_brats_cache(needed, source, variant="main"):
    out = os.path.join(CACHE_ROOT, "brats" if variant == "main" else f"brats_{variant}")
    if variant == "main":
        reuse_cache_from_inputs(out, {"brats", "imgcache", "imgcache_brats"})
    opts = CACHE_VARIANTS.get(variant, {})
    os.makedirs(out, exist_ok=True)
    flag = os.path.join(out, "done.flag")
    if os.path.isfile(flag):
        cov = set(int(x) for x in open(flag).read().split())
        print(f"reusing BraTS {variant} cache ({len(cov)} patients)"); return out, cov
    t0 = time.time(); covered = set(); kind, data, extra = source
    print(f"preparing BraTS images ({variant}) for {len(needed)} patients (progress every 50) ...")

    def add(sid):
        covered.add(sid)
        if len(covered) % 50 == 0:
            print(f"   BraTS: {len(covered)}/{len(needed)} patients ({(time.time()-t0)/60:.1f} min)")

    if kind == "tar":
        os.makedirs(TMP, exist_ok=True)
        for path in [data, *extra]:
            cur_id, cur = None, {}
            with tarfile.open(path) as tf:
                for m in tf:
                    if not m.isfile(): continue
                    sid = _sid(m.name)
                    if sid not in needed: continue
                    k = _brats_key(m.name)
                    if k is None: continue
                    if sid != cur_id and cur_id is not None:
                        if build_stack(cur, out, cur_id, **opts) is not None: add(cur_id)
                        for f in cur.values():
                            try: os.remove(f)
                            except OSError: pass
                        cur = {}
                    cur_id = sid
                    dst = os.path.join(TMP, os.path.basename(m.name))
                    with tf.extractfile(m) as src, open(dst, "wb") as fo:
                        shutil.copyfileobj(src, fo)
                    cur[k] = dst
                if cur_id is not None and build_stack(cur, out, cur_id, **opts) is not None:
                    add(cur_id)
    elif kind == "dirs":
        for sid, folder in data.items():
            if sid not in needed: continue
            files = {}
            for f in glob.glob(os.path.join(folder, "*.nii.gz")):
                k = _brats_key(f)
                if k: files[k] = f
            if build_stack(files, out, sid, **opts) is not None: add(sid)
    if not covered:
        sys.exit("No BraTS images could be cached -- check the BraTS 2021 input.")
    open(flag, "w").write(" ".join(map(str, sorted(covered))))
    print(f"built BraTS cache: {len(covered)}/{len(needed)} patients in {time.time()-t0:.0f}s")
    return out, covered


# ---- extra cohorts -------------------------------------------------------------
def parse_mgmt(v):
    s = str(v).strip().lower()
    if s in ("", "nan", "none"): return None
    if s.startswith("unmethyl") or s in ("negative", "neg", "0", "0.0", "no"): return 0
    if s.startswith("methyl") or s in ("positive", "pos", "1", "1.0", "yes"): return 1
    return None                   # indeterminate / not available


def pick_col(df, *needles):
    for c in df.columns:
        if all(n in c.lower() for n in needles): return c
    return None


def _digits(v):
    m = re.search(r"(\d+)\D*$", str(v)); return int(m.group(1)) if m else None


def site_key(site, sid):
    return _digits(sid) if site == "UCSF" else str(sid).strip().upper()


def discover_site_files(site):
    """{subject_id: {seg, flair, t1, t1ce, t2}} for one extra cohort."""
    subs = {}
    if site == "UPENN":
        rx = re.compile(r"^(UPENN-GBM-\d+_\d+)_(.+)\.nii(?:\.gz)?$", re.I)
        for f in all_input_files():
            m = rx.match(os.path.basename(f))
            if not m: continue
            sid, tag = m.group(1).upper(), m.group(2).lower()
            d = subs.setdefault(sid, {})
            if "seg" in tag:
                # prefer the manual segmentation over the automated one
                if "automated" not in tag or "seg" not in d: d["seg"] = f
            else:
                key = {"flair": "flair", "t1": "t1", "t1gd": "t1ce", "t1ce": "t1ce", "t2": "t2"}.get(tag)
                if key: d[key] = f
    elif site == "UCSF":
        rx = re.compile(r"^(UCSF-PDGM-\d+)_(.+)\.nii(?:\.gz)?$", re.I)
        for f in all_input_files():
            m = rx.match(os.path.basename(f))
            if not m: continue
            sid, tag = m.group(1).upper(), m.group(2).lower()
            d = subs.setdefault(sid, {})
            if tag == "tumor_segmentation":
                d["seg"] = f; continue
            bias = tag.endswith("_bias")
            key = {"flair": "flair", "t1": "t1", "t1c": "t1ce", "t2": "t2"}.get(tag.replace("_bias", ""))
            if key and (not bias or key not in d):   # prefer the non-bias-corrected file
                d[key] = f
    return subs


def load_site_labels(site):
    """DataFrame[key, subject_id, MGMT_value, brats21_id]."""
    rows = []
    if site == "UPENN":
        lab = find_files("upenn_mgmt_labels.csv")
        if lab:
            d = pd.read_csv(lab[0])
            for _, r in d.iterrows():
                y = parse_mgmt(r["MGMT_value"])
                if y is not None: rows.append((site_key(site, r["subject_id"]), r["subject_id"], y, None))
        else:
            clin = find_files("UPENN-GBM*clinical*.csv")
            if clin:
                d = pd.read_csv(clin[0]); idc = pick_col(d, "id"); mc = pick_col(d, "mgmt")
                for _, r in d.iterrows():
                    y = parse_mgmt(r[mc])
                    if y is not None: rows.append((site_key(site, r[idc]), r[idc], y, None))
    elif site == "UCSF":
        meta = find_files("UCSF-PDGM*metadata*.csv") or find_files("*PDGM*meta*.csv")
        if meta:
            d = pd.read_csv(meta[0])
            idc = pick_col(d, "id") if "ID" not in d.columns else "ID"
            mc = pick_col(d, "mgmt", "status") or pick_col(d, "mgmt")
            ic = pick_col(d, "mgmt", "index")
            gc = pick_col(d, "grade")
            bc = pick_col(d, "brats", "id") or pick_col(d, "brats")
            for _, r in d.iterrows():
                y = parse_mgmt(r[mc])
                if y is None: continue
                if UCSF_MGMT_INDEX_CUTOFF is not None and ic is not None:
                    v = pd.to_numeric(r[ic], errors="coerce")
                    if pd.notna(v): y = int(v >= UCSF_MGMT_INDEX_CUTOFF)
                if EXTRA_GRADE4_ONLY and gc is not None and _digits(r[gc]) not in (4, None):
                    continue
                b = _digits(r[bc]) if bc is not None and pd.notna(r[bc]) else None
                rows.append((site_key(site, r[idc]), r[idc], y, b))
    return pd.DataFrame(rows, columns=["key", "subject_id", "MGMT_value", "brats21_id"])


def build_site_cache(site, axcodes):
    out = os.path.join(CACHE_ROOT, site.lower())
    reuse_cache_from_inputs(out, {site.lower()})
    meta_p = os.path.join(out, "meta.csv")
    if os.path.isfile(os.path.join(out, "done.flag")) and os.path.isfile(meta_p):
        meta = pd.read_csv(meta_p)
        print(f"reusing {site} cache ({len(meta)} patients)"); return out, meta
    labels = load_site_labels(site)
    if labels.empty:
        print(f"{site}: no MGMT labels found -> site skipped"); return None, None
    files = discover_site_files(site)
    fkey = {site_key(site, s): (s, f) for s, f in files.items()}
    jobs, rows = [], []
    for i, r in enumerate(labels.drop_duplicates("key").itertuples()):
        if r.key in fkey:
            sid, f = fkey[r.key]
            jobs.append((f, out, str(i), axcodes))
            rows.append({"idx": i, "subject_id": sid, "MGMT_value": int(r.MGMT_value),
                         "brats21_id": r.brats21_id})
    print(f"{site}: {len(labels)} labelled, {len(files)} on disk, {len(jobs)} matched")
    if not jobs:
        print(f"{site}: no labelled subject has images -> site skipped"); return None, None
    os.makedirs(out, exist_ok=True)
    # plain loop (a forked process pool can freeze once OpenCV threads exist)
    t0 = time.time(); res = {}
    for k, (f, o, name, ax) in enumerate(jobs, 1):
        res[name] = build_stack(f, o, name, reorient_to=ax)
        if k % 50 == 0 or k == len(jobs):
            print(f"   {site}: {k}/{len(jobs)} patients prepared ({(time.time()-t0)/60:.1f} min)")
    meta = pd.DataFrame([dict(r, tumor_vox=res[str(r["idx"])]) for r in rows
                         if res.get(str(r["idx"])) is not None])
    if meta.empty:
        print(f"{site}: no subject could be cached -> site skipped"); return None, None
    meta["site"] = site
    meta.to_csv(meta_p, index=False)
    open(os.path.join(out, "done.flag"), "w").write(str(len(meta)))
    print(f"{site}: cached {len(meta)} patients (pos rate {meta.MGMT_value.mean():.3f}) "
          f"in {time.time()-t0:.0f}s")
    return out, meta


def build_site_variant_cache(site, axcodes, variant, main_meta):
    """Ablation copy of an extra cohort's cache (whole slice / centre slice) for
    exactly the patients of the main cache, with the same patient numbering."""
    out = os.path.join(CACHE_ROOT, f"{site.lower()}_{variant}")
    flag = os.path.join(out, "done.flag")
    if os.path.isfile(flag):
        print(f"reusing {site} {variant} cache"); return out
    labels, files = load_site_labels(site), discover_site_files(site)
    fkey = {site_key(site, s): f for s, f in files.items()}
    want = set(int(i) for i in main_meta.idx)
    os.makedirs(out, exist_ok=True); t0 = time.time(); k = 0
    for i, r in enumerate(labels.drop_duplicates("key").itertuples()):
        if i not in want or r.key not in fkey: continue
        build_stack(fkey[r.key], out, str(i), reorient_to=axcodes, **CACHE_VARIANTS[variant]); k += 1
        if k % 50 == 0 or k == len(want):
            print(f"   {site} ({variant}): {k}/{len(want)} patients prepared ({(time.time()-t0)/60:.1f} min)")
    open(flag, "w").write(str(k))
    return out


def cache_complete(cache_dir, names):
    """True if every patient has all modality files in this cache."""
    return all(os.path.isfile(os.path.join(cache_dir, f"{m}_{n}.npy")) for n in names for m in MODALITIES)


# ---- duplicate audit ---------------------------------------------------------
def dedup_audit(site, meta, site_cache, brats_cache, eval_ids, eval_labels):
    import cv2
    def sig(cache, name):
        parts = []
        for mod in ("FLAIR", "T2w"):
            p = os.path.join(cache, f"{mod}_{name}.npy")
            if not os.path.isfile(p): continue
            a = cv2.resize(np.load(p).astype(np.float32).mean(-1), (32, 32))
            parts.append((a - a.mean()) / (a.std() + 1e-6))
        return np.concatenate([x.ravel() for x in parts]) if parts else None
    bsig = {i: sig(brats_cache, i) for i in eval_ids}
    bids = [i for i, v in bsig.items() if v is not None]
    B = np.stack([bsig[i] for i in bids]); B /= np.linalg.norm(B, axis=1, keepdims=True)
    eval_set = set(int(i) for i in eval_ids)
    rows = []
    for r in meta.itertuples():
        s = sig(site_cache, r.idx)
        sims = B @ (s / np.linalg.norm(s)) if s is not None else np.zeros(len(bids))
        j = int(np.argmax(sims))
        exact = pd.notna(r.brats21_id) and int(r.brats21_id) in eval_set
        matched = str(int(r.brats21_id)) if exact else bids[j]
        rows.append({"site": site, "idx": r.idx, "subject_id": r.subject_id,
                     "nearest_brats": bids[j], "max_cosine": float(sims[j]),
                     "brats21_id_match": bool(exact),
                     "image_match": bool(sims[j] > DEDUP_THRESHOLD),
                     "excluded": bool(exact or sims[j] > DEDUP_THRESHOLD),
                     "site_label": int(r.MGMT_value), "rsna_label": eval_labels.get(matched)})
    rep = pd.DataFrame(rows)
    print(f"{site} dedup: {int(rep.brats21_id_match.sum())} exact BraTS21-ID matches, "
          f"{int((rep.max_cosine > DEDUP_THRESHOLD).sum())} image matches > {DEDUP_THRESHOLD} "
          f"-> {int(rep.excluded.sum())} excluded | similarity quantiles 50/90/99% = "
          f"{np.percentile(rep.max_cosine, [50, 90, 99]).round(3).tolist()}")
    ex = rep[rep.brats21_id_match]
    if len(ex):
        print(f"   image check found {int(ex.image_match.sum())}/{len(ex)} of the known (ID) duplicates")
    dup = rep[rep.excluded]
    if len(dup):
        print(f"   label agreement with RSNA on duplicates: "
              f"{(dup.site_label == dup.rsna_label).mean():.3f} (n={len(dup)})")
    return rep


# ------------------------------- GPU worker ----------------------------------
class TrainDS:
    """Training dataset (module level so DataLoader worker processes can load it
    under any multiprocessing start method -- Python 3.13+ on Kaggle)."""
    def __init__(self, arrs, labels, tf): self.arrs, self.labels, self.tf = arrs, labels, tf
    def __len__(self): return len(self.arrs)
    def __getitem__(self, i):
        import torch
        return self.tf(image=self.arrs[i])["image"], torch.tensor(self.labels[i], dtype=torch.float32)


def worker(rank, n_workers, jobs, cfg):
    """Runs in its own process on one GPU (CUDA_VISIBLE_DEVICES = that GPU).
    Both workers walk the same job list; a lock file decides who trains what."""
    import torch, torch.nn as nn, timm, cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cv2.setNumThreads(1)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // n_workers - LOADER_WORKERS))
    if DEVICE == "cuda":
        torch.cuda.set_device(0); dev = torch.device("cuda", 0)   # only this GPU is visible
        tag = f"[GPU{rank} {torch.cuda.get_device_name(0)}]"
    else:
        dev = torch.device(DEVICE); tag = f"[worker{rank} {DEVICE}]"
    use_amp = dev.type == "cuda"
    print(f"{tag} started", flush=True)

    tr_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(0.10, 0.10, 20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.Normalize((0.5,) * 3, (0.5,) * 3), ToTensorV2()])

    store, store_variant = {}, [None]
    def load(site, name, mod, variant="main"):
        # variant: "main" | ablation name. "nocrop"/"center" read their own cache;
        # single-slice input keeps only the middle slice (repeated to 3 channels).
        if variant != store_variant[0]:
            store.clear(); store_variant[0] = variant      # bound memory per worker
        cache = variant if variant in cfg["caches"] else "main"
        k = (site, name, mod)
        if k not in store:
            a = np.load(os.path.join(cfg["caches"][cache][site], f"{mod}_{name}.npy"))
            if SINGLE_SLICE or variant == "single":
                a = np.ascontiguousarray(np.repeat(a[:, :, 1:2], 3, axis=2))
            store[k] = a
        return store[k]

    def to_tensor(arrs):
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float()
        return (x / 255.0 - 0.5) / 0.5

    @torch.no_grad()
    def predict(model, X, tta):
        model.eval(); out = []
        for i in range(0, len(X), 64):
            xb = X[i:i + 64].to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                p = torch.sigmoid(model(xb).squeeze(1).float())
                if tta:
                    p = (p + torch.sigmoid(model(torch.flip(xb, [3])).squeeze(1).float())) / 2
            out.append(torch.nan_to_num(p, nan=0.5).cpu())
        return torch.cat(out).numpy()

    def safe_auc(y, p):
        ok = np.isfinite(p)
        if ok.sum() < 2 or len(np.unique(y[ok])) < 2: return 0.5
        return roc_auc_score(y[ok], p[ok])

    b_ids = np.array(cfg["brats_ids"]); b_y = np.array(cfg["brats_y"])
    splits = {s: list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(b_ids, b_y))
              for s in cfg["seeds"]}

    claim_dir = os.path.join(JOB_DIR, "claims"); total = len(jobs)
    for job in jobs:
        if os.path.isfile(os.path.join(JOB_DIR, job["name"] + ".npz")): continue
        try:
            os.close(os.open(os.path.join(claim_dir, job["name"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                      # the other GPU took this job
        if time.time() > cfg["deadline"]:
            print(f"{tag} time budget reached -> skipping {job['name']}", flush=True); continue
        t0 = time.time()
        try:
            mod, seed, fold = job["modality"], job["seed"], job["fold"]
            tr, va = splits[seed][fold]
            tr_keys = [("BRATS", str(i)) for i in b_ids[tr]]; tr_y = list(b_y[tr])
            for site in job["train_sites"]:
                tr_keys += [(site, str(i)) for i in cfg["sites"][site]["ids"]]
                tr_y += list(cfg["sites"][site]["y"])
            variant = job.get("variant", "main")
            tr_arrs = [load(s, n, mod, variant) for s, n in tr_keys]
            Xva = to_tensor([load("BRATS", str(i), mod, variant) for i in b_ids[va]]); yva = b_y[va]
            Xte = None
            if job.get("test_site"):
                ts = cfg["sites"][job["test_site"]]
                Xte = to_tensor([load(job["test_site"], str(i), mod, variant) for i in ts["ids"]])

            print(f"{tag} start {job['name']} ({len(tr_arrs)} training images)", flush=True)
            seed_all(seed + fold)
            # num_workers=0: no loader processes (see LOADER_WORKERS); a timeout
            # turns any loader stall into an error instead of a silent hang
            dl = DataLoader(TrainDS(tr_arrs, np.array(tr_y, np.float32), tr_tf), batch_size=BATCH,
                            shuffle=True, drop_last=True, num_workers=LOADER_WORKERS,
                            persistent_workers=LOADER_WORKERS > 0,
                            timeout=600 if LOADER_WORKERS > 0 else 0)
            model = make_model(job["backbone"], PRETRAINED and variant != "randinit").to(dev)
            optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=job["epochs"])
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            crit = nn.BCEWithLogitsLoss()
            best, best_ep, best_state = -1.0, -1, None
            for ep in range(job["epochs"]):
                model.train()
                for xb, yb in dl:
                    xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                        loss = crit(model(xb).squeeze(1), yb)
                    if not torch.isfinite(loss):
                        optim.zero_grad(set_to_none=True); continue
                    optim.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                    scaler.unscale_(optim); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim); scaler.update()
                # paper protocol: epoch chosen on the held-out BraTS fold (no TTA)
                va_auc = safe_auc(yva, predict(model, Xva, tta=False)); sched.step()
                if va_auc > best:
                    best, best_ep = va_auc, ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            res = {"va_idx": va, "last_va": predict(model, Xva, tta=True)}
            if Xte is not None: res["last_te"] = predict(model, Xte, tta=True)
            if best_state is not None: model.load_state_dict(best_state)
            res["best_va"] = predict(model, Xva, tta=True)
            if Xte is not None: res["best_te"] = predict(model, Xte, tta=True)
            res["best_epoch"] = best_ep; res["best_sel_auc"] = best
            if job.get("save_weights") and best_state is not None:     # for Grad-CAM
                os.makedirs(WEIGHT_DIR, exist_ok=True)
                torch.save(best_state, os.path.join(WEIGHT_DIR, job["name"] + ".pt"))
            tmp = os.path.join(JOB_DIR, job["name"] + ".tmp.npz")
            np.savez(tmp, **res); os.replace(tmp, os.path.join(JOB_DIR, job["name"] + ".npz"))
            del model, optim, dl
            if DEVICE == "cuda": torch.cuda.empty_cache()
            done = len(glob.glob(os.path.join(JOB_DIR, "*.npz"))) - cfg["done_before"]
            print(f"{tag} {done:3d}/{total}  {job['name']:<38s} sel-AUC {best:.3f} @ep{best_ep+1:<2d} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception:
            print(f"{tag} FAILED {job['name']}\n{traceback.format_exc()}", flush=True)
    print(f"{tag} finished", flush=True)


def _script_path():
    """Path of this script, so each GPU can run it as its own process. When the
    code was pasted into a notebook cell, the cell source is written to a file."""
    p = globals().get("__file__")
    if p and os.path.isfile(p) and p.endswith(".py"):
        return os.path.abspath(p)
    import linecache
    src = "".join(linecache.getlines(worker.__code__.co_filename))
    # a cell that also holds !shell lines cannot run as a plain script
    if "def worker(" not in src or re.search(r"^\s*get_ipython\(\)", src, re.M):
        sys.exit("Could not find the script file. Run it as:\n"
                 "  !python /kaggle/working/kaggle_merged_training.py")
    p = os.path.join(WORK, "_merged_runtime.py")
    with open(p, "w") as fh: fh.write(src)
    return p


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_jobs(jobs, cfg):
    import subprocess, threading
    os.makedirs(JOB_DIR, exist_ok=True)
    pending = [j for j in jobs if not os.path.isfile(os.path.join(JOB_DIR, j["name"] + ".npz"))]
    print(f"\n{len(jobs)} models in total, {len(jobs)-len(pending)} already done, "
          f"{len(pending)} to train")
    if not pending: return
    if DEVICE == "cuda":
        import torch; n_workers = torch.cuda.device_count()
    else:
        n_workers = 2
    print(f"training on {n_workers} device(s) in parallel\n", flush=True)
    claim_dir = os.path.join(JOB_DIR, "claims")
    shutil.rmtree(claim_dir, ignore_errors=True); os.makedirs(claim_dir)
    cfg = dict(cfg, done_before=len(glob.glob(os.path.join(JOB_DIR, "*.npz"))))
    spec = os.path.join(JOB_DIR, "worker_spec.json")
    with open(spec, "w") as fh:
        json.dump({"jobs": pending, "cfg": cfg, "n_workers": n_workers}, fh, default=_json_default)
    script = _script_path()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else [str(i) for i in range(n_workers)]
    procs, relays = [], []
    for r in range(n_workers):
        # weights were downloaded by main(); workers read the local copy (no network calls)
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
        if DEVICE == "cuda": env["CUDA_VISIBLE_DEVICES"] = gpu_ids[r]
        p = subprocess.Popen([sys.executable, "-u", script, "--worker", str(r), spec], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=lambda p=p: [print(l, end="", flush=True) for l in p.stdout])
        t.start(); procs.append(p); relays.append(t)

    # progress line every PROGRESS_MINUTES; stop the workers if nothing finishes
    # for STALL_MINUTES, so a frozen run still writes its results and ends
    t0 = time.time(); done_before = cfg["done_before"]
    finished = lambda: len(glob.glob(os.path.join(JOB_DIR, "*.npz"))) - done_before
    last_n, last_change, last_print = 0, time.time(), time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(15)
        n = finished()
        if n != last_n: last_n, last_change = n, time.time()
        if time.time() - last_print >= PROGRESS_MINUTES * 60:
            last_print = time.time(); el = (time.time() - t0) / 3600
            eta = f", about {el / n * (len(pending) - n):.1f} h left" if n else ""
            running = [os.path.basename(c) for c in glob.glob(os.path.join(claim_dir, "*"))
                       if not os.path.isfile(os.path.join(JOB_DIR, os.path.basename(c) + ".npz"))]
            print(f"[progress] {n}/{len(pending)} models finished | {el:.2f} h elapsed{eta} "
                  f"| training now: {', '.join(running) or '-'}", flush=True)
        if time.time() - last_change > STALL_MINUTES * 60:
            print(f"\nSTALL: no model finished in {STALL_MINUTES} min -- stopping the GPU workers. "
                  f"Finished models are saved; send this log to Claude.", flush=True)
            for p in procs:
                if p.poll() is None: p.kill()
            break
    for p in procs: p.wait()
    for t in relays: t.join(timeout=30)
    for r, p in enumerate(procs):
        if p.returncode != 0:
            print(f"WARNING: GPU worker {r} exited with code {p.returncode} (see its messages above)")


def _worker_cli(rank, spec_path):
    with open(spec_path) as fh: spec = json.load(fh)
    worker(rank, spec["n_workers"], spec["jobs"], spec["cfg"])


def restore_jobs_from_inputs():
    """Resume: copy finished job files from a previous run attached as input."""
    os.makedirs(JOB_DIR, exist_ok=True); n = 0
    for f in all_input_files():
        parent = os.path.dirname(f)
        if f.endswith(".npz") and os.path.basename(parent) == "jobs" \
                and os.path.basename(os.path.dirname(parent)) == os.path.basename(OUT_DIR):
            dst = os.path.join(JOB_DIR, os.path.basename(f))
            if not os.path.isfile(dst): shutil.copy(f, dst); n += 1
    if n: print(f"resumed {n} finished models from inputs")


# ------------------------------- statistics ----------------------------------
def rankise(p):
    from scipy.stats import rankdata
    return rankdata(p) / len(p)


def auc_ci(y, p, B=2000, seed=42):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); n = len(y); b = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) > 1: b.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def delong(y, p1, p2):
    """DeLong test for two correlated AUCs on the same patients. Returns (z, p)."""
    from scipy.stats import norm
    y = np.asarray(y).astype(int); pos, neg = y == 1, y == 0
    def comps(p):
        d = p[pos][:, None] - p[neg][None, :]
        psi = (d > 0) + 0.5 * (d == 0)
        return psi.mean(1), psi.mean(0), psi.mean()
    a10, a01, a = comps(np.asarray(p1)); b10, b01, b = comps(np.asarray(p2))
    S = np.cov(np.vstack([a10, b10])) / pos.sum() + np.cov(np.vstack([a01, b01])) / neg.sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0: return 0.0, 1.0
    z = (a - b) / np.sqrt(var); return float(z), float(2 * norm.sf(abs(z)))


def mcc_youden(y, p):
    from sklearn.metrics import roc_curve, matthews_corrcoef
    fpr, tpr, thr = roc_curve(y, p); t = thr[np.argmax(tpr - fpr)]
    return matthews_corrcoef(y, (p >= t).astype(int))


def skill_weighted(per_mod, y):
    from sklearn.metrics import roc_auc_score
    w = np.array([max(roc_auc_score(y, per_mod[m]) - 0.5, 1e-3) for m in MODALITIES]); w /= w.sum()
    return sum(w[i] * rankise(per_mod[m]) for i, m in enumerate(MODALITIES))


def assemble_cv(kind, backbone, n, which):
    """Seed-averaged OOF per modality for a cv/control group. None if incomplete."""
    per_mod = {}
    for mod in MODALITIES:
        acc = np.zeros(n)
        for seed in SEEDS:
            oof = np.full(n, np.nan)
            for fold in range(N_FOLDS):
                f = os.path.join(JOB_DIR, job_name(kind, backbone, mod, seed, fold) + ".npz")
                if not os.path.isfile(f): return None
                d = np.load(f); oof[d["va_idx"]] = d[which]
            acc += oof / len(SEEDS)
        per_mod[mod] = acc
    return per_mod


def assemble_ext(site, backbone, n_test, which):
    per_mod = {}
    for mod in MODALITIES:
        preds = []
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                f = os.path.join(JOB_DIR, job_name(f"ext{site}", backbone, mod, seed, fold) + ".npz")
                if not os.path.isfile(f): return None
                preds.append(np.load(f)[which])
        per_mod[mod] = np.mean(preds, axis=0)
    return per_mod


def load_prior(df):
    """Earlier OOF predictions attached as inputs, aligned to the cohort.
    These were all trained with a different input configuration, so they are
    not reported when SINGLE_SLICE is on."""
    found = {}
    if SINGLE_SLICE:
        return found
    names = {"oof_predictions.csv": "Paper headline B0 (BraTS only)",
             "oof_improve_b1_3sl.csv": "Prior B1 (BraTS only)",
             "oof_improve_b2_3sl.csv": "Prior B2 (BraTS only)",
             "oof_improve_b0_5sl.csv": "Prior B0 5-slice (BraTS only)",
             "oof_ablation_nocrop.csv": "Prior B0 no-crop (BraTS only)"}
    for fn, label in names.items():
        hits = [h for h in find_files(fn) if "outputs_merged" not in h]
        if not hits: continue
        d = pd.read_csv(hits[0]); m = dict(zip(d.BraTS21ID.astype(int), d.pred.astype(float)))
        p = np.array([m.get(int(i), np.nan) for i in df.BraTS21ID], float)
        if np.isnan(p).any():
            print(f"   prior {fn}: {int(np.isnan(p).sum())} patients missing -> not used"); continue
        found[label] = p
    return found


# ------------------------------- reporting ------------------------------------
def report(df, y, fold42, groups, prior, external, cohort):
    from sklearn.metrics import roc_auc_score
    ref_name = "Paper headline B0 (BraTS only)"
    ref = prior.get(ref_name)
    ctrl = groups.get("Control B0 (BraTS only, this code)", {}).get("best")
    merged = [n for n in groups if n.startswith("Merged")]
    # baselines are compared with the primary ensemble AND with the best single
    # merged model (single-vs-single, so the ensemble is not the only reference)
    primary = (np.mean([rankise(groups[n]["best"]) for n in merged], 0) if len(merged) >= 2
               else (groups[merged[0]]["best"] if merged else None))
    single = max(merged, key=lambda n: roc_auc_score(y, groups[n]["best"])) if merged else None

    rows, oof_out = [], {}
    def add(name, p, strict=None):
        a, lo, hi = auc_ci(y, p)
        row = {"Model": name, "AUC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})",
               "AUC strict": f"{roc_auc_score(y, strict):.3f}" if strict is not None else "-",
               "MCC": round(mcc_youden(y, p), 3)}
        if ref is not None and name != ref_name:
            z, pv = delong(y, p, ref)
            row["dAUC vs paper"] = f"{a - roc_auc_score(y, ref):+.3f}"
            row["DeLong p vs paper"] = f"{pv:.4f}"
        if ctrl is not None and "Merged" in name:
            z, pv = delong(y, p, ctrl)
            row["dAUC vs control"] = f"{a - roc_auc_score(y, ctrl):+.3f}"
            row["DeLong p vs control"] = f"{pv:.4f}"
        if name.startswith("Baseline") and primary is not None:
            z, pv = delong(y, primary, p)
            row["primary - baseline"] = f"{roc_auc_score(y, primary) - a:+.3f}"
            row["DeLong p (primary vs baseline)"] = f"{pv:.4f}"
            sp = groups[single]["best"]; z, pv = delong(y, sp, p)
            row[f"{single.split()[1]} - baseline"] = f"{roc_auc_score(y, sp) - a:+.3f}"
            row[f"DeLong p ({single.split()[1]} vs baseline)"] = f"{pv:.4f}"
        for f in range(N_FOLDS):
            m = fold42 == f; row[f"Fold {f+1}"] = round(roc_auc_score(y[m], p[m]), 3)
        rows.append(row); oof_out[name] = p

    if ref is not None: add(ref_name, ref)
    for name, g in groups.items(): add(name, g["best"], g["last"])
    merged = [n for n in groups if n.startswith("Merged")]
    if len(merged) >= 2:
        add("Merged ENS (" + " + ".join(n.split()[1] for n in merged) + ")  <- primary",
            np.mean([rankise(groups[n]["best"]) for n in merged], 0),
            np.mean([rankise(groups[n]["last"]) for n in merged], 0))
    if merged and prior:
        comps = [groups[n]["best"] for n in merged] + list(prior.values())
        add(f"Merged + prior EfficientNet runs ({len(comps)} models)",
            np.mean([rankise(p) for p in comps], 0))
    for label, p in prior.items():
        if label != ref_name: add(label, p)

    tbl = pd.DataFrame(rows).fillna("-")
    tbl.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
    for name, p in oof_out.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
        pd.DataFrame({"BraTS21ID": df.BraTS21ID.values, "MGMT_value": y, "pred": p,
                      "fold": fold42}).to_csv(os.path.join(OUT_DIR, f"oof_merged_{slug}.csv"), index=False)

    print("\n" + "=" * 110)
    print(f"RESULTS on the same {len(y)} RSNA-MICCAI + BraTS 2021 patients as the paper")
    print("=" * 110)
    print(tbl.to_string(index=False))
    print("-" * 110)
    print("AUC = paper protocol (epoch picked on the held-out fold, as in the paper).")
    print("AUC strict = final epoch, nothing tuned on the held-out fold (no optimism).")

    ext_tbl = None
    if external:
        ext_rows = []
        for site, d in external.items():
            a, lo, hi = auc_ci(d["y"], d["best"])
            ext_rows.append({"Held-out site": site, "n": len(d["y"]),
                             "pos rate": round(float(np.mean(d["y"])), 3),
                             "AUC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                             "AUC strict": round(roc_auc_score(d["y"], d["last"]), 3),
                             **{m: round(roc_auc_score(d["y"], d["per_mod"][m]), 3) for m in MODALITIES}})
            # sensitivity check: leave out kept patients that look similar (0.90-0.95)
            # to a BraTS patient, in case any of them is an undetected duplicate
            keep = ~np.isin(np.array(d["subject_id"]), list(d["borderline"]))
            if keep.sum() < len(keep) and len(np.unique(d["y"][keep])) > 1:
                a2, lo2, hi2 = auc_ci(d["y"][keep], d["best"][keep])
                ext_rows[-1]["AUC excl. borderline (sim 0.90-0.95)"] = \
                    f"{a2:.3f} ({lo2:.3f}-{hi2:.3f}), n={int(keep.sum())}"
            pd.DataFrame({"subject_id": d["subject_id"], "MGMT_value": d["y"], "pred": d["best"]}
                         ).to_csv(os.path.join(OUT_DIR, f"external_preds_{site}.csv"), index=False)
        ext_tbl = pd.DataFrame(ext_rows)
        ext_tbl.to_csv(os.path.join(OUT_DIR, "external_table.csv"), index=False)
        print("\nLEAVE-ONE-SITE-OUT (trained without the site, duplicates removed)")
        print(ext_tbl.to_string(index=False))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump({"cohort": cohort, "main": tbl.to_dict("records"),
                   "external": ext_tbl.to_dict("records") if ext_tbl is not None else []},
                  fh, indent=2, default=str)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cols = ["Model", "AUC (95% CI)", "AUC strict", "MCC"] + \
               [c for c in ("dAUC vs paper", "DeLong p vs paper") if c in tbl.columns]
        t = tbl[cols]
        fig, ax = plt.subplots(figsize=(13, 1.6 + 0.45 * len(t))); ax.axis("off")
        ax.set_title(f"Multi-cohort training, evaluated on the paper's {len(y)} patients",
                     fontsize=11, fontweight="bold", loc="left")
        tb = ax.table(cellText=t.values, colLabels=t.columns, cellLoc="center", loc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.5)
        tb.auto_set_column_width(col=list(range(len(cols))))
        for (r, c), cell in tb.get_celld().items():
            if r == 0: cell.set_facecolor("#1f3a5f"); cell.set_text_props(color="w", fontweight="bold")
            elif "primary" in str(t.iloc[r - 1, 0]): cell.set_facecolor("#e8f0fe")
        fig.savefig(os.path.join(OUT_DIR, "results_table.png"), dpi=200, bbox_inches="tight")
    except Exception as e:
        print("(table image skipped:", e, ")")
    print("\nsaved to", os.path.abspath(OUT_DIR))


# ------------------------------- ablation --------------------------------------
def ablation_report(df, y, fold42, groups, abl):
    """Remove one component of the proposed B0+B2 ensemble at a time. Every row is
    an ensemble of the same backbones (except '− ensemble'), compared with the full
    model on the same 574 patients (DeLong)."""
    from sklearn.metrics import roc_auc_score
    find = lambda kind, bb: next((g for g in groups.values()
                                  if g.get("kind") == kind and g.get("backbone") == bb), None)
    merged = {bb: find("cv", bb) for bb in BACKBONES}
    if any(g is None for g in merged.values()):
        print("ablation: proposed model incomplete -> skipped"); return
    ens = lambda arrs: np.mean([rankise(a) for a in arrs], 0)
    mean_mod = lambda pm: np.mean([rankise(pm[m]) for m in MODALITIES], 0)
    both = lambda f: (ens([f(b, "best") for b in BACKBONES]), ens([f(b, "last") for b in BACKBONES]))
    rows_def = [("Proposed (full)", *both(lambda b, w: merged[b][w]))]
    ctrl = {bb: find("control", bb) for bb in BACKBONES}
    if all(ctrl.values()):
        rows_def.append(("− Multi-cohort data (BraTS only)", *both(lambda b, w: ctrl[b][w])))
    if len(BACKBONES) > 1:
        b2 = merged[BACKBONES[0]]
        rows_def.append((f"− Ensemble (EfficientNet-{short(BACKBONES[0]).upper()} only)",
                         b2["best"], b2["last"]))
    rows_def.append(("− Skill weighting (plain mean of modalities)",
                     *both(lambda b, w: mean_mod(merged[b]["pm_" + w]))))
    if "FLAIR" in MODALITIES:
        rows_def.append(("− Multimodal fusion (FLAIR only)",
                         *both(lambda b, w: merged[b]["pm_" + w]["FLAIR"])))
    for v in ABLATIONS:
        if v in abl and all(bb in abl[v] for bb in BACKBONES):
            rows_def.append((ABLATION_LABELS[v], *both(lambda b, w: abl[v][b][w])))

    full, full_last = rows_def[0][1], rows_def[0][2]
    a_full, s_full = roc_auc_score(y, full), roc_auc_score(y, full_last)
    rows = []
    for k, (name, p, pl) in enumerate(rows_def):
        a, lo, hi = auc_ci(y, p); s = roc_auc_score(y, pl)
        row = {"Variant": name, "AUC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})",
               "dAUC": "-" if k == 0 else f"{a - a_full:+.3f}",
               "DeLong p vs full": "-" if k == 0 else f"{delong(y, full, p)[1]:.4f}",
               "AUC strict": f"{s:.3f}", "dAUC strict": "-" if k == 0 else f"{s - s_full:+.3f}",
               "MCC": round(mcc_youden(y, p), 3)}
        for f in range(N_FOLDS):
            m = fold42 == f; row[f"Fold {f+1}"] = round(roc_auc_score(y[m], p[m]), 3)
        rows.append(row)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
        pd.DataFrame({"BraTS21ID": df.BraTS21ID.values, "MGMT_value": y, "pred": p, "pred_strict": pl,
                      "fold": fold42}).to_csv(os.path.join(OUT_DIR, f"oof_ablation_{slug}.csv"), index=False)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT_DIR, "ablation_table.csv"), index=False)
    print("\n" + "=" * 110 + "\nABLATION of the proposed model (one component removed at a time)\n" + "=" * 110)
    print(tbl.to_string(index=False))
    print("dAUC < 0 means the removed component helps. 'strict' = final epoch (no selection on the test fold).")


# ------------------------------- Grad-CAM --------------------------------------
def gradcam_figure(df, y, brats_cache):
    """Grad-CAM heat-maps of the saved B2 models on held-out fold-1 patients
    (the 4 most confident correct methylated and 4 unmethylated cases)."""
    import torch, cv2
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    mods = [m for m in GRADCAM_MODS if m in MODALITIES]
    paths = {m: os.path.join(WEIGHT_DIR, job_name("cam", GRADCAM_BACKBONE, m, SEEDS[0], 0) + ".pt")
             for m in mods}
    if not mods or not all(os.path.isfile(p) for p in paths.values()):
        print("Grad-CAM: saved model weights not found -> skipped"); return
    ids = df.BraTS21ID.astype(str).values
    _, va = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEEDS[0]).split(ids, y))[0]
    dev = torch.device("cuda:0" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    def load_model(p):
        m = make_model(GRADCAM_BACKBONE, False)
        m.load_state_dict(torch.load(p, map_location="cpu")); return m.to(dev).eval()

    def cam(model, arr):
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].float().to(dev)
        x = (x / 255.0 - 0.5) / 0.5
        feat = model.forward_features(x); feat.retain_grad()
        logit = model.forward_head(feat).reshape(())
        model.zero_grad(set_to_none=True); logit.backward()
        w = feat.grad.mean(dim=(2, 3), keepdim=True)
        c = torch.relu((w * feat).sum(1))[0].detach().cpu().numpy()
        c = cv2.resize(c, (arr.shape[1], arr.shape[0])); c = c / (c.max() + 1e-8)
        return c, float(torch.sigmoid(logit).item())

    models = {m: load_model(p) for m, p in paths.items()}
    def load_arr(m, i):
        a = np.load(os.path.join(brats_cache, f"{m}_{i}.npy"))
        return np.ascontiguousarray(np.repeat(a[:, :, 1:2], 3, axis=2)) if SINGLE_SLICE else a
    arrs = {m: {i: load_arr(m, i) for i in ids[va]} for m in mods}
    yv = dict(zip(ids[va], y[va]))
    probs = {i: cam(models[mods[0]], arrs[mods[0]][i])[1] for i in ids[va]}
    pos = sorted([i for i in ids[va] if yv[i] == 1 and probs[i] > 0.5], key=lambda i: -probs[i])[:4]
    neg = sorted([i for i in ids[va] if yv[i] == 0 and probs[i] < 0.5], key=lambda i: probs[i])[:4]
    pick = pos + neg
    if not pick:
        print("Grad-CAM: no confidently correct cases -> skipped"); return
    fig, axes = plt.subplots(len(pick), 2 * len(mods), figsize=(5.2 * len(mods), 2.7 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        for k, m in enumerate(mods):
            c, p = cam(models[m], arrs[m][i]); img = arrs[m][i][:, :, 1]
            ax = axes[r, 2 * k]; ax.imshow(img, cmap="gray"); ax.axis("off")
            ax.set_title(f"{m} | BraTS21 {i} | {'methylated' if yv[i] else 'unmethylated'}", fontsize=8)
            ax = axes[r, 2 * k + 1]; ax.imshow(img, cmap="gray"); ax.imshow(c, cmap="jet", alpha=0.4)
            ax.axis("off"); ax.set_title(f"Grad-CAM | p(methylated) = {p:.2f}", fontsize=8)
    fig.suptitle(f"Grad-CAM, {GRADCAM_BACKBONE} (held-out fold 1; most confident correct cases)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_gradcam.{ext}"), dpi=200, bbox_inches="tight")
    print("saved fig_gradcam.png / .pdf")


# ---------------------------------- main --------------------------------------
def preflight():
    import torch
    if DEVICE != "cuda": print(f"PREFLIGHT: running on {DEVICE} (test mode)"); return
    if not torch.cuda.is_available():
        sys.exit("PREFLIGHT: no GPU. Settings -> Accelerator -> GPU T4 x2, then re-run.")
    n = torch.cuda.device_count()
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        print(f"PREFLIGHT: GPU{i} {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]}")
    if n < 2:
        print("PREFLIGHT WARNING: only one GPU visible -- choose 'GPU T4 x2' to halve the run time.")



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


def main():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    t_start = time.time()
    print_run_mode()
    print(f"SCRIPT VERSION: {SCRIPT_VERSION}\n", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CACHE_ROOT, exist_ok=True)
    print("QUICK_TEST" if QUICK_TEST else "FULL RUN", "| epochs", EPOCHS, "| seeds", SEEDS,
          "| backbones", BACKBONES)
    list_inputs()

    # ---- evaluation cohort: exactly the paper's 574 patients ----------------
    lab = find_files("train_labels.csv")
    assert lab, "train_labels.csv not found -- add the RSNA-MICCAI competition as an input."
    train_df = pd.read_csv(lab[0])
    train_df = train_df[~train_df.BraTS21ID.isin(BAD_IDS)].reset_index(drop=True)
    source = find_brats_source()
    brats_cache, covered = build_brats_cache(set(train_df.BraTS21ID.astype(int)), source)
    df = train_df[train_df.BraTS21ID.astype(int).isin(covered)].reset_index(drop=True)
    y = df.MGMT_value.values.astype(int)
    print(f"evaluation cohort: {len(df)} patients, positive rate {y.mean():.3f}")
    if len(df) != 574 and not QUICK_TEST and ON_KAGGLE:
        print(f"WARNING: expected 574 patients (paper), got {len(df)}.")

    # ---- extra cohorts --------------------------------------------------------
    axcodes = brats_axcodes(source) if source[0] else ("L", "P", "S")
    print("BraTS orientation:", axcodes)
    sites, caches, cohort, dedup = {}, {"BRATS": brats_cache}, {"BRATS_eval": int(len(df))}, []
    site_meta = {}
    for site in EXTRA_SITES:
        cache, meta = build_site_cache(site, axcodes)
        if meta is None or meta.empty: continue
        site_meta[site] = meta
        rep = dedup_audit(site, meta, cache, brats_cache, [str(i) for i in df.BraTS21ID],
                          {str(i): int(v) for i, v in zip(df.BraTS21ID, y)})
        dedup.append(rep)
        keep = meta[~meta.idx.isin(rep.loc[rep.excluded, "idx"])]
        sites[site] = {"ids": [str(i) for i in keep.idx], "y": [int(v) for v in keep.MGMT_value],
                       "subject_id": list(keep.subject_id)}
        caches[site] = cache
        cohort[site] = {"cached": int(len(meta)), "excluded_duplicates": int(rep.excluded.sum()),
                        "used": int(len(keep)), "pos_rate": round(float(keep.MGMT_value.mean()), 3)}
    if dedup:
        pd.concat(dedup).to_csv(os.path.join(OUT_DIR, "dedup_report.csv"), index=False)
    if not sites:
        print("\nWARNING: no extra cohort found -- only the BraTS-only control can run.")
    print("cohort summary:", json.dumps(cohort))
    n_train = int(len(df) * (N_FOLDS - 1) / N_FOLDS) + sum(len(s["ids"]) for s in sites.values())
    print(f"training patients per fold: ~{n_train} (paper: ~{int(len(df)*0.8)})")

    # ---- ablation image caches (whole slice / centre slice) -------------------
    caches_all = {"main": caches}
    abl_ok = []
    if RUN_ABLATION and sites:
        for v in ABLATIONS:
            if v in CACHE_VARIANTS:
                bdir, _ = build_brats_cache(set(df.BraTS21ID.astype(int)), source, v)
                vc = {"BRATS": bdir}
                for site in sites:
                    vc[site] = build_site_variant_cache(site, axcodes, v, site_meta[site])
                if cache_complete(bdir, [str(i) for i in df.BraTS21ID]) and \
                        all(cache_complete(vc[s], sites[s]["ids"]) for s in sites):
                    caches_all[v] = vc
                else:
                    print(f"WARNING: '{v}' images incomplete -> that ablation row is skipped"); continue
            abl_ok.append(v)

    # ---- job list -------------------------------------------------------------
    jobs = []
    def add_jobs(kind, backbone, train_sites, test_site=None, variant="main"):
        for mod in MODALITIES:
            for seed in SEEDS:
                for fold in range(N_FOLDS):
                    jobs.append({"name": job_name(kind, backbone, mod, seed, fold),
                                 "backbone": backbone, "modality": mod, "seed": seed, "fold": fold,
                                 "epochs": EPOCHS, "train_sites": train_sites, "test_site": test_site,
                                 "variant": variant})
    if sites:
        for bb in BACKBONES: add_jobs("cv", bb, list(sites))
        if RUN_BASELINES:
            for bb in BASELINE_BACKBONES: add_jobs("base", bb, list(sites))
        if RUN_EXTERNAL:
            for held in sites:
                for bb in EXTERNAL_BACKBONES:
                    add_jobs(f"ext{held}", bb, [s for s in sites if s != held], held)
        if RUN_GRADCAM:      # short: 1 seed x fold 1, weights kept for the heat-maps
            for mod in GRADCAM_MODS:
                jobs.append({"name": job_name("cam", GRADCAM_BACKBONE, mod, SEEDS[0], 0),
                             "backbone": GRADCAM_BACKBONE, "modality": mod, "seed": SEEDS[0],
                             "fold": 0, "epochs": EPOCHS, "train_sites": list(sites),
                             "test_site": None, "variant": "main", "save_weights": True})
        for v in abl_ok:
            for bb in BACKBONES: add_jobs(f"abl{v}", bb, list(sites), variant=v)
    if RUN_CONTROL or not sites:
        add_jobs("control", "efficientnet_b0", [])
    if RUN_CONTROL_B2 and sites:
        add_jobs("control", "efficientnet_b2", [])

    restore_jobs_from_inputs()
    preflight()
    if PRETRAINED:   # download weights once, before the workers start
        import timm
        for bb in set(j["backbone"] for j in jobs):
            if bb != "simple_cnn": timm.create_model(bb, pretrained=True)
    cfg = {"brats_ids": [str(i) for i in df.BraTS21ID], "brats_y": [int(v) for v in y],
           "sites": sites, "caches": caches_all, "seeds": SEEDS,
           "deadline": t_start + TIME_BUDGET_H * 3600}
    run_jobs(jobs, cfg)

    # ---- assemble + report ----------------------------------------------------
    fold42 = np.empty(len(y), int)
    for f, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEEDS[0]).split(y, y)):
        fold42[va] = f
    groups, missing = {}, []
    group_defs = [("cv", bb, f"Merged {short(bb).upper()} (BraTS + {'+'.join(sites)})") for bb in BACKBONES] \
        if sites else []
    if RUN_BASELINES and sites:
        group_defs += [("base", bb, f"Baseline {BASELINE_LABELS.get(bb, bb)} (BraTS + {'+'.join(sites)})")
                       for bb in BASELINE_BACKBONES]
    if RUN_CONTROL or not sites:
        group_defs.append(("control", "efficientnet_b0", "Control B0 (BraTS only, this code)"))
    if RUN_CONTROL_B2 and sites:
        group_defs.append(("control", "efficientnet_b2", "Control B2 (BraTS only, this code)"))
    for kind, bb, label in group_defs:
        best, last = assemble_cv(kind, bb, len(y), "best_va"), assemble_cv(kind, bb, len(y), "last_va")
        if best is None: missing.append(label); continue
        groups[label] = {"best": skill_weighted(best, y), "last": skill_weighted(last, y),
                         "pm_best": best, "pm_last": last, "backbone": bb, "kind": kind,
                         "per_mod": {m: round(float(roc_auc_score(y, best[m])), 3) for m in MODALITIES}}
        print(f"{label}: per-modality AUC {groups[label]['per_mod']}")
    abl = {}
    for v in abl_ok:
        for bb in BACKBONES:
            b, l = assemble_cv(f"abl{v}", bb, len(y), "best_va"), assemble_cv(f"abl{v}", bb, len(y), "last_va")
            if b is None: missing.append(f"ablation {v} {short(bb).upper()}"); continue
            abl.setdefault(v, {})[bb] = {"best": skill_weighted(b, y), "last": skill_weighted(l, y)}
    external = {}
    if RUN_EXTERNAL:
        for held, s in sites.items():
            # one prediction per backbone, then rank-averaged over backbones and MRI types
            bests = [assemble_ext(held, bb, len(s["y"]), "best_te") for bb in EXTERNAL_BACKBONES]
            lasts = [assemble_ext(held, bb, len(s["y"]), "last_te") for bb in EXTERNAL_BACKBONES]
            if any(b is None for b in bests): missing.append(f"external {held}"); continue
            rep = next((r for r in dedup if len(r) and r.site.iloc[0] == held), None)
            near = set() if rep is None else set(rep.loc[~rep.excluded & (rep.max_cosine > 0.90), "subject_id"])
            external[held] = {"y": np.array(s["y"]), "subject_id": s["subject_id"],
                              "per_mod": {m: np.mean([rankise(b[m]) for b in bests], 0) for m in MODALITIES},
                              "borderline": near,
                              "best": np.mean([rankise(b[m]) for b in bests for m in MODALITIES], 0),
                              "last": np.mean([rankise(l[m]) for l in lasts for m in MODALITIES], 0)}
    if missing:
        print("\nINCOMPLETE (re-run with this output attached as input to finish):", missing)
    prior = load_prior(df)
    if groups or prior:
        report(df, y, fold42, groups, prior, external, cohort)
    for step, fn in (("ablation", lambda: ablation_report(df, y, fold42, groups, abl)),
                     ("Grad-CAM", lambda: gradcam_figure(df, y, caches["BRATS"]) if RUN_GRADCAM else None)):
        try:
            fn()
        except Exception:
            print(f"({step} step failed -- other results are unaffected)\n{traceback.format_exc()}")
    print(f"\nDONE in {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _worker_cli(int(sys.argv[2]), sys.argv[3])     # one GPU process
    else:
        main()
