"""Rebuild MGMT Run 4 results from the archived model predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from training.kaggle_merged_training import auc_ci, delong, rankise, skill_weighted

MODALITIES = ("FLAIR", "T1w", "T1wCE", "T2w")
BACKBONES = ("b2", "b0")
ESTIMATES = ("checkpoint_selected", "fixed_epoch")
BASELINES = {
    "DenseNet-121": ("densenet121", "densenet_121"),
    "ResNet-18": ("resnet18", "resnet_18"),
    "MobileNet-v2": ("mobilenetv2_100", "mobilenet_v2"),
    "Four-layer CNN": ("simple_cnn", "simple_cnn_4_layer_from_scratch"),
}


def archive_members(archive):
    return {
        PurePosixPath(name).name: name
        for name in archive.namelist()
        if not name.endswith("/") and "__MACOSX" not in PurePosixPath(name).parts
    }


def open_run(stack, path):
    archive = stack.enter_context(ZipFile(path))
    members = archive_members(archive)
    if "worker_spec.json" in members:
        return archive, members
    nested = members.get("outputs_merged_run4.zip")
    if not nested:
        raise ValueError("Supply the Run 4 ZIP or its inner outputs_merged_run4.zip.")
    archive = stack.enter_context(ZipFile(BytesIO(archive.read(nested))))
    return archive, archive_members(archive)


def rank_average(*scores):
    return np.mean([rankise(score) for score in scores], axis=0)


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(order))
    cumulative = 0.0
    for rank, index in enumerate(order):
        cumulative = max(cumulative, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = cumulative
    return adjusted


def calibrate(scores, labels, folds):
    probabilities = np.empty(len(labels))
    for fold in np.unique(folds):
        test = folds == fold
        model = LogisticRegression(C=1e6).fit(scores[~test, None], labels[~test])
        probabilities[test] = model.predict_proba(scores[test, None])[:, 1]
    return probabilities


def classification_metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp) if tp + fp else 0.0,
        "NPV": tn / (tn + fn) if tn + fn else 0.0,
        "F1": f1_score(labels, predictions),
        "MCC": matthews_corrcoef(labels, predictions),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def target_specificity_threshold(labels, probabilities, target=0.90):
    fpr, _, thresholds = roc_curve(labels, probabilities)
    return thresholds[np.flatnonzero(1 - fpr >= target)[-1]]


class Run4:
    """Read the saved jobs and check each validation fold against its run configuration."""

    def __init__(self, archive, members):
        self.archive = archive
        self.members = members
        spec = json.loads(self.read("worker_spec.json"))
        self.config = spec["cfg"]
        self.labels = np.array(self.config["brats_y"], dtype=int)
        self.patient_ids = np.array(self.config["brats_ids"], dtype=int)
        self.seeds = self.config["seeds"]
        self.splits = {
            seed: [test for _, test in StratifiedKFold(5, shuffle=True, random_state=seed)
                   .split(self.patient_ids, self.labels)]
            for seed in self.seeds
        }
        self.folds = np.empty(len(self.labels), dtype=int)
        for fold, test in enumerate(self.splits[self.seeds[0]]):
            self.folds[test] = fold
        self.scheduled_jobs = len(spec["jobs"])
        missing = [job["name"] for job in spec["jobs"] if job["name"] + ".npz" not in members]
        if missing:
            raise ValueError(f"Missing {len(missing)} scheduled jobs, starting with {missing[0]}.")
        self.archived_jobs = sum(name.endswith(".npz") for name in members)

    def read(self, name):
        if name not in self.members:
            raise FileNotFoundError(f"{name} is missing from the Run 4 archive.")
        return self.archive.read(self.members[name])

    def csv(self, name):
        return pd.read_csv(BytesIO(self.read(name)), float_precision="round_trip")

    def job(self, prefix, backbone, modality, seed, fold, keys):
        name = f"{prefix}_{backbone}_{modality}_s{seed}_f{fold}_e20.npz"
        with np.load(BytesIO(self.read(name)), allow_pickle=False) as data:
            if not np.array_equal(data["va_idx"], self.splits[seed][fold]):
                raise ValueError(f"Validation patient order differs for {name}.")
            return [data[key] for key in keys]

    def internal(self, prefix, backbone):
        result = {estimate: {} for estimate in ESTIMATES}
        for modality in MODALITIES:
            repetitions = []
            for seed in self.seeds:
                scores = np.full((2, len(self.labels)), np.nan)
                for fold, test in enumerate(self.splits[seed]):
                    scores[:, test] = self.job(
                        prefix, backbone, modality, seed, fold, ("best_va", "last_va")
                    )
                if not np.isfinite(scores).all():
                    raise ValueError(f"Incomplete predictions for {prefix}/{backbone}/{modality}.")
                repetitions.append(scores)
            for estimate, score in zip(ESTIMATES, np.mean(repetitions, axis=0)):
                result[estimate][modality] = score
        return result

    def external(self, site):
        result = {estimate: {} for estimate in ESTIMATES}
        for backbone in BACKBONES:
            for modality in MODALITIES:
                predictions = [
                    self.job(f"ext{site}SS", backbone, modality, seed, fold, ("best_te", "last_te"))
                    for seed in self.seeds
                    for fold in range(5)
                ]
                for estimate, score in zip(ESTIMATES, np.mean(predictions, axis=0)):
                    result[estimate][backbone, modality] = score
        return result

    def check_internal_csv(self, name, score, column="pred"):
        saved = self.csv(name)
        if not all((
            np.array_equal(saved.BraTS21ID, self.patient_ids),
            np.array_equal(saved.MGMT_value, self.labels),
            np.array_equal(saved.fold, self.folds),
        )):
            raise ValueError(f"Patient IDs, labels, or folds differ in {name}.")
        self.check_scores(name, saved[column], score)

    @staticmethod
    def check_scores(name, archived, rebuilt):
        if len(archived) != len(rebuilt) or not np.allclose(archived, rebuilt, atol=1e-15, rtol=0):
            raise ValueError(f"Rebuilt predictions differ from {name}.")


def fuse_modalities(by_modality, labels):
    return {estimate: skill_weighted(by_modality[estimate], labels) for estimate in ESTIMATES}


def combine_backbones(by_backbone):
    return {
        estimate: rank_average(*(by_backbone[b][estimate] for b in BACKBONES))
        for estimate in ESTIMATES
    }


def compare(labels, reference, alternatives):
    rows = []
    for estimate in ESTIMATES:
        family = []
        for name, scores in alternatives.items():
            a, b = reference[name][estimate], scores[estimate]
            z, p_value = delong(labels, a, b)
            family.append({
                "comparison": name, "estimate": estimate,
                "reference_auc": roc_auc_score(labels, a), "comparison_auc": roc_auc_score(labels, b),
                "reference_minus_comparison_auc": roc_auc_score(labels, a) - roc_auc_score(labels, b),
                "DeLong_z": z, "DeLong_p": p_value,
            })
        for row, adjusted in zip(family, holm([row["DeLong_p"] for row in family])):
            row["Holm_p"] = adjusted
        rows.extend(family)
    return rows


def write_table(output, name, rows):
    pd.DataFrame(rows).to_csv(output / name, index=False)


def summarize_internal(run, output):
    y = run.labels
    modal, fused = {}, {}
    for b in BACKBONES:
        modal[b] = run.internal("ablsingle", b)
        fused[b] = fuse_modalities(modal[b], y)
        run.check_internal_csv(f"oof_merged_merged_{b}_brats_upenn_ucsf.csv", fused[b][ESTIMATES[0]])
    for name, (backbone, slug) in BASELINES.items():
        modal[name] = run.internal("baseSS", backbone)
        fused[name] = fuse_modalities(modal[name], y)
        run.check_internal_csv(f"oof_merged_baseline_{slug}_brats_upenn_ucsf.csv", fused[name][ESTIMATES[0]])
    for b in BACKBONES:
        name = f"BraTS only {b}"
        modal[name] = run.internal("controlSS", b)
        fused[name] = fuse_modalities(modal[name], y)
        run.check_internal_csv(f"oof_merged_control_{b}_brats_only_this_code.csv", fused[name][ESTIMATES[0]])

    full = fused["Proposed"] = combine_backbones(fused)
    fused["BraTS-only ensemble"] = combine_backbones({b: fused[f"BraTS only {b}"] for b in BACKBONES})
    modal["Proposed"] = {
        estimate: {m: rank_average(*(modal[b][estimate][m] for b in BACKBONES)) for m in MODALITIES}
        for estimate in ESTIMATES
    }
    run.check_internal_csv("oof_merged_merged_ens_b2_b0_primary.csv", full[ESTIMATES[0]])

    rows, fold_rows = [], []
    for name, estimates in fused.items():
        for estimate, score in estimates.items():
            sequences = {**modal[name][estimate], "Fused": score} if name in modal else {"Fused": score}
            for sequence, predictions in sequences.items():
                probabilities = calibrate(predictions, y, run.folds)
                rows.append({
                    "model": name, "sequence": sequence, "estimate": estimate,
                    "AUC": roc_auc_score(y, predictions), "AP": average_precision_score(y, predictions),
                    **classification_metrics(y, probabilities >= 0.5),
                })
            for fold in np.unique(run.folds):
                test = run.folds == fold
                fold_rows.append({
                    "model": name, "estimate": estimate, "fold": int(fold + 1),
                    "AUC": roc_auc_score(y[test], score[test]),
                })
    write_table(output, "internal_metrics.csv", rows)
    write_table(output, "fold_auc.csv", fold_rows)
    write_table(output, "baseline_tests.csv", compare(
        y, {name: full for name in BASELINES}, {name: fused[name] for name in BASELINES}
    ))
    write_table(output, "data_addition_tests.csv", compare(
        y, {"B0": fused["b0"], "B2": fused["b2"], "Ensemble": full},
        {"B0": fused["BraTS only b0"], "B2": fused["BraTS only b2"],
         "Ensemble": fused["BraTS-only ensemble"]},
    ))

    ablations = {
        "multi_cohort_data_brats_only": fused["BraTS-only ensemble"],
        "ensemble_efficientnet_b2_only": fused["b2"],
        "skill_weighting_plain_mean_of_modalities": {
            estimate: rank_average(*(np.mean(
                [rankise(modal[b][estimate][m]) for m in MODALITIES], axis=0
            ) for b in BACKBONES))
            for estimate in ESTIMATES
        },
        "multimodal_fusion_flair_only": {
            estimate: modal["Proposed"][estimate]["FLAIR"] for estimate in ESTIMATES
        },
    }
    for prefix, label in (
        ("ablrandinitSS", "imagenet_pretraining_random_init"),
        ("ablnocropSS", "tumour_focus_crop_whole_slice_tumour_slice"),
        ("ablcenterSS", "tumour_guidance_centre_slice_whole_image"),
    ):
        ablations[label] = combine_backbones({
            b: fuse_modalities(run.internal(prefix, b), y) for b in BACKBONES
        })
    ablation_rows = []
    for name, estimates in {**ablations, "proposed_full": full}.items():
        for estimate, column in zip(ESTIMATES, ("pred", "pred_strict")):
            score = estimates[estimate]
            run.check_internal_csv(f"oof_ablation_{name}.csv", score, column)
            ablation_rows.append({
                "variant": name, "estimate": estimate, "AUC": roc_auc_score(y, score),
                **classification_metrics(y, calibrate(score, y, run.folds) >= 0.5),
            })
    write_table(output, "ablation_metrics.csv", ablation_rows)
    write_table(output, "ablation_tests.csv", compare(y, {name: full for name in ablations}, ablations))

    earlier = {
        prefix: combine_backbones({b: fuse_modalities(run.internal(prefix, b), y) for b in BACKBONES})
        for prefix in ("cv", "control", "ablrandinit")
    }
    three_slice_rows = []
    for name, reference, alternative in (
        ("single_slice_vs_three_slices", full, earlier["cv"]),
        ("three_slices_vs_BraTS_only", earlier["cv"], earlier["control"]),
        ("three_slices_vs_random_initialization", earlier["cv"], earlier["ablrandinit"]),
    ):
        rows = compare(y, {name: reference}, {name: alternative})
        for row in rows:
            del row["Holm_p"]
        three_slice_rows.extend(rows)
    write_table(output, "three_slice_comparisons.csv", three_slice_rows)
    return full, earlier["cv"]


def summarize_external(run, output):
    dedup = run.csv("dedup_report.csv")
    cohort_rows, external_rows = [], []
    for site, site_config in run.config["sites"].items():
        records = dedup[dedup.site == site]
        retained = records[~records.excluded]
        labels = np.array(site_config["y"], dtype=int)
        borderline = retained.max_cosine.to_numpy() > 0.90
        if (list(retained.subject_id) != site_config["subject_id"]
                or not np.array_equal(retained.site_label, labels)):
            raise ValueError(f"Deduplication and job configuration disagree for {site}.")
        cohort_rows.append({
            "cohort": site, "screened": len(records), "ID_matches": records.brats21_id_match.sum(),
            "image_matches": records.image_match.sum(), "duplicates_excluded": records.excluded.sum(),
            "retained": len(retained), "retained_methylated": int(labels.sum()),
            "borderline_retained": int(borderline.sum()), "max_retained_similarity": retained.max_cosine.max(),
        })
        sequences = run.external(site)
        archived = run.csv(f"external_preds_{site}.csv")
        if list(archived.subject_id) != list(retained.subject_id) or not np.array_equal(archived.MGMT_value, labels):
            raise ValueError(f"Held-out patient IDs or labels differ for {site}.")
        for estimate, predictions in sequences.items():
            score = rank_average(*predictions.values())
            if estimate == ESTIMATES[0]:
                run.check_scores(site, archived.pred, score)
            for keep, scope in ((np.ones(len(labels), dtype=bool), "all_retained"),
                                (~borderline, "exclude_borderline")):
                auc, low, high = auc_ci(labels[keep], score[keep])
                row = {
                    "cohort": site, "estimate": estimate, "scope": scope, "patients": int(keep.sum()),
                    "AUC": auc, "AUC_CI_low": low, "AUC_CI_high": high,
                    "AP": average_precision_score(labels[keep], score[keep]),
                }
                for modality in MODALITIES:
                    sequence_score = rank_average(*(predictions[b, modality] for b in BACKBONES))
                    row[f"{modality}_AUC"] = roc_auc_score(labels[keep], sequence_score[keep])
                external_rows.append(row)
    write_table(output, "cohort_summary.csv", cohort_rows)
    write_table(output, "external_metrics.csv", external_rows)
    return cohort_rows


def summarize_clinical(run, scores, output):
    labels, folds = run.labels, run.folds
    probabilities = calibrate(scores, labels, folds)
    default = probabilities >= 0.5
    per_fold = np.empty(len(labels), dtype=bool)
    threshold_rows = []
    for fold in np.unique(folds):
        test = folds == fold
        threshold = target_specificity_threshold(labels[~test], probabilities[~test])
        per_fold[test] = probabilities[test] >= threshold
        threshold_rows.append({"fold": int(fold + 1), "threshold": threshold})
    pooled_threshold = target_specificity_threshold(labels, probabilities)
    pooled = probabilities >= pooled_threshold
    rows = [
        {"procedure": "default_probability_0.5", **classification_metrics(labels, default)},
        {"procedure": "manuscript_other_fold_thresholds", **classification_metrics(labels, per_fold)},
        {"procedure": "original_plot_script_pooled_threshold", **classification_metrics(labels, pooled)},
    ]
    write_table(output, "clinical_metrics.csv", rows)
    write_table(output, "high_specificity_thresholds.csv", threshold_rows)

    edges = np.quantile(probabilities, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, probabilities, side="right") - 1, 0, 9)
    bin_rows = []
    for bin_id in np.unique(bins):
        selected = bins == bin_id
        bin_rows.append({
            "bin": int(bin_id + 1), "patients": int(selected.sum()),
            "mean_probability": probabilities[selected].mean(),
            "observed_methylation_rate": labels[selected].mean(),
        })
    write_table(output, "calibration_bins.csv", bin_rows)
    brier = float(np.mean((probabilities - labels) ** 2))
    ece = float(sum(abs(row["mean_probability"] - row["observed_methylation_rate"])
                    * row["patients"] / len(labels) for row in bin_rows))
    (output / "calibration_summary.json").write_text(json.dumps({"Brier": brier, "ECE": ece}, indent=2))

    decisions = []
    for threshold in np.linspace(0.05, 0.95, 91):
        pos = probabilities >= threshold
        tp, fp = np.sum(pos & (labels == 1)), np.sum(pos & (labels == 0))
        odds = threshold / (1 - threshold)
        decisions.append({
            "threshold": threshold, "model": (tp - fp * odds) / len(labels),
            "all_methylated": labels.mean() - (1 - labels.mean()) * odds, "none_methylated": 0.0,
        })
    write_table(output, "decision_curve.csv", decisions)

    confidence = np.abs(probabilities - 0.5)
    ordered = np.argsort(-confidence)
    correct = default == labels
    coverages = (1.0, 0.5, 0.3, 0.2)
    sample_sizes = [round(len(labels) * coverage) for coverage in coverages]
    rng = np.random.default_rng(42)
    boot = np.empty((2000, len(coverages)))
    for i in range(len(boot)):
        sample = rng.integers(0, len(labels), len(labels))
        ranked_sample = sample[np.argsort(-confidence[sample])]
        boot[i] = [correct[ranked_sample[:n]].mean() for n in sample_sizes]
    write_table(output, "selective_prediction.csv", [
        {
            "coverage": coverage, "patients": n, "accuracy": correct[ordered[:n]].mean(),
            "recomputed_bootstrap_CI_low": np.quantile(boot[:, i], 0.025),
            "recomputed_bootstrap_CI_high": np.quantile(boot[:, i], 0.975),
        }
        for i, (coverage, n) in enumerate(zip(coverages, sample_sizes))
    ])
    write_table(output, "calibrated_predictions.csv", {
        "BraTS21ID": run.patient_ids, "fold": folds, "MGMT_value": labels,
        "fused_score": scores, "calibrated_probability": probabilities,
        "default_prediction": default.astype(int), "other_fold_high_specificity_prediction": per_fold.astype(int),
    })
    return rows[1]["specificity"]


def main(archive_path, output):
    output.mkdir(parents=True, exist_ok=True)
    with archive_path.open("rb") as archive_file:
        archive_digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    with ExitStack() as stack:
        run = Run4(*open_run(stack, archive_path))
        full, earlier = summarize_internal(run, output)
        cohorts = summarize_external(run, output)
        high_spec = summarize_clinical(run, full[ESTIMATES[0]], output)
        auc, low, high = auc_ci(run.labels, full[ESTIMATES[0]])
        summary = {
            "archive": archive_path.name,
            "archive_sha256": archive_digest,
            "patients": len(run.labels), "methylated": int(run.labels.sum()),
            "scheduled_jobs_present": run.scheduled_jobs, "archived_jobs": run.archived_jobs,
            "archived_predictions_reconstructed": True,
            "checkpoint_selected_AUC": auc, "checkpoint_selected_AUC_CI": [low, high],
            "fixed_epoch_AUC": roc_auc_score(run.labels, full[ESTIMATES[1]]),
            "three_slice_AUC": {estimate: roc_auc_score(run.labels, score) for estimate, score in earlier.items()},
            "duplicate_exclusions": int(sum(row["duplicates_excluded"] for row in cohorts)),
            "other_fold_high_specificity": high_spec,
        }
    (output / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Checkpoint-selected AUC: {100 * auc:.2f}% (95% CI {100 * low:.2f}–{100 * high:.2f}%)")
    print(f"Fixed-epoch AUC: {100 * summary['fixed_epoch_AUC']:.2f}%")
    print(f"Duplicate exclusions: {summary['duplicate_exclusions']}")
    print(f"Other-fold specificity: {100 * high_spec:.2f}%")
    print(f"Saved results to {output.resolve()}")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Run 4 ZIP or outputs_merged_run4.zip")
    parser.add_argument("--output", type=Path, default=Path("results/rebuilt_run4"))
    options = parser.parse_args()
    main(options.archive, options.output)
