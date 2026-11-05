"""Read-only checkpoint diagnostics using the laboratory data and metrics paths.

No optimizer, training loop, data split, or replacement classification metric.
Frozen-readout and branch interventions are diagnostic, not retrained ablations.
"""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import config
from src.config import ConfigManager
from src.data import getMyData, set_random_seed
from src.metrics import getMyReport
from src.model_registry import get_model_adapter
from src.models.transmixer_paper.model import NeighborTransMixer
from scripts.prepare_sr_native import report_metrics

TAG = "sr_lg_mechanisms_20261104"
DEST = ROOT / "report" / TAG
NATIVE = ROOT / "report/sr_native/sr_native_20261029"
GROUPS = ("M0", "M1", "M2", "M3", "M4")
SEEDS = (6, 7, 8)
DEVICE = "cuda:0"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def native_report(net, adapter, loader, path, model):
    if path.exists():
        raise FileExistsError(path)
    with contextlib.redirect_stdout(sys.stdout):
        getMyReport(0, {"net": net}, adapter, loader, str(path), DEVICE, model)
    return report_metrics(path)


def confusion(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    start = text.rindex("[[")
    end = text.index("]]", start) + 2
    values = np.array([int(v) for v in re.findall(r"\d+", text[start:end])])
    assert len(values) == 225
    return values.reshape(15, 15)


class Probe:
    def __init__(self, adapter, net, sr_settings):
        self.adapter, self.net = adapter, net
        self.logits, self.labels, self.variants = [], [], {}
        self.offset = 0
        self.current_fused = None
        self.sr = SimpleNamespace(variant="SR2", spatial_sigma=3.,
            spectral_scale=torch.tensor(sr_settings["spectral_scale"], device=DEVICE))
        self.spatial = SimpleNamespace(variant="SR1", spatial_sigma=3.)
        # Fixed before observing diagnostics, no fixed points, same for all seeds.
        self.permutation = torch.roll(torch.arange(121, device=DEVICE), 37)
        self.weight_records = []
        self.branch_records = []
        self.handles = [net.fusion_conv.register_forward_hook(self.capture_fused)]
        for index, block in enumerate(net.blocks):
            if block.mode == "decoupled":
                self.handles.append(block.fusion.register_forward_pre_hook(self.branch_hook(index)))

    def capture_fused(self, module, inputs, output):
        self.current_fused = output

    def branch_hook(self, index):
        def hook(module, inputs):
            x = inputs[0].detach()
            self.branch_records.append({"block": index, "batch_n": len(x),
                "global_energy_sum": float(x[:, :, :32].square().mean((1, 2)).sum()),
                "local_energy_sum": float(x[:, :, 32:].square().mean((1, 2)).sum())})
        return hook

    def forward_eval(self, bundle, batch):
        outputs = self.adapter.forward_eval(bundle, batch)
        self.logits.append(outputs.detach().cpu())
        self.labels.append(batch["label"].detach().cpu())
        fused = self.current_fused
        w = NeighborTransMixer.neighbor_weights(self.sr, batch["hsi_pca"], batch["aux"])
        spatial = NeighborTransMixer.neighbor_weights(self.spatial, batch["hsi_pca"], batch["aux"])
        wrong = w.flatten(2)[:, :, self.permutation].reshape_as(w)
        center = torch.zeros_like(w); center[:, :, 5, 5] = 1.
        pools = {"uniform": fused.mean((2, 3)), "correct_sr": (fused*w).sum((2, 3)),
                 "distance_only": (fused*spatial).sum((2, 3)),
                 "misaligned_sr": (fused*wrong).sum((2, 3)),
                 "center_only": (fused*center).sum((2, 3))}
        for name, pooled in pools.items():
            logits = self.net.classifier(pooled)
            self.variants.setdefault(name, []).append(logits.detach().cpu())
        if self.net.mode == "pure_ssm" and hasattr(self.net, "neighbor_weights"):
            # Collected once per seed (M2), input-derived and model-independent.
            torch.testing.assert_close(w, self.net.gap.weights, atol=0, rtol=0)
            self.weight_records.append((w.detach().cpu().numpy(), spatial.detach().cpu().numpy()))
        self.offset += len(outputs)
        if self.offset % 3072 == 0:
            print(f"  native inference {self.offset}/12197", flush=True)
        return outputs

    def close(self):
        for handle in self.handles: handle.remove()


class Replay:
    """Return cached logits to ORIGINAL metrics while iterating ORIGINAL loader."""
    def __init__(self, logits, labels):
        self.logits, self.labels, self.cursor = logits, labels, 0

    def forward_eval(self, bundle, batch):
        count = len(batch["label"])
        sl = slice(self.cursor, self.cursor + count)
        assert torch.equal(self.labels[sl], batch["label"].cpu())
        self.cursor += count
        return self.logits[sl].to(DEVICE)


def sample_metadata(dataset, weights, spatial):
    pos, gt = dataset.pos, dataset.gt
    padded = np.pad(gt, 5, mode="constant", constant_values=0)
    neighbors = np.stack([padded[r:r+11, c:c+11].reshape(-1) for r, c in pos])
    target = gt[pos[:, 0], pos[:, 1]]
    known = neighbors > 0
    known[:, 60] = False  # Do not count the known center as supporting evidence.
    same = (neighbors == target[:, None]) & known
    other = known & ~same
    n_known, n_other = known.sum(1), other.sum(1)
    w, s = weights.reshape(-1, 121), spatial.reshape(-1, 121)
    denom_w = (w*known).sum(1)
    denom_s = (s*known).sum(1)
    cov = n_known / 120
    fraction = np.divide(n_other, n_known, out=np.zeros(len(pos)), where=n_known>0)
    return {"row": pos[:, 0], "col": pos[:, 1], "label": target-1,
        "known_neighbors": n_known, "known_fraction": cov,
        "observed_other_fraction": fraction,
        "mixed": n_other > 0,
        "no_observed_other_sufficient": (n_other == 0) & (n_known >= 30),
        "low_coverage": n_known < 30,
        "center_weight": w[:, 60], "effective_neighbors": 1/(w*w).sum(1),
        "entropy_normalized": -(w*np.log(w.clip(1e-30))).sum(1)/np.log(121),
        "other_weight_fraction_sr": np.divide((w*other).sum(1), denom_w, out=np.zeros(len(pos)), where=denom_w>0),
        "other_weight_fraction_distance": np.divide((s*other).sum(1), denom_s, out=np.zeros(len(pos)), where=denom_s>0),
        "other_weight_fraction_uniform": fraction,
        "known_weight_sr": denom_w}


def summarize_weights(meta):
    eligible = meta["known_neighbors"] >= 30
    mixed = eligible & meta["mixed"]
    summary = {"eligible_n": int(eligible.sum()), "eligible_mixed_n": int(mixed.sum()),
               "observed_mixed_total": int(meta["mixed"].sum())}
    for name in ("center_weight", "effective_neighbors", "entropy_normalized", "known_fraction"):
        a = meta[name]
        summary[name] = {"mean": float(a.mean()), "median": float(np.median(a)),
                         "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95))}
    for name in ("other_weight_fraction_sr", "other_weight_fraction_distance", "other_weight_fraction_uniform"):
        summary[name+"_eligible_mixed_mean"] = float(meta[name][mixed].mean()) if mixed.any() else None
    summary["sr_less_other_mass_than_distance_fraction"] = float((meta["other_weight_fraction_sr"][mixed] < meta["other_weight_fraction_distance"][mixed]).mean()) if mixed.any() else None
    return summary


def run_seed(seed, jobs, seals):
    out = DEST / f"seed{seed}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    set_random_seed(seed)
    loaders = getMyData(0, 30, 11, 128, 0)
    loader = loaders[1]
    assert len(loader.dataset) == 12197 and not loader.drop_last and loader.batch_size == 128
    assert not loader.dataset.train
    pos = loader.dataset.pos
    input_digest = sha(ROOT / "configs/runs/sr_native_20261029" / f"M3_seed{seed}.json")
    all_preds, label_array, rows, norms = {}, None, [], []
    meta = None
    settings = read(ROOT / "configs/runs/sr_native_20261029" / f"M3_seed{seed}.json")["sr"]
    for group in GROUPS:
        identifier = f"{group}_seed{seed}"
        job = next(j for j in jobs if j["id"] == identifier)
        seal = next(s for s in seals if s["id"] == identifier)
        for kind in ("checkpoint", "report"):
            assert sha(job[kind]) == seal[kind+"_sha256"], (identifier, kind)
        manager = ConfigManager(); manager.load_config(job["config"])
        config._init(manager.config)
        net = torch.load(job["checkpoint"], map_location=DEVICE, weights_only=False).eval()
        initial_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        adapter = get_model_adapter(job["model"])
        print(f"START {identifier} {now()}", flush=True)
        probe = Probe(adapter, net, settings)
        try:
            original = native_report(net, probe, loader, out / f"{group}_native.txt", job["model"])
        finally:
            probe.close()
        assert all(torch.equal(v, net.state_dict()[k]) for k,v in initial_state.items())
        expected = report_metrics(job["report"])
        difference = {k: float(original[k]-expected[k]) for k in ("oa", "aa", "kappa")}
        historical_matrix = confusion(job["report"])
        replay_matrix = confusion(out / f"{group}_native.txt")
        matrix_equal = np.array_equal(historical_matrix, replay_matrix)
        write(out / f"{group}_replay_audit.json", {"metrics_difference":difference,
            "confusion_exact":bool(matrix_equal), "confusion_difference":(replay_matrix-historical_matrix).tolist(),
            "prediction_provenance":"current native checkpoint replay; historical per-sample predictions unavailable"})
        # Collect diagnostic replays even when floating point sensitivity changes
        # historical predictions. Review every metric drift before interpreting.
        if not np.array_equal(historical_matrix.sum(1), replay_matrix.sum(1)):
            write(out / f"{group}_replay_mismatch.json", {"expected":expected, "actual":original, "difference":difference})
            raise RuntimeError(f"Original checkpoint replay differs: {identifier}: {difference}")
        logits = torch.cat(probe.logits)
        labels = torch.cat(probe.labels)
        if label_array is None: label_array = labels.numpy()
        assert np.array_equal(label_array, labels.numpy())
        all_preds[group] = logits.argmax(1).numpy()
        variants = {name: torch.cat(parts) for name, parts in probe.variants.items()}
        matched = "correct_sr" if group in ("M2", "M3") else "distance_only" if group == "M4" else "uniform"
        torch.testing.assert_close(logits, variants[matched], atol=2e-5, rtol=2e-5)
        assert torch.equal(logits.argmax(1), variants[matched].argmax(1))
        rows.append({"seed":seed, "group":group, "intervention":"native", **original})
        for name, values in variants.items():
            all_preds[f"{group}_{name}"] = values.argmax(1).numpy()
            if name == matched: continue
            replay = Replay(values, labels)
            result = native_report(net, replay, loader, out / f"{group}_pool_{name}.txt", job["model"])
            assert replay.cursor == len(loader.dataset)
            rows.append({"seed":seed, "group":group, "intervention":"pool_"+name, **result})
        if group == "M2":
            w = np.concatenate([r[0] for r in probe.weight_records])
            spatial = np.concatenate([r[1] for r in probe.weight_records])
            meta = sample_metadata(loader.dataset, w, spatial)
            np.savez_compressed(out / "neighbor_metadata.npz", **meta)
            write(out / "weight_summary.json", summarize_weights(meta))
            # Keep all sample weights: small diagnostic artifact, no raw inputs.
            np.savez_compressed(out / "sr_weights.npz", weights=w[:,0], spatial=spatial[:,0])
        if group in ("M1", "M3"):
            norms.extend({"seed":seed, "group":group, **r} for r in probe.branch_records)
            for disabled in ("global", "local"):
                collected = []
                class Intervention:
                    def forward_eval(self, bundle, batch):
                        outputs = adapter.forward_eval(bundle, batch)
                        collected.append(outputs.detach().cpu())
                        return outputs
                def mute(module, inputs):
                    x = inputs[0].clone()
                    if disabled == "global": x[:, :, :32] = 0
                    else: x[:, :, 32:] = 0
                    return (x,)
                handles = [b.fusion.register_forward_pre_hook(mute) for b in net.blocks]
                try:
                    result = native_report(net, Intervention(), loader, out / f"{group}_mute_{disabled}.txt", job["model"])
                finally:
                    for handle in handles: handle.remove()
                all_preds[f"{group}_mute_{disabled}"] = torch.cat(collected).argmax(1).numpy()
                rows.append({"seed":seed, "group":group, "intervention":"mute_"+disabled, **result})
            assert all(torch.equal(v, net.state_dict()[k]) for k,v in initial_state.items())
        write(out / "progress.json", {"finished_group":group, "time":now(), "metrics":rows})
        print(f"DONE {identifier} exact report replay; {now()}", flush=True)
        del net, initial_state, logits, variants, probe
        gc.collect(); torch.cuda.empty_cache()
    np.savez_compressed(out / "predictions.npz", labels=label_array, row=pos[:,0], col=pos[:,1], **all_preds)
    write(out / "metrics.json", rows)
    write(out / "branch_energy.json", norms)
    csv_write(out / "metrics.csv", rows)
    return {"seed":seed, "completed_at":now(), "config_sha256":input_digest, "reports":len(rows)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume-analysis", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    manifest = read(NATIVE / "manifest.json")
    seals = read(NATIVE / "final_integrity_audit.json")["outputs"]
    # Analysis provenance freezes current bytes; historical model/data/config are checked separately.
    selected = [j for j in manifest["jobs"] if j["group"] in GROUPS]
    if not (DEST / "manifest.json").exists():
        DEST.mkdir(parents=True, exist_ok=True)
        files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT/"configs/train.json", ROOT/"AGENTS.md"]
        hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
        for rel, digest in manifest["data_sha256"].items(): assert sha(ROOT/rel)==digest, rel
        for job in selected:
            for kind in ("checkpoint", "report"):
                seal = next(s for s in seals if s["id"]==job["id"])
                assert sha(job[kind]) == seal[kind+"_sha256"]
            rel = str(Path(job["config"]).relative_to(ROOT)).replace("\\", "/")
            assert sha(job["config"]) == manifest["source_and_config_sha256"][rel]
        write(DEST / "manifest.json", {"created_at":now(), "tag":TAG, "training_runs":0,
            "checkpoint_selection":"historical native test-OA best; unchanged",
            "seeds":list(SEEDS), "groups":list(GROUPS), "source_sha256":hashes,
            "data_sha256":manifest["data_sha256"], "python":sys.executable,
            "torch":torch.__version__, "numpy":np.__version__, "platform":platform.platform(),
            "interventions":"fixed checkpoint; no optimizer, weight updates, labels in forward, or retraining",
            "label_strata":"posthoc observed labels only; minimum 30 of 120 non-center neighbors for supported strata",
            "permutation":"cyclic shift 37 of 121 positions; no fixed points; one fixed diagnostic control",
            "metric_entry":"src.metrics.getMyReport; cached logits replayed using original native loader"})
    completed = []
    for seed in ([args.seed] if args.seed else SEEDS):
        if args.resume_analysis and (DEST / f"seed{seed}" / "metrics.json").exists():
            completed.append({"seed":seed,"reused_completed_analysis":True})
            continue
        result = run_seed(seed, selected, seals)
        completed.append(result)
        write(DEST / "progress.json", {"state":"running", "seeds_completed_this_run":completed, "time":now()})
    for rel,digest in read(DEST/"manifest.json")["source_sha256"].items(): assert sha(ROOT/rel)==digest, rel
    all_complete = all((DEST/f"seed{s}/metrics.json").exists() for s in SEEDS)
    write(DEST / "progress.json", {"state":"completed" if all_complete else "partial", "seeds_completed_this_run":completed, "time":now()})


if __name__ == "__main__":
    main()
