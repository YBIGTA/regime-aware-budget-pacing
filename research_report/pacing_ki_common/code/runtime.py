"""Read-only adapter to the archived B/E2 controller and causal auction engine.

Definitions are loaded from their original AST without executing the old training
commands or their platform-specific import-time hash checks. Source fingerprints
and cross-platform replay differences are recorded by the new experiment.
"""
import ast
import csv
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parents[1]
STATE = REPO / "research_report/pacing_abc_state_pi"
GAIN = REPO / "research_report/pacing_gain_mlp_v2"
ARCHIVE = REPO / "research_report/pacing_abc_A/20261106_132615_119414/results"
sys.path.insert(0, str(REPO / "experiments/pacing_abc"))
import core


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path, normalized=False):
    value = Path(path).read_bytes()
    if normalized:
        value = value.replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


params = read_json(ARCHIVE / "pacing_abc_ki_check_v3/frozen_A.json")["parameters"]
cfg = replace(core.Config(**read_json(ARCHIVE / "pacing_abc_strong_v2/common_data/protocol.json")["config"]),
              initial_multiplier=params["m0"])
scaler, y_mean, y_std, linear, _ = joblib.load(STATE / "code/abc/predictors.joblib")
gain_scale = np.array([params["kp"], params["ki"]])
anchor = read_json(GAIN / "results/anchor.json")["anchor"]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Archived definitions changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


ns = {"np": np}
definitions(STATE / "code/B/experiment2_b.py", ["state_features"], ns)
lib = SimpleNamespace(grid=np.linspace(0, 1, 101), features=core.features, scaler=scaler,
                      y_std=y_std, y_mean=y_mean, gain_scale=gain_scale,
                      state_features=ns["state_features"])
ns.update(lib=lib, cfg=cfg, PREDICTOR=linear, THRESHOLD=0.005)
definitions(GAIN / "code/train_gain_mlp.py", ["base_and_slope", "GainNet", "StatePI"], ns)
StatePI = ns["StatePI"]

TRAIN = list(range(400000, 400080))
HOLDOUT = list(range(400080, 400120))
VALID = list(range(100000, 100200))
PROBE = list(range(600000, 600005))
ALLOWED = set(TRAIN + HOLDOUT + VALID + PROBE)
METRICS = ["tracking_error", "unspent", "objective", "multiplier_tv", "max_ahead", "saturation_fraction"]


def baseline_specs():
    old = read_json(STATE / "code/B/frozen_experiment2.json")
    state = old["B_state_PI"]
    specs = {"B_state_PI": {"anchor": state["anchor"], "weights": state["weights"], "behind_input": False}}
    for label, pick in read_json(GAIN / "results/picks.json")["picks"].items():
        model = read_json(GAIN / f"results/model_{pick['model']}.json")
        specs[f"team_cma_{label}"] = {"anchor": model["anchor"], "weights": model[pick["weights"]], "behind_input": True}
    return specs


def policy(spec):
    return StatePI(spec["anchor"], spec["weights"], spec.get("behind_input", True))


_tapes = {}


def episode(day):
    if day not in ALLOWED or day in cfg.ids("test"):
        raise ValueError(f"Forbidden episode: {day}. Final test remains sealed.")
    if day not in _tapes:
        if len(_tapes) >= 125:
            _tapes.clear()
        _tapes[day] = core.generate(cfg, day)
    return _tapes[day]


def task(item):
    day, specs = item
    ep = episode(day)
    rows = []
    for key, spec in specs:
        score, _ = core.rollout(ep, cfg, policy(spec))
        rows.append({"model": key, **score})
    return rows


def evaluate(pool, specs, days):
    if not set(days) <= ALLOWED:
        raise ValueError("Invalid evaluation split")
    items = list(specs.items())
    groups = [(day, items) for day in days]
    batches = map(task, groups) if pool is None else pool.imap_unordered(task, groups)
    rows = [row for batch in batches for row in batch]
    return sorted(rows, key=lambda r: (str(r["model"]), r["day"]))


def summarize(rows):
    keys = sorted(set(r["model"] for r in rows), key=str)
    return {key: {metric: float(np.mean([r[metric] for r in rows if r["model"] == key])) for metric in METRICS}
            for key in keys}


def spec(weights):
    return {"anchor": anchor, "weights": np.asarray(weights).tolist(), "behind_input": True}


def fingerprint():
    paths = ["experiments/pacing_abc/core.py", "experiments/pacing_abc/run.py",
             "research_report/pacing_abc_state_pi/code/abc/predictors.joblib",
             "research_report/pacing_gain_mlp_v2/code/train_gain_mlp.py",
             "research_report/pacing_abc_state_pi/code/B/experiment2_b.py"]
    return {p: {"sha256": sha(REPO / p), "lf_sha256": sha(REPO / p, True)} for p in paths}
