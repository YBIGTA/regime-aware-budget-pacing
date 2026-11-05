import os
for key in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
]:
    os.environ.setdefault(key, "2")

import json
import hashlib
import math
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    accuracy_score,
    balanced_accuracy_score,
)
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = HERE / "outputs" / ("diagnostic_" + CODE[:8])

RUN = (
    ROOT / "pacing_v11/auctionnet_lab/runs"
    / "pacing_protocol_v11_1412b714"
)

DISCOVERY = [16, 17, 18, 19]
TRANSFER = [20, 21]
SEED = 20261106

FEATURES = [
    "log_volume", "volume_change", "volume_cv",
    "value_median", "value_spread",
    "price_value_median", "price_value_q90",
    "price_value_change", "price_shift",
    "value_shift", "volume_shift",
]
LOG_FEATURES = [
    "value_median", "value_spread",
    "price_value_median", "price_value_q90",
]

NOTES = []


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            plain(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp.replace(path)


def save_csv(name, frame):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


def status(stage, state="RUNNING"):
    save_json(
        "STATUS.json",
        {
            "status": state,
            "stage": stage,
            "time": datetime.now().isoformat(),
        },
    )
    print(state, stage, flush=True)


def markdown(frame):
    if frame.empty:
        return "해당 결과 없음"

    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "/")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(x) for x in row) + " |")
    return "\n".join(lines)


def input_files():
    path = RUN / "advertisers.json"
    if not path.is_file():
        raise FileNotFoundError(f"v11 결과를 찾을 수 없습니다: {path}")

    advertisers = json.loads(path.read_text())
    if len(advertisers) != 48 or len(set(advertisers)) != 48:
        raise ValueError("광고주 48명 목록과 다릅니다.")

    files = []
    for period in DISCOVERY + TRANSFER:
        for advertiser in advertisers:
            path = (
                RUN / "cache"
                / f"episode_{period}_{advertiser}.joblib"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"기존 에피소드 캐시가 없습니다: {path}"
                )
            files.append((period, advertiser, path))
    return files


def extract(files):
    rows = []
    for index, (period, advertiser, path) in enumerate(files):
        episode = joblib.load(path)

        if (
            episode["period"] != period
            or episode["ad"] != advertiser
            or len(episode["ticks"]) != 48
        ):
            raise ValueError(f"에피소드 정보 불일치: {path}")

        # 결정 직전까지 관측 가능한 시장 정보.
        # 마지막 틱 이후의 정보는 사용하지 않는다.
        for observed in range(47):
            values = np.asarray(
                episode["ticks"][observed]["features"],
                dtype=float,
            )
            if values.shape != (11,) or not np.isfinite(values).all():
                raise ValueError(f"시장 피처가 잘못됐습니다: {path}")

            rows.append({
                "period": period,
                "advertiser": advertiser,
                "decision_tick": observed + 1,
                **dict(zip(FEATURES, values)),
            })

        del episode
        if (index + 1) % 24 == 0:
            print(
                "EXTRACTED", index + 1, "/", len(files),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    if len(frame) != 13536:
        raise ValueError(f"예상 관측 13536행과 다릅니다: {len(frame)}")
    return frame


class Preprocessor:
    def __init__(self, centered):
        self.centered = centered

    def values(self, frame):
        values = frame[FEATURES].copy()
        for name in LOG_FEATURES:
            if (values[name] < 0).any():
                raise ValueError(f"음수 피처: {name}")
            values[name] = np.log(values[name] + 1e-8)
        return values

    def fit(self, frame):
        values = self.values(frame)
        self.centers = None
        if self.centered:
            self.centers = values.groupby(frame.advertiser).median()
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )

        self.scaler = RobustScaler().fit(values)
        scaled = self.scaler.transform(values)
        self.keep = scaled.std(axis=0) > 1e-10
        if self.keep.sum() < 2:
            raise ValueError("변동이 있는 피처가 2개 미만입니다.")
        return self

    def transform(self, frame):
        values = self.values(frame)
        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )
        result = self.scaler.transform(values)[:, self.keep]
        if not np.isfinite(result).all():
            raise ValueError("전처리 결과에 비유한 값이 있습니다.")
        return result


def fit_model(family, parameter, values, hdbscan):
    if family == "kmeans":
        return KMeans(
            n_clusters=parameter["k"],
            n_init=10,
            random_state=42,
        ).fit(values)

    return hdbscan.HDBSCAN(
        min_cluster_size=max(
            30, math.ceil(parameter["fraction"] * len(values))
        ),
        min_samples=parameter["min_samples"],
        prediction_data=True,
        gen_min_span_tree=True,
        core_dist_n_jobs=2,
    ).fit(np.asarray(values, dtype=np.float64))


def predict(model, family, values, hdbscan):
    if family == "kmeans":
        return model.predict(values).astype(int)

    if not np.any(model.labels_ >= 0):
        return np.full(len(values), -1, dtype=int)

    labels, strengths = hdbscan.approximate_predict(
        model, np.asarray(values, dtype=np.float64)
    )
    return labels.astype(int)


def geometry(values, labels):
    mask = labels >= 0
    unique, counts = np.unique(labels[mask], return_counts=True)
    silhouette = None

    if len(unique) >= 2 and mask.sum() > len(unique):
        indices = np.where(mask)[0]
        if len(indices) > 1800:
            indices = np.random.default_rng(SEED).choice(
                indices, 1800, replace=False
            )
        sampled_labels = labels[indices]
        if 2 <= len(np.unique(sampled_labels)) < len(indices):
            silhouette = float(
                silhouette_score(values[indices], sampled_labels)
            )

    return {
        "clusters": len(unique),
        "coverage": float(mask.mean()),
        "noise_fraction": float(1 - mask.mean()),
        "largest_cluster_share": (
            float(counts.max() / len(labels)) if len(counts) else 0
        ),
        "silhouette_assigned": silhouette,
    }


def stability(
    discovery, transfer, reference,
    family, parameter, centered, hdbscan,
):
    rows = []
    for omitted in DISCOVERY:
        subset = discovery[discovery.period.ne(omitted)]

        prep = Preprocessor(centered).fit(subset)
        model = fit_model(
            family, parameter, prep.transform(subset), hdbscan
        )
        labels = predict(
            model, family, prep.transform(transfer), hdbscan
        )

        common = (reference >= 0) & (labels >= 0)
        union = (reference >= 0) | (labels >= 0)

        ari = None
        if (
            common.sum() > 1
            and len(np.unique(reference[common])) >= 2
            and len(np.unique(labels[common])) >= 2
        ):
            ari = float(
                adjusted_rand_score(
                    reference[common], labels[common]
                )
            )

        rows.append({
            "omitted_period": omitted,
            "common_inlier_fraction": float(common.mean()),
            "inlier_jaccard": (
                float(common.sum() / union.sum())
                if union.any() else None
            ),
            "ARI_common_inliers": ari,
        })
    return pd.DataFrame(rows)


def nuisance(discovery, labels, transfer, predicted):
    train_mask = labels >= 0
    check_mask = predicted >= 0
    if not train_mask.any() or not check_mask.any():
        return pd.DataFrame()

    train = discovery.loc[
        train_mask, ["advertiser", "decision_tick"]
    ].copy()
    check = transfer.loc[
        check_mask, ["advertiser", "decision_tick"]
    ].copy()

    train["cluster"] = labels[train_mask]
    check["cluster"] = predicted[check_mask]
    train["time_bin"] = train.decision_tick // 6
    check["time_bin"] = check.decision_tick // 6

    majority = int(train.cluster.mode().iloc[0])
    rows = []

    for name, columns in [
        ("global_majority", []),
        ("advertiser_only", ["advertiser"]),
        ("time_only", ["time_bin"]),
        ("advertiser_and_time", ["advertiser", "time_bin"]),
    ]:
        if not columns:
            output = np.full(len(check), majority)
        else:
            mapping = train.groupby(columns).cluster.agg(
                lambda group: int(group.mode().iloc[0])
            )
            if len(columns) == 1:
                output = (
                    check[columns[0]]
                    .map(mapping)
                    .fillna(majority)
                    .to_numpy(int)
                )
            else:
                output = np.array([
                    mapping.get(tuple(row), majority)
                    for row in check[columns].to_numpy()
                ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced = balanced_accuracy_score(
                check.cluster, output
            )

        rows.append({
            "predictor": name,
            "evaluated_inliers": len(check),
            "accuracy": accuracy_score(check.cluster, output),
            "balanced_accuracy": balanced,
        })

    return pd.DataFrame(rows)


def plots(frame, coordinates, labels, folder):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        NOTES.append("matplotlib 미설치: 그림 대신 PCA CSV를 확인하세요.")
        return

    indices = np.random.default_rng(SEED).choice(
        len(frame), min(2500, len(frame)), replace=False
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for axis, colors, title in zip(
        axes,
        [
            labels[indices],
            frame.advertiser.to_numpy()[indices],
            frame.decision_tick.to_numpy()[indices],
        ],
        ["Cluster (-1=noise)", "Advertiser", "Decision tick"],
    ):
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            c=colors,
            s=5,
            alpha=.6,
            cmap="tab20",
        )
        axis.set_title(title)
        axis.set_xlabel("Discovery PCA 1")
        axis.set_ylabel("Discovery PCA 2")

    fig.suptitle("Visualization only; metrics use original feature space")
    fig.tight_layout()
    fig.savefig(folder / "pca_comparison.png", dpi=160)
    plt.close(fig)

    view = frame[
        ["period", "advertiser", "decision_tick"]
    ].copy()
    view["cluster"] = labels

    for period in TRANSFER:
        matrix = view[view.period.eq(period)].pivot(
            index="advertiser",
            columns="decision_tick",
            values="cluster",
        )
        fig, axis = plt.subplots(figsize=(12, 7))
        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="tab20",
            vmin=-1,
            vmax=max(1, int(labels.max())),
        )
        axis.set_title(f"Period {period}; cluster -1 = noise")
        axis.set_xlabel("Decision ticks 1..47")
        axis.set_ylabel("Advertiser, sorted ID")
        fig.colorbar(image, ax=axis, label="Cluster")
        fig.tight_layout()
        fig.savefig(
            folder / f"period_{period}_heatmap.png",
            dpi=160,
        )
        plt.close(fig)


def analyze(frame, centered, hdbscan):
    variant = (
        "advertiser_centered"
        if centered else "market_absolute"
    )
    status(variant)

    discovery = frame[
        frame.period.isin(DISCOVERY)
    ].reset_index(drop=True)
    transfer = frame[
        frame.period.isin(TRANSFER)
    ].reset_index(drop=True)

    prep = Preprocessor(centered).fit(discovery)
    train_x = prep.transform(discovery)
    check_x = prep.transform(transfer)

    candidates = []
    fitted = {}

    configurations = [
        ("kmeans", {"k": k}) for k in [2, 3, 4, 5, 6]
    ]
    if hdbscan is not None:
        configurations += [
            (
                "hdbscan",
                {"fraction": fraction, "min_samples": minimum},
            )
            for fraction in [.02, .05]
            for minimum in [10, 30]
        ]

    for family, parameter in configurations:
        key = family + "_" + "_".join(
            str(value) for value in parameter.values()
        )
        print("FIT", variant, key, flush=True)

        model = fit_model(
            family, parameter, train_x, hdbscan
        )
        labels = model.labels_.astype(int)
        row = {
            "key": key,
            "family": family,
            "parameter": json.dumps(parameter),
            **geometry(train_x, labels),
        }
        candidates.append(row)
        fitted[key] = (model, parameter, labels)

        save_csv(
            f"{variant}/discovery_candidates.csv",
            pd.DataFrame(candidates),
        )

    selected = {}
    for family in ["kmeans", "hdbscan"]:
        options = [
            row for row in candidates if row["family"] == family
        ]
        if not options:
            continue

        eligible = [
            row for row in options
            if row["clusters"] >= 2
            and row["coverage"] >= .5
            and row["silhouette_assigned"] is not None
        ]

        if eligible:
            chosen = max(
                eligible,
                key=lambda row: (
                    (
                        row["silhouette_assigned"]
                        if family == "kmeans"
                        else max(
                            0, row["silhouette_assigned"]
                        ) * row["coverage"]
                    ),
                    -row["clusters"],
                ),
            )
        else:
            chosen = max(
                options,
                key=lambda row: (
                    row["coverage"], row["clusters"]
                ),
            )

        selected[family] = chosen

    # 이전 확인 기간의 결과를 계산하기 전에 선택을 기록.
    save_json(
        f"{variant}/LOCKED_DISCOVERY_SELECTION.json",
        selected,
    )

    pca = PCA(n_components=2).fit(train_x)
    check_coordinates = pca.transform(check_x)
    save_json(
        f"{variant}/pca_variance.json",
        pca.explained_variance_ratio_,
    )

    reports = []

    for family, chosen in selected.items():
        model, parameter, labels = fitted[chosen["key"]]
        predicted = predict(
            model, family, check_x, hdbscan
        )

        relative = f"{variant}/{family}"
        folder = OUT / relative
        folder.mkdir(parents=True, exist_ok=True)

        assignment = pd.concat([
            discovery.assign(
                split="discovery", cluster=labels
            ),
            transfer.assign(
                split="transfer", cluster=predicted
            ),
        ], ignore_index=True)
        save_csv(f"{relative}/assignments.csv", assignment)

        joblib.dump(
            {
                "preprocessor": prep,
                "model": model,
                "features": FEATURES,
            },
            folder / "model.joblib",
            compress=3,
        )

        stability_frame = stability(
            discovery, transfer, predicted,
            family, parameter, centered, hdbscan,
        )
        save_csv(
            f"{relative}/leave_period_out_stability.csv",
            stability_frame,
        )

        nuisance_frame = nuisance(
            discovery, labels, transfer, predicted
        )
        save_csv(
            f"{relative}/nuisance_prediction.csv",
            nuisance_frame,
        )

        profiles = (
            assignment[assignment.cluster.ge(0)]
            .groupby(["split", "cluster"])[FEATURES]
            .median()
            .reset_index()
        )
        save_csv(
            f"{relative}/profiles_raw_median.csv",
            profiles,
        )

        counts = (
            assignment
            .groupby(["split", "period", "cluster"])
            .size()
            .rename("rows")
            .reset_index()
        )
        save_csv(
            f"{relative}/cluster_counts.csv",
            counts,
        )

        transitions = []
        for (period, advertiser), group in transfer.assign(
            cluster=predicted
        ).groupby(["period", "advertiser"]):
            sequence = group.sort_values(
                "decision_tick"
            ).cluster.to_numpy()
            valid = (
                (sequence[:-1] >= 0)
                & (sequence[1:] >= 0)
            )
            transitions.append({
                "period": period,
                "advertiser": advertiser,
                "distinct_clusters": len(
                    np.unique(sequence[sequence >= 0])
                ),
                "inlier_fraction": float(
                    (sequence >= 0).mean()
                ),
                "valid_adjacent_pairs": int(valid.sum()),
                "switch_rate_assigned": (
                    float(
                        (
                            sequence[:-1][valid]
                            != sequence[1:][valid]
                        ).mean()
                    )
                    if valid.any() else None
                ),
            })
        save_csv(
            f"{relative}/temporal_transitions.csv",
            pd.DataFrame(transitions),
        )

        projection = transfer[
            ["period", "advertiser", "decision_tick"]
        ].copy()
        projection["pc1"] = check_coordinates[:, 0]
        projection["pc2"] = check_coordinates[:, 1]
        projection["cluster"] = predicted
        save_csv(
            f"{relative}/pca_projection.csv",
            projection,
        )
        plots(
            transfer, check_coordinates, predicted, folder
        )

        local_rows = []
        for period in TRANSFER:
            mask = transfer.period.eq(period).to_numpy()
            inliers = mask & (predicted >= 0)

            association = {
                "advertiser_AMI": None,
                "time_bin_AMI": None,
            }
            if (
                inliers.sum() > 1
                and len(np.unique(predicted[inliers])) >= 2
            ):
                association["advertiser_AMI"] = (
                    adjusted_mutual_info_score(
                        transfer.advertiser.to_numpy()[inliers],
                        predicted[inliers],
                    )
                )
                association["time_bin_AMI"] = (
                    adjusted_mutual_info_score(
                        (
                            transfer.decision_tick.to_numpy()
                            // 6
                        )[inliers],
                        predicted[inliers],
                    )
                )

            row = {
                "variant": variant,
                "family": family,
                "period": period,
                **geometry(
                    check_x[mask], predicted[mask]
                ),
                **association,
                "mean_LOPO_ARI": (
                    float(
                        stability_frame
                        .ARI_common_inliers.mean()
                    )
                    if stability_frame
                    .ARI_common_inliers.notna().any()
                    else None
                ),
                "mean_LOPO_common_inlier_fraction": float(
                    stability_frame
                    .common_inlier_fraction.mean()
                ),
            }
            reports.append(row)
            local_rows.append(row)

        text = [
            f"# {variant} / {family}",
            "\n## 발견 기간에서 선택한 설정\n",
            "```json\n"
            + json.dumps(
                plain(chosen),
                ensure_ascii=False,
                indent=2,
            )
            + "\n```",
            "\n## 다른 기간에서의 분리도\n",
            markdown(pd.DataFrame(local_rows)),
            "\n## 발견 기간 하나씩 제외한 재현성\n",
            markdown(stability_frame),
            "\n## 광고주·시간 정보만으로 군집 예측\n",
            markdown(nuisance_frame),
        ]
        (folder / "REPORT.md").write_text(
            "\n".join(text), encoding="utf-8"
        )

    return pd.DataFrame(reports)


def worker():
    status("extract_training_features")
    files = input_files()
    frame = extract(files)
    save_csv("market_states_raw.csv", frame)

    try:
        import hdbscan
    except ImportError:
        hdbscan = None
        NOTES.append(
            "hdbscan 미설치: KMeans만 실행. "
            "밀도 군집 진단은 미완료."
        )

    results = []
    with threadpool_limits(limits=2):
        for centered in [False, True]:
            results.append(
                analyze(frame, centered, hdbscan)
            )
            save_csv(
                "transfer_summary.csv",
                pd.concat(results, ignore_index=True),
            )

    summary = pd.concat(results, ignore_index=True)
    report = [
        "# 시장 상태 군집화 진단",
        "",
        "발견: 기간16~19. 내부 기간 이전 확인:20~21.",
        "광고주48명, 에피소드별47개 과거 관측.",
        "최종 평가 기간24~27은 읽지 않았다.",
        "",
        "## 기간 이전 결과",
        "",
        markdown(summary),
        "",
        "## 지표를 읽는 방법",
        "",
        "- clusters: noise(-1)를 제외한 군집 수.",
        "- coverage: 군집에 할당된 비율. 낮으면 "
        "일부 점만 묶은 결과다.",
        "- silhouette_assigned: 원래 피처 공간에서 "
        "계산한 할당 표본의 분리도. "
        "높을수록 분리되어 있지만 자연 레짐의 증명은 아니다.",
        "- mean_LOPO_ARI: 발견 기간 하나를 빼고 "
        "전처리부터 재학습해도 같은 이전 표본을 "
        "비슷하게 묶는지 확인한 값. "
        "공통 inlier coverage와 함께 읽는다.",
        "- ARI가 NA면 공통 표본에서 비교할 "
        "두 개 이상의 군집이 없는 것이며 성공이 아니다.",
        "- advertiser_AMI/time_bin_AMI: "
        "군집과 광고주·시간대의 연관성. "
        "높으면 시장 상태 대신 광고주/시간대 "
        "구분일 가능성을 점검해야 한다.",
        "",
        "## 필수 추가 확인",
        "",
        "각 방법 폴더의 nuisance_prediction.csv에서 "
        "광고주·시간만 사용하는 분류가 군집을 "
        "얼마나 재현하는지 확인한다.",
        "",
        "market_absolute는 시장 피처의 절대 수준, "
        "advertiser_centered는 발견 기간의 광고주별 "
        "중앙값을 뺀 피처다. "
        "중심화 후 구조가 사라지면 광고주 고유 수준이 "
        "분리를 주도했을 수 있다. "
        "중심화 자체가 정답이라는 뜻은 아니다.",
        "",
        "profiles_raw_median.csv로 군집별 가격·물량·"
        "전환가치 차이를 확인하고, "
        "temporal_transitions.csv와 heatmap으로 "
        "같은 광고주 안에서 상태 전환이 있는지 확인한다.",
        "",
        "## 해석 제한",
        "",
        "KMeans는 연속 분포도 지정 개수로 나눈다. "
        "HDBSCAN의 높은 silhouette는 "
        "noise 비율과 함께 해석한다. "
        "단일 점수 문턱으로 명확한 레짐을 확정하지 않는다.",
        "",
        "군집이 잘 나뉘어도 서로 다른 최적 PI계수가 "
        "필요하다거나 정책 성능이 좋아진다는 뜻은 아니다.",
        "",
        "13536행은 독립 시장13536개가 아니다. "
        "인접 틱·같은 기간·같은 광고주가 반복된다. "
        "통계적 유의성을 주장하지 않는다.",
        "",
        "20~21은 전체 연구에서 이미 사용한 훈련 기간이다. "
        "새로운 최종 테스트가 아니라 "
        "이번 군집화의 내부 기간 이전 진단이다.",
        "",
        "PCA2차원 그림은 설명용이다. "
        "분리도는 원래 피처 공간에서 계산한다.",
        "",
        "## 주의·누락",
        "",
        "\n".join(dict.fromkeys(NOTES))
        or "필수 수치 진단 완료.",
        "",
        "## 참고",
        "",
        "https://hdbscan.readthedocs.io/en/latest/parameter_selection.html",
        "",
        "https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html",
        "",
        "https://scikit-learn.org/1.5/modules/clustering.html",
    ]
    (OUT / "REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    save_json(
        "DONE.json",
        {
            "status": "COMPLETE",
            "rows": len(frame),
            "hdbscan_available": hdbscan is not None,
            "notes": NOTES,
        },
    )
    status("all_done", "COMPLETE")
    print("\n" + markdown(summary), flush=True)
    print("\nREPORT:", OUT / "REPORT.md", flush=True)


def main():
    if "--status" in sys.argv:
        for name in ["STATUS.json", "REPORT.md"]:
            path = OUT / name
            if path.exists():
                print(path.read_text(encoding="utf-8"))
        return

    OUT.mkdir(parents=True, exist_ok=True)

    import fcntl
    lock = (OUT / "run.lock").open("a")
    try:
        fcntl.flock(
            lock,
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        print("ALREADY RUNNING:", OUT)
        return

    if (OUT / "DONE.json").exists():
        print("ALREADY COMPLETE:", OUT)
        return

    if "--worker" in sys.argv:
        try:
            worker()
        except Exception:
            error = traceback.format_exc()
            (OUT / "ERROR.txt").write_text(
                error, encoding="utf-8"
            )
            status("exception", "INCOMPLETE")
            print(error, flush=True)
            raise
        return

    files = input_files()

    # 기존 캐시 및 설치된 HDBSCAN 경로 사전 확인.
    sample = joblib.load(files[0][2])
    if len(sample["ticks"][0]["features"]) != 11:
        raise ValueError("시장 피처 사전 검사 실패")
    del sample

    try:
        import hdbscan
    except ImportError:
        print(
            "주의: HDBSCAN 미설치. "
            "KMeans만 실행하고 미완료 항목을 기록합니다."
        )
    else:
        rng = np.random.default_rng(42)
        tiny = np.r_[
            rng.normal(-3, .1, (40, 3)),
            rng.normal(3, .1, (40, 3)),
        ]
        model = fit_model(
            "hdbscan",
            {"fraction": .02, "min_samples": 10},
            tiny,
            hdbscan,
        )
        predict(model, "hdbscan", tiny[:2], hdbscan)

    save_json(
        "PROTOCOL.json",
        {
            "code_sha256": CODE,
            "discovery": DISCOVERY,
            "internal_transfer": TRANSFER,
            "excluded_final_periods": [24, 25, 26, 27],
            "advertisers": 48,
            "features": FEATURES,
            "variants": [
                "market_absolute",
                "advertiser_centered",
            ],
            "kmeans_k": [2, 3, 4, 5, 6],
            "hdbscan_min_cluster_fractions": [.02, .05],
            "hdbscan_min_samples": [10, 30],
            "selection": (
                "발견 기간에서만 선택. "
                "KMeans는 silhouette, "
                "HDBSCAN은 coverage*max(0,silhouette). "
                "2군집 이상·coverage .5 이상 후보를 우선. "
                "없으면 진단용 후보만 남기고 성공으로 간주하지 않음."
            ),
            "scope": (
                "사후 군집 탐색. 정책 개선·"
                "통계적 유의성을 검정하지 않음."
            ),
        },
    )

    status("launching")
    command = [
        sys.executable, "-u",
        str(Path(__file__).resolve()),
        "--worker",
    ]

    import shutil
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-i"] + command

    log = (OUT / "overnight.log").open("a")
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print("PREFLIGHT PASSED; STARTED PID:", process.pid)
    print("OUTPUT:", OUT)
    print("LOG:", OUT / "overnight.log")


if __name__ == "__main__":
    main()
