import os
for name in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
]:
    os.environ.setdefault(name, "2")

import sys
import json
import hashlib
import importlib.util
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
dependency = ROOT / "cluster_v13/domain_clusters.py"

spec = importlib.util.spec_from_file_location(
    "domain_v13_dependency", dependency
)
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)
m = d.m

CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BASE = ROOT / "cluster_v14/outputs" / ("balanced_" + CODE[:8])
INPUT = ROOT / "cluster_v13/outputs/domain_1b178778"
PCA_MODE = False


class BalancedPreprocessor:
    def __init__(self, centered, drop=None):
        self.centered = centered
        self.drop = drop
        self.use_pca = PCA_MODE

    def fit(self, frame):
        self.names = [
            name for name in d.FEATURES
            if d.GROUPS[name] != self.drop
        ]

        values = frame[self.names].to_numpy(float)
        self.centers = (
            frame.groupby("advertiser")[self.names].median()
            if self.centered else None
        )

        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )

        # すべての変換は発見期間のみで学習する。
        self.lo, self.hi = np.quantile(
            values, [.005, .995], axis=0
        )
        clipped = np.clip(values, self.lo, self.hi)
        active = np.where(clipped.std(axis=0) > 1e-10)[0]

        if len(active) < 2:
            raise ValueError("変動のあるフィーチャが2個未満")

        # 도메인 내부의 중복만 제거한다.
        correlation = np.corrcoef(
            clipped[:, active], rowvar=False
        )
        keep = []

        for j, index in enumerate(active):
            redundant = any(
                d.GROUPS[self.names[index]]
                == d.GROUPS[self.names[active[k]]]
                and abs(correlation[j, k]) > .95
                for k in keep
            )
            if not redundant:
                keep.append(j)

        self.indices = active[keep]
        if len(self.indices) < 2:
            raise ValueError("중복 제거 후 피처가 2개 미만")

        self.scaler = StandardScaler().fit(
            clipped[:, self.indices]
        )
        standardized = self.scaler.transform(
            clipped[:, self.indices]
        )

        if not np.allclose(
            standardized.var(axis=0),
            1,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError("표준화 후 피처 분산 검사 실패")

        groups = [
            d.GROUPS[self.names[i]] for i in self.indices
        ]

        # 각 묶음의 제곱거리를 피처 수로 나눈다.
        self.weights = np.array([
            1 / np.sqrt(groups.count(group))
            for group in groups
        ])
        balanced = standardized * self.weights

        # 발견 기간에서 묶음별 총분산이 실제로 1인지 확인.
        for group in sorted(set(groups)):
            mask = np.array(groups) == group
            block_variance = balanced[:, mask].var(axis=0).sum()

            if not np.isclose(block_variance, 1, atol=1e-6):
                raise ValueError(
                    f"묶음 분산 균형 검사 실패: {group}"
                )

        self.pca = None
        if self.use_pca:
            self.pca = PCA(
                n_components=.90,
                svd_solver="full",
            ).fit(balanced)

            if self.pca.n_components_ < 2:
                self.pca = PCA(
                    n_components=2,
                    svd_solver="full",
                ).fit(balanced)

        return self

    def before_pca(self, frame):
        values = frame[self.names].to_numpy(float)

        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )

        clipped = np.clip(
            values, self.lo, self.hi
        )[:, self.indices]

        result = self.scaler.transform(clipped) * self.weights

        if not np.isfinite(result).all():
            raise ValueError("변환 결과에 비유한 값이 있음")

        return result

    def transform(self, frame):
        values = self.before_pca(frame)
        return (
            self.pca.transform(values)
            if self.pca is not None else values
        )

    def audit(self, frame):
        values = frame[self.names].to_numpy(float)
        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )

        transformed = self.before_pca(frame)
        variance = transformed.var(axis=0)

        return pd.DataFrame({
            "feature": [
                self.names[i] for i in self.indices
            ],
            "group": [
                d.GROUPS[self.names[i]] for i in self.indices
            ],
            "std_after_weighting": transformed.std(axis=0),
            "variance": variance,
            "variance_share": (
                variance / max(variance.sum(), 1e-30)
            ),
            "below_tail": (
                values < self.lo
            ).mean(axis=0)[self.indices],
            "above_tail": (
                values > self.hi
            ).mean(axis=0)[self.indices],
        })

    def metadata(self):
        return {
            "centered": self.centered,
            "mode": "PCA90" if self.use_pca else "NO_PCA",
            "kept": [
                self.names[i] for i in self.indices
            ],
            "weights": self.weights,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "lo": self.lo,
            "hi": self.hi,
            "pca_components": (
                self.pca.n_components_
                if self.pca is not None else None
            ),
            "definition": (
                "발견 기간 꼬리 제한 및 StandardScaler. "
                "묶음 내 피처 수의 제곱근으로 나눔. "
                "발견 기간에서 묶음별 총분산 1. "
                "최적 가중치나 MFA의 정확한 구현은 아님."
            ),
        }


def load_data():
    groups = json.loads(
        (INPUT / "feature_groups.json").read_text()
    )
    frame = pd.read_csv(INPUT / "domain_states_raw.csv")

    if (
        len(frame) != 12096
        or set(frame.period) != {16, 17, 18, 19, 20, 21}
    ):
        raise ValueError("입력 행 수·기간 불일치")

    if (
        frame.advertiser.nunique() != 48
        or frame.duplicated(
            ["period", "advertiser", "decision_tick"]
        ).any()
    ):
        raise ValueError("광고주 수·중복 검사 실패")

    if not np.isfinite(
        frame[list(groups)].to_numpy(float)
    ).all():
        raise ValueError("입력에 비유한 값이 있음")

    if not frame.groupby(
        ["period", "advertiser"]
    ).size().eq(42).all():
        raise ValueError("불완전한 에피소드")

    d.GROUPS.clear()
    d.GROUPS.update(groups)
    d.FEATURES.clear()
    d.FEATURES.extend(groups)

    m.FEATURES = d.FEATURES
    m.LOG_FEATURES = []
    m.Preprocessor = BalancedPreprocessor
    m.Preprocess = BalancedPreprocessor
    d.DomainPreprocessor = BalancedPreprocessor

    return frame


def preprocessing_audits(frame, use_pca):
    global PCA_MODE
    PCA_MODE = use_pca

    discovery = frame[
        frame.period.isin([16, 17, 18, 19])
    ]
    transfer = frame[
        frame.period.isin([20, 21])
    ]

    for centered, variant in [
        (False, "market_absolute"),
        (True, "advertiser_centered"),
    ]:
        prep = BalancedPreprocessor(centered).fit(discovery)

        for split, subset in [
            ("discovery", discovery),
            ("transfer", transfer),
        ]:
            audit = prep.audit(subset)
            block = audit.groupby("group")[
                ["variance", "variance_share"]
            ].sum().reset_index()

            m.save_csv(
                f"{variant}/{split}_feature_variance.csv",
                audit,
            )
            m.save_csv(
                f"{variant}/{split}_block_variance.csv",
                block,
            )

            print(
                variant, split, "BLOCK_VARIANCE",
                block.to_dict("records"),
                flush=True,
            )

        m.save_json(
            variant + "/PREPROCESSING_LOCK.json",
            prep.metadata(),
        )


def run():
    global PCA_MODE

    BASE.mkdir(parents=True, exist_ok=True)
    m.OUT = BASE

    import fcntl
    lock = (BASE / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ALREADY RUNNING", BASE)
        return

    if (BASE / "DONE.json").exists():
        print("ALREADY COMPLETE", BASE)
        return

    try:
        import hdbscan
        frame = load_data()

        protocol = {
            "code_sha256": CODE,
            "dependencies": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in [
                    ROOT / "cluster_v13/domain_clusters.py",
                    ROOT / "cluster_v12/cluster_diagnostic.py",
                ]
            },
            "input_sha256": hashlib.sha256(
                (INPUT / "domain_states_raw.csv").read_bytes()
            ).hexdigest(),
            "discovery": [16, 17, 18, 19],
            "internal_transfer": [20, 21],
            "primary": (
                "balanced_no_pca/advertiser_centered"
            ),
            "controls": [
                "absolute levels",
                "PCA90 sensitivity",
            ],
            "distance": (
                "피처를 발견 기간에서 표준화한 뒤 "
                "각 도메인 묶음의 평균 제곱거리를 합산. "
                "발견 기간 묶음별 총분산 1."
            ),
            "tail_policy": (
                "기존 v13의 0.5/99.5% 제한 유지. "
                "보편적 최적값이라는 주장 없음."
            ),
            "selection": (
                "v12 발견 기간 전용 후보 선택 유지. "
                "KMeans k2~6, HDBSCAN 4설정. "
                "이전 기간 결과로 표현을 재선택하지 않음."
            ),
            "scope": (
                "사후 개발 진단. 22~27을 읽지 않음. "
                "군집 수·유의성·정책 개선을 보장하지 않음."
            ),
            "references": [
                "https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html",
                "https://scikit-learn.org/1.5/modules/clustering.html",
                "https://hdbscan.readthedocs.io/en/latest/parameter_selection.html",
            ],
        }
        m.save_json("PROTOCOL.json", protocol)

        results = []
        with threadpool_limits(limits=2):
            for use_pca, label in [
                (False, "balanced_no_pca"),
                (True, "balanced_pca90"),
            ]:
                PCA_MODE = use_pca
                m.OUT = BASE / label
                d.OUT = m.OUT
                m.OUT.mkdir(exist_ok=True)

                m.status("verify_preprocessing")
                preprocessing_audits(frame, use_pca)

                for centered in [False, True]:
                    result = m.analyze(
                        frame, centered, hdbscan
                    )
                    result["representation"] = label
                    results.append(result)

                    current = [
                        item for item in results
                        if (item.representation == label).all()
                    ]
                    m.save_csv(
                        "transfer_summary.csv",
                        pd.concat(current, ignore_index=True),
                    )

                m.status("block_ablation")
                d.extras(frame, hdbscan)
                m.status("all_done", "COMPLETE")

        m.OUT = BASE
        summary = pd.concat(results, ignore_index=True)
        m.save_csv("ALL_transfer_summary.csv", summary)

        render = (
            m.markdown if hasattr(m, "markdown")
            else m.table
        )

        text = (
            "# 분산 균형을 검사한 군집 진단\n\n"
            + render(summary)
            + "\n\n주 분석은 balanced_no_pca/advertiser_centered."
            + " 절대 수준과 PCA90은 민감도 비교다.\n\n"
            + "발견 기간의 각 유효 피처 분산1, "
            + "각 묶음 총분산1을 실제 검사했다. "
            + "더 많은 군집이 생긴다는 보장은 없다.\n\n"
            + "묶음 균등 가중은 연구자의 거리 설계이며 "
            + "최적 가중이나 정확한 MFA 구현이 아니다.\n\n"
            + "꼬리 제한이 실제 충격을 숨길 수 있으므로 "
            + "tail audit와 원시표를 함께 확인한다. "
            + "StandardScaler도 희귀 사건에 민감하다.\n\n"
            + "발견 기간의 분산 균형이 이전 기간에서도 "
            + "유지된다는 보장은 없다. "
            + "이전 기간에 맞춰 재표준화하지 않는다.\n\n"
            + "분리도·noise·공동 inlier 비율·기간 재현성·"
            + "광고주/시간 의존·묶음 제거를 함께 확인한다. "
            + "다른 표현 공간의 silhouette만 직접 비교해 "
            + "가장 높은 결과를 고르지 않는다.\n\n"
            + "군집 분리는 최적 계수 차이나 정책 개선의 "
            + "검증이 아니다. 통계적 유의성을 주장하지 않는다.\n\n"
            + "## 참고\n\n"
            + "\n".join(protocol["references"])
        )

        (BASE / "REPORT.md").write_text(
            text, encoding="utf-8"
        )
        m.save_json("DONE.json", {"status": "COMPLETE"})
        m.status("all_done", "COMPLETE")
        print(text, flush=True)
        print("\nREPORT:", BASE / "REPORT.md", flush=True)

    except Exception:
        m.OUT = BASE
        (BASE / "ERROR.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        m.status("exception", "INCOMPLETE")
        raise


if __name__ == "__main__":
    if "--status" in sys.argv:
        for name in ["STATUS.json", "REPORT.md"]:
            path = BASE / name
            if path.exists():
                print(path.read_text())
    else:
        run()
