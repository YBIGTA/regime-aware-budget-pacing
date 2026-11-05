AuctionNet paper-guided baseline training (v7)

Run from the project root after extracting:
  bash auctionnet_paper_v7/START.command
The setup first installs an isolated Python environment and runs training smoke checks.
It then launches background data preparation/training/evaluation with a log.
Mac: connect power, leave lid open. Background process uses caffeinate -i.

Today: train official BC and IQL architectures/updates, NOT the earlier gain selector.
Official source pin: alimama-tech/AuctionNet c20de631a40e24669d1d6a8e2acefd8f492f172d.
Vendor Python files are copied unchanged; original LICENSE included.
Source: https://github.com/alimama-tech/AuctionNet
GAS: https://www3.ntu.edu.sg/home/boan/papers/WWW25_GAS.pdf
GUIDE: https://github.com/Tongyun1/GUIDE
This package is an adapted official-baseline experiment, NOT a full reproduction
of GAS/GUIDE or their reported numbers. No generative model is trained here.

Data:
  Official raw final-round data packs 7-8,9-10,11-12,13 downloaded automatically.
  Training: periods 7..12, all 48 advertisers.
  Validation: period 13. Action/reward data from period 13 NEVER enter training.
  Offline rollout validation: advertisers 1,13,25,37 (development-scale validation).
  Periods 14..27 are not downloaded/evaluated; keep them locked for future final tests.
  Streaming aggregation avoids loading full raw CSV into RAM.
  Successful downloads are converted to compact transitions; our compressed archive
  is removed ONLY after successful conversion. Existing auctionnet_lab files untouched.
  Transition artifacts reuse on rerun. At least 12 GiB free space required per new pack.

Models/training:
  Official BC and IQL with original default learning rates/architecture.
  Seeds 42,43,44. Each seed/model: 20000 steps; batch 100.
  BC retains official 3 internal updates per step.
  CPU uses 2 threads; a small CPU-safe baseline, not a GPU requirement.
  Official 16D state, action=sum(bid)/sum(pValue), continuous exposed-pValue reward.
  State dimensions 13,14,15 min-max normalized on training data only;
  reward min-max normalized on training data only.
  Checkpoints every 5000 steps. Model selected on period-13 CPA-penalized score,
  NOT training loss, action imitation MSE or test score.
  Final validation budget ratios: 0.5,0.75,1,1.25,1.5.
  Checkpoints include optimizer and RNG state; rerun same settings to resume.

Explicit adaptations:
  Memory-efficient streaming aggregation matches official generator in synthetic parity test.
  Train/eval use the same normalization (+0.01 denominator).
  Evaluation last-three history uses actual last three; official strategy has a slice bug.
  Below minimum remaining budget, participation is explicitly zero (avoids free wins).
  Fixed-price official OfflineEnv and stochastic conversion evaluation, CPU only.
  These scores cannot be numerically compared to papers using other splits/simulators.
  TE is a secondary diagnostic; BC/IQL optimize conversion reward, NOT uniform pacing.
  Previous PI experiments remain useful but must be evaluated under matched protocol
  before claiming that either approach outperforms the other.

Outputs:
  outputs/install.log and smoke.log: installation/smoke-test diagnostics.
  outputs/overnight.log: preparation/training/checkpoint progress.
  outputs/protocol.json: split/objective/adaptations/source hashes.
  outputs/BC_seed42 (etc.): last.pt, best.pt, training_curve.csv, validation_budget_sweep.csv.
  outputs/validation_summary.csv: all seeds and budget-ratio validation summary.
  outputs/DONE: every scheduled run finished (does NOT mean performance improved).
  outputs/FAILED.txt: error trace. Fix cause and rerun START.command to resume.

Validation performed here:
  Python/shell syntax; actual BC/IQL parameter updates;
  official-generator parity on synthetic tick data;
  end-to-end synthetic train/checkpoint/budget-sweep/report pipeline.
No full raw-data training or performance claim was made in the assistant environment.
