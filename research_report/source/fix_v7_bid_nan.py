from pathlib import Path
from datetime import datetime
import shutil

path = Path("auctionnet_paper_v7/overnight.py")
s = path.read_text()
if "V7_BID_NAN_QUARANTINE" not in s:
    edits = [("                        if not np.isfinite(chunk[RAW].to_numpy(dtype=float)).all():\n                            raise ValueError('Nonfinite raw values')", "                        # V7_BID_NAN_QUARANTINE: never invent missing action labels.\n                        checked = chunk[RAW].to_numpy(dtype=float)\n                        bad_counts = pd.Series((~np.isfinite(checked)).sum(axis=0), index=RAW)\n                        forbidden = bad_counts.drop('bid')\n                        if forbidden.sum() or np.isinf(chunk.bid.to_numpy(dtype=float)).any():\n                            raise ValueError('Unexpected nonfinite values: '+bad_counts[bad_counts>0].to_json())\n                        missing_bid = chunk.bid.isna()\n                        if missing_bid.any():\n                            for (pid, ad), group in chunk.loc[missing_bid].groupby(['deliveryPeriodIndex','advertiserNumber']):\n                                key = (int(pid), int(ad))\n                                excluded[key] = excluded.get(key, 0) + len(group)\n                            write_json(data/f'exclusions_{pack}.json', {\n                                'reason': 'NaN bid; whole advertiser-period trajectory excluded from transitions',\n                                'episodes': [dict(period=k[0],advertiser=k[1],missing_bid_rows=n)\n                                             for k,n in sorted(excluded.items())]})\n                            print('QUARANTINE', pack, sorted(excluded), 'missing rows', int(missing_bid.sum()), flush=True)"), ('parts=[]; count=0; writers={}', 'parts=[]; count=0; writers={}; excluded={}'), ('        rows=trajectories(agg)', "        if excluded:\n            keep = [(int(pid),int(ad)) not in excluded\n                    for pid,ad in zip(agg.index.get_level_values(0),agg.index.get_level_values(1))]\n            agg = agg.loc[keep]\n            if agg.empty: raise ValueError('No usable trajectories remain')\n        rows=trajectories(agg)\n        if not all(np.isfinite(np.r_[r['state'], r['next_state'], r['action'], r['reward_continuous']]).all() for r in rows):\n            raise ValueError('Nonfinite transitions after quarantine')"), ("training_advertisers='All 48'", "training_advertisers='All 48 except audited NaN-bid trajectories'")]
    for old, new in edits:
        if s.count(old) != 1:
            raise SystemExit("코드 버전이 달라 수정하지 않았습니다: " + old[:65])
        s = s.replace(old, new, 1)
    compile(s, str(path), "exec")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, path.with_name("overnight_before_nan_fix_" + stamp + ".py"))
    temporary = path.with_suffix(".py.tmp")
    temporary.write_text(s)
    temporary.replace(path)
    print("수정 완료: 결측 입찰가 궤적 제외 및 감사 기록")
else:
    print("이미 수정되어 있습니다.")
failed = path.parent / "outputs/FAILED.txt"
if failed.exists():
    failed.rename(failed.with_name("FAILED_previous_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"))
