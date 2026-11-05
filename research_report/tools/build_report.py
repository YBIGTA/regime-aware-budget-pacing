#!/usr/bin/env python3
"""Read-only experiment archival; writes README and research_report only."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
EXPECTED = '1412b714122726eac0e7012d73547b98bf82f4722da43c0d0a92d063f74059bd'
CHUNK = 40 * 1024 * 1024


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def command(*args, cwd):
    r = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {'returncode': r.returncode, 'stdout': r.stdout.strip(),
            'stderr': r.stderr.strip()}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    return x


def table(frame):
    f = frame.copy()
    def cell(v):
        if pd.isna(v): return 'NA'
        if isinstance(v, (float, np.floating)): return f'{v:.9g}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join([
        '| ' + ' | '.join(map(str, f.columns)) + ' |',
        '| ' + ' | '.join(['---'] * len(f.columns)) + ' |',
        *['| ' + ' | '.join(cell(v) for v in row) + ' |'
          for row in f.itertuples(index=False, name=None)],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    root = (args.root or Path(command('git', 'rev-parse', '--show-toplevel',
                                    cwd=Path.cwd())['stdout'])).resolve()
    if not root.is_dir(): raise SystemExit('프로젝트 루트를 찾지 못했습니다.')
    run = root / 'pacing_v11/auctionnet_lab/runs/pacing_protocol_v11_1412b714'
    required = [
        'PROTOCOL.json', 'FROZEN_BEFORE_TEST.json', 'advertisers.json',
        'fixed_baseline.json', 'fixed_baseline_search.csv', 'state_sampling.json',
        'learning_audit.csv', 'policy_selection.csv', 'final_episodes.csv',
        'final_summary.csv', 'final_by_period.csv', 'paired_final.csv', 'DONE.json',
    ]
    missing = [str(run / n) for n in required if not (run / n).is_file()]
    if missing: raise SystemExit('필수 결과가 없어 중단합니다:\n' + '\n'.join(missing))
    done = json.loads((run / 'DONE.json').read_text())
    if done.get('status') != 'COMPLETE': raise SystemExit('v11 완료 상태가 아닙니다.')
    source = root / 'pacing_v11/pacing_protocol_v11.py'
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    if sha(source) != EXPECTED or protocol.get('code') != EXPECTED:
        raise SystemExit('v11 소스와 프로토콜 hash가 알려진 실행 버전과 다릅니다.')
    for name, digest in protocol.get('sources', {}).items():
        if sha(source.parent / name) != digest:
            raise SystemExit(f'실행 종속 소스 hash 불일치: {name}')
    final = root / 'research_report'
    if final.exists():
        raise SystemExit('research_report가 이미 있습니다. 기존 보고서를 보존하기 위해 중단했습니다. 이미 생성됐다면 README 확인 후 아래 git 명령만 실행하세요.')
    df = pd.read_csv(run / 'final_episodes.csv')
    needed = ['method', 'period', 'advertiser', 'te', 'value', 'j', 'tv', 'unspent', 'changed_ticks']
    if not set(needed).issubset(df): raise SystemExit('final_episodes.csv 스키마 불일치')
    methods = ['fixed', 'learned_raw', 'selected_policy', 'time_only', 'matched_full_tree']
    ads = sorted(json.loads((run / 'advertisers.json').read_text()))
    if len(ads) != 48: raise SystemExit('광고주48명 조건 불일치')
    expected_cases = {(p, a) for p in [24, 25, 26, 27] for a in ads}
    if set(df.method) != set(methods): raise SystemExit('최종 비교 방법 목록 불일치')
    for method in methods:
        part = df[df.method.eq(method)]
        if part.duplicated(['period', 'advertiser']).any(): raise SystemExit('중복 대응 사례')
        if set(zip(part.period, part.advertiser)) != expected_cases:
            raise SystemExit(f'{method}: 평가 사례 누락/추가')
    metrics = ['te', 'value', 'j', 'tv', 'unspent', 'changed_ticks']
    if not np.isfinite(df[metrics].to_numpy(float)).all(): raise SystemExit('비유한 최종 지표')
    summary = df.groupby(['method', 'period'])[metrics].mean().groupby('method').mean().reset_index()
    saved = pd.read_csv(run / 'final_summary.csv').set_index('method').sort_index()
    measured = summary.set_index('method').sort_index()
    if not np.allclose(saved[metrics], measured[metrics], rtol=1e-9, atol=1e-10):
        raise SystemExit('원시 에피소드 재집계와 저장된 최종 요약이 다릅니다.')
    by_period = df.groupby(['method', 'period'])[metrics].mean().reset_index()
    saved_period = pd.read_csv(run / 'final_by_period.csv').set_index(['method', 'period']).sort_index()
    recalculated_period = by_period.set_index(['method', 'period']).sort_index()
    for col in ['te', 'value', 'j', 'unspent']:
        if not np.allclose(saved_period[col], recalculated_period[col], rtol=1e-9, atol=1e-10):
            raise SystemExit('기간별 요약 재집계 불일치')
    keys = ['period', 'advertiser']
    paired = df[df.method.eq('fixed')][keys + metrics].merge(
        df[df.method.eq('learned_raw')][keys + metrics], on=keys,
        suffixes=('_fixed', '_learned'), validate='one_to_one')
    eps = 1e-10
    paired['te_delta'] = paired.te_learned - paired.te_fixed
    paired['te_gain_pct'] = -100 * paired.te_delta / paired.te_fixed.where(paired.te_fixed > eps)
    paired['value_change_pct'] = 100 * (paired.value_learned - paired.value_fixed) / paired.value_fixed.where(paired.value_fixed > eps)
    paired['te_improved'] = paired.te_delta < -eps
    paired['te_worsened'] = paired.te_delta > eps
    paired['te_tied'] = paired.te_delta.abs() <= eps
    paired['value_preserved_98pct'] = paired.value_learned >= .98 * paired.value_fixed - eps
    paired['both_pass'] = paired.te_improved & paired.value_preserved_98pct
    counts = paired.groupby('period').agg(cases=('te_improved','size'), te_improved=('te_improved','sum'), te_worsened=('te_worsened','sum'), te_tied=('te_tied','sum'), value_preserved=('value_preserved_98pct','sum'), both_pass=('both_pass','sum')).reset_index()
    per_ad = paired.groupby('advertiser')[[f'{m}_{s}' for m in metrics for s in ['fixed','learned']]].mean()
    per_ad['te_gain_pct'] = 100 * (per_ad.te_fixed - per_ad.te_learned) / per_ad.te_fixed.where(per_ad.te_fixed > eps)
    per_ad['value_change_pct'] = 100 * (per_ad.value_learned - per_ad.value_fixed) / per_ad.value_fixed.where(per_ad.value_fixed > eps)
    per_ad['periods_te_improved'] = paired.groupby('advertiser').te_improved.sum()
    per_ad['periods_both_pass'] = paired.groupby('advertiser').both_pass.sum()
    per_ad['both_pass_on_average'] = (per_ad.te_learned < per_ad.te_fixed - eps) & (per_ad.value_learned >= .98 * per_ad.value_fixed - eps)
    per_ad = per_ad.reset_index().sort_values('te_gain_pct')
    frozen = json.loads((run / 'FROZEN_BEFORE_TEST.json').read_text())
    if frozen.get('enabled'):
        a = df[df.method.eq('learned_raw')].sort_values(keys)[metrics].to_numpy()
        b = df[df.method.eq('selected_policy')].sort_values(keys)[metrics].to_numpy()
        if not np.allclose(a, b): raise SystemExit('enabled임에도 selected와raw가 다릅니다.')
    work = root / f'.research_report_build_{os.getpid()}'
    work.mkdir()
    manifest = []; warnings = []; label_counts = []
    def archive(src, relative):
        if src.is_symlink():
            warnings.append(f'symlink 미수집: {src.relative_to(root)}'); return
        dest = work / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha(src)
        record = {'source': str(src.relative_to(root)), 'archived': str(Path('research_report')/relative), 'bytes': src.stat().st_size, 'sha256': original_hash, 'encoding':'identity', 'parts':''}
        if src.stat().st_size <= CHUNK:
            shutil.copy2(src, dest)
            if sha(dest) != original_hash: raise RuntimeError('보존 복사 hash 불일치')
        else:
            gz = dest.with_name(dest.name + '.gz')
            with src.open('rb') as inp, gzip.GzipFile(filename=str(gz),mode='wb',mtime=0) as out:
                shutil.copyfileobj(inp, out)
            pieces=[]
            if gz.stat().st_size > CHUNK:
                with gz.open('rb') as inp:
                    i=0
                    while True:
                        data=inp.read(CHUNK)
                        if not data: break
                        pp=gz.with_name(gz.name+f'.part{i:04d}');pp.write_bytes(data)
                        pieces.append(str(Path('research_report')/pp.relative_to(work)));i+=1
                gz.unlink();record['encoding']='gzip-split'
                record['archived']='';record['parts']=json.dumps(pieces,ensure_ascii=False)
            else:
                record['encoding']='gzip';record['archived']=str(Path('research_report')/gz.relative_to(work))
        manifest.append(record)
    derived = work / 'derived'; derived.mkdir()
    for name, frame in [('paired_cases.csv',paired),('advertiser_means.csv',per_ad),('period_counts.csv',counts),('recomputed_summary.csv',summary),('recomputed_by_period.csv',by_period)]:
        frame.to_csv(derived/name,index=False,float_format='%.17g')
    distribution = pd.DataFrame({
        'metric':['absolute_TE_change','relative_TE_gain_pct','relative_value_change_pct'],
        'mean':[paired.te_delta.mean(),paired.te_gain_pct.mean(),paired.value_change_pct.mean()],
        'median':[paired.te_delta.median(),paired.te_gain_pct.median(),paired.value_change_pct.median()],
        'min':[paired.te_delta.min(),paired.te_gain_pct.min(),paired.value_change_pct.min()],
        'max':[paired.te_delta.max(),paired.te_gain_pct.max(),paired.value_change_pct.max()],
    })
    distribution.to_csv(derived/'case_distribution.csv',index=False,float_format='%.17g')
    result_roots=[root/'auctionnet_lab/runs',root/'pacing_v11/auctionnet_lab/runs',root/'auctionnet_paper_v7/outputs',root/'레거시/outputs',root/'레거시/experiments',TOOLS/'historical_reference']
    text_suffix={'.csv','.json','.jsonl','.md','.txt','.log','.yaml','.yml'}
    scanned_roots=[]
    for folder in result_roots:
        if not folder.is_dir(): warnings.append('해당 결과 디렉터리 없음: '+str(folder.relative_to(root)));continue
        scanned_roots.append(str(folder.relative_to(root)))
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(folder)
            if any(x in rel.parts for x in ['cache','data','.venv','__pycache__']):continue
            if p.suffix.lower() in text_suffix or p.name in ['DONE','FAILED'] or (p.suffix=='.joblib' and p.parent==run):
                archive(p,Path('results')/p.relative_to(root))
    for data_folder in [root/'auctionnet_lab/data',run/'data',root/'auctionnet_paper_v7/data']:
        for p in sorted(data_folder.glob('pack-*/manifest.json')):
            archive(p,Path('results')/p.relative_to(root))
    inventory=[]
    data_roots=[root/'auctionnet_lab/data',run/'data',root/'auctionnet_paper_v7/data']
    for folder in data_roots:
        if not folder.exists():continue
        for p in sorted(folder.rglob('*.parquet')):
            if p.is_symlink():continue
            try:
                import pyarrow.parquet as pq
                meta=pq.ParquetFile(p)
                row=dict(rows=meta.metadata.num_rows,row_groups=meta.metadata.num_row_groups,schema=str(meta.schema_arrow))
            except ImportError:
                row=dict(rows=None,row_groups=None,schema='pyarrow not installed; metadata unavailable')
                if 'pyarrow 없음: 원본 Parquet metadata 미검사' not in warnings:warnings.append('pyarrow 없음: 원본 Parquet metadata 미검사')
            inventory.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,**row,content_sha256='not computed; pack ZIP sha256 in source manifests'))
    pd.DataFrame(inventory,columns=['path','bytes','rows','row_groups','schema','content_sha256']).to_csv(work/'data_inventory.csv',index=False)
    # Q caches contain dictionaries/arrays from this user's completed experiment.
    # No external or downloaded pickle is loaded.
    import joblib
    label_dir=work/'training_labels';label_dir.mkdir()
    qpaths=sorted((run/'cache').glob('Q_*.joblib'))
    if not qpaths: warnings.append('v11 Q 라벨 캐시 없음: 라벨 원시 상태를 복구하지 못함')
    if qpaths:
        with gzip.open(label_dir/'v11_Q_states.jsonl.gz','wt',encoding='utf-8') as out:
            for p in qpaths:
                obj=joblib.load(p)
                if not isinstance(obj,dict) or not {'x','j','value','period','ad','start'}.issubset(obj):
                    raise ValueError('예상과 다른 Q 캐시: '+p.name)
                out.write(json.dumps(dict(cache_file=p.name,cache_sha256=sha(p),payload=clean(obj)),ensure_ascii=False,allow_nan=False)+'\n')
                label_counts.append({'cache_file':p.name,'period':obj['period'],'advertiser':obj['ad'],'decision_tick':obj['start'],'prefix':obj.get('prefix'),'actions':len(obj['j'])})
        pd.DataFrame(label_counts).to_csv(label_dir/'cache_inventory.csv',index=False)
    # Preserve source and original dependency declarations, never environment secrets.
    candidates=set()
    for folder in [root/'auctionnet_lab',root/'pacing_v11',root/'auctionnet_paper_v7',root/'레거시/src',root/'레거시/configs',root/'레거시/docs']:
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(folder)
            if any(x in relative.parts for x in ['.venv','__pycache__','cache','data','runs','outputs','.git']):continue
            legacy_config = '레거시' in p.relative_to(root).parts and p.suffix in ['.json','.yaml','.yml','.md','.toml']
            if legacy_config or p.suffix in ['.py','.sh','.command'] or p.name in ['README.md','README.txt','AUDIT.md','PROTOCOL.txt','requirements.txt','requirements.lock','pyproject.toml','uv.lock']:
                candidates.add(p)
    for p in root.iterdir():
        if p.is_file() and (p.name.startswith(('gain_','train_gain','fix_v7','auctionnet_')) and p.suffix in ['.py','.sh'] or p.name in ['requirements.txt','pyproject.toml','uv.lock','requirements.lock']):candidates.add(p)
    for p in sorted(candidates):archive(p,Path('source')/p.relative_to(root))
    evidence=TOOLS/'evidence_user_console.md'
    if evidence.exists():
        (work/'evidence').mkdir();shutil.copy2(evidence,work/'evidence/user_console.md')
    (work/'tools').mkdir()
    for name in ['build_report.py','README_BASE.md','PUBLISH.sh']:
        shutil.copy2(TOOLS/name,work/'tools'/name)
    old=root/'README.md'
    if old.exists():shutil.copy2(old,work/'previous_root_README.md')
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','joblib','pyarrow','optuna','hdbscan','torch']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='not installed'
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'source_code_sha256':sha(source),'protocol_code_sha256':protocol['code'],'completed_run':str(run.relative_to(root)), 'validation':'5 methods x192 matched episodes; finite metrics; summary/by-period parity; source hash parity', 'git_head_before_report':command('git','rev-parse','HEAD',cwd=root), 'python':sys.version,'platform':platform.platform(),'packages_at_report_generation':packages,'result_roots_scanned':scanned_roots,'Q_cache_files_exported':len(qpaths),'data_parquet_files_inventoried':len(inventory),'warnings':warnings,'excluded':'Raw auction Parquet/ZIP, episode feature caches, virtual environments, secret configuration, historical large neural checkpoints. Result text/CSV and v11 top-level joblib models included. Other directories outside declared roots not recursively searched.', 'statistical_inference':'Descriptive. No p-values, no independence claim, no unperformed experiment.', 'build_status':'COMPLETE'}
    (work/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(manifest).to_csv(work/'file_manifest.csv',index=False)
    # Hash every archived/generated file except the checksum index itself.
    with (work/'CHECKSUMS.sha256').open('w',encoding='utf-8') as f:
        for p in sorted(work.rglob('*')):
            if p.is_file() and p.name!='CHECKSUMS.sha256':f.write(sha(p)+'  '+p.relative_to(work).as_posix()+'\n')
    total=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
    if total>400*1024*1024:
        raise SystemExit(f'보고서 증거가 {total/2**20:.1f}MiB로 큽니다. 자동 Git 커밋을 중단합니다. 보존된 임시 폴더: {work}')
    sections=[(TOOLS/'README_BASE.md').read_text(encoding='utf-8')]
    def add(title,content):sections.append('\n## '+title+'\n\n'+content+'\n')
    add('A. 실제 자료 검증 및 포함 범위', '```json\n'+json.dumps(audit,ensure_ascii=False,indent=2)+'\n```')
    add('B. 원시 에피소드에서 재계산한 최종 요약',table(summary))
    add('C. 원시 에피소드에서 재계산한 기간별 요약',table(by_period))
    add('D. 사례별 빈도: 개선·악화·동률을 분리',table(counts))
    add('E. 광고주48명 각각의 전체 기간 평균',table(per_ad))
    add('F. 사례별 변화 분포',table(distribution)+'\n\n개별 비율 평균은 전체 평균의 개선율과 다르다. NA는 기준값0 등 비율 미정의다.')
    add('G. 절대 TE 악화가 가장 큰 10사례',table(paired.sort_values('te_delta',ascending=False)[keys+['te_fixed','te_learned','te_delta','te_gain_pct','value_change_pct','both_pass']].head(10)))
    add('H. 상대 TE 악화가 가장 큰 10사례',table(paired.sort_values('te_gain_pct')[keys+['te_fixed','te_learned','te_delta','te_gain_pct','value_change_pct','both_pass']].head(10)))
    for name in ['PROTOCOL.json','FROZEN_BEFORE_TEST.json','fixed_baseline.json','state_sampling.json','calibration_audit.json']:
        p=run/name
        if p.exists(): add('I. '+name,'```json\n'+p.read_text(encoding='utf-8')+'\n```')
    for name in ['fixed_baseline_search.csv','policy_selection.csv','learning_audit.csv']:
        add('J. '+name,'```csv\n'+(run/name).read_text(encoding='utf-8')+'\n```')
    add('K. 최종 raw 에피소드 지표 전체 (5방법×192=960행)', '원시 수치는 반올림하지 않고 그대로 삽입한다. 전체 경매 행이 아니라 에피소드 단위 평가 원시표다.\n\n```csv\n'+(run/'final_episodes.csv').read_text(encoding='utf-8')+'\n```')
    # Embed all small historical/result tables too; larger files have exact links.
    entries=[];budget=2*1024*1024
    for r in manifest:
        target=work/Path(r['archived']).relative_to('research_report') if r['archived'] else None
        entries.append('| '+r['source']+' | '+str(r['bytes'])+' | '+r['sha256']+' | '+('['+r['encoding']+']('+r['archived']+')' if r['archived'] else 'split: file_manifest.csv')+' |')
        if target and target.suffix in ['.csv','.json','.txt','.md'] and target.stat().st_size<128*1024 and 'results' in target.relative_to(work).parts and target.parent!=work/'results'/run.relative_to(root) and target.stat().st_size<=budget:
            text=target.read_text(encoding='utf-8',errors='replace').replace('```','~~~')
            add('L. 보존된 이전 결과 원문: '+r['source'],'```text\n'+text+'\n```');budget-=target.stat().st_size
    add('M. 보존 원본 파일 전체 인덱스','| 원본 경로 | bytes | SHA256 | 보존 파일 |\n|---|---:|---|---|\n'+'\n'.join(entries))
    add('N. 전체 수집 결과와 원본 보존 원칙',f'Q 라벨 캐시 {len(qpaths)}개 추출, Parquet {len(inventory)}개 metadata 기록, 증거 패키지 {total/2**20:.2f}MiB. 데이터 자체의 독립성·안전성이 이 수로 입증되는 것은 아니다.\n\n원본 실험 파일은 수정하지 않았다. 현재 소스·설정·결과·라벨·데이터 출처를 묶었으나 저장되지 않은 틱별 trace·중간 모델·실험 당시 환경을 새로 만들어 내지는 않는다. 없는 자료는 위 경고 및 원시 파일 인덱스 기준으로 추가 확보해야 한다.\n\n40MiB 초과 원본은 gzip/분할될 수 있다. 복원은 file_manifest.csv의 encoding 및 parts를 따르며, 분할은 번호순으로 합친 뒤 gzip을 풀고 원래 SHA256을 확인한다.')
    report='\n'.join(sections)
    if '\x00' in report:raise ValueError('문서 내 잘못된 문자')
    work.rename(final)
    temp=root/'.README.research.tmp';temp.write_text(report,encoding='utf-8');temp.replace(root/'README.md')
    print(f'완료: {root/"README.md"}')
    print(f'원시 결과·소스·감사 자료: {final}')
    print(f'검증 통과: {len(df)}행; 증거 {total/2**20:.2f}MiB; Q 라벨 {len(qpaths)}개')
    for w in warnings:print('자료 경고:',w)

if __name__=='__main__':main()
