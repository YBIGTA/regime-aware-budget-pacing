
from pathlib import Path
import csv
import hashlib
import json
import tarfile
import tempfile

here = Path(__file__).resolve().parent
meta = json.loads((here / "archive_manifest.json").read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

output = Path(tempfile.mkdtemp(prefix="pacing_A_restored_"))
archive = output / "A_originals.tar.gz"
with archive.open("wb") as target:
    for part in meta["parts"]:
        path = here / "raw_archive" / part["name"]
        if sha(path) != part["sha256"]:
            raise RuntimeError("압축 조각 해시 불일치: " + part["name"])
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
if sha(archive) != meta["archive_sha256"]:
    raise RuntimeError("전체 압축 해시 불일치")

with tarfile.open(archive, "r:gz") as tar:
    for member in tar:
        path = Path(member.name)
        if not member.isfile() or path.is_absolute() or ".." in path.parts:
            raise RuntimeError("잘못된 압축 경로")
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, target.open("wb") as dest:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(block)

with (here / "original_manifest.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        path = output / row["path"]
        if path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
            raise RuntimeError("복원 원본 불일치: " + row["path"])
print("전체 원본 복원 및 무결성 확인 완료:", output)
