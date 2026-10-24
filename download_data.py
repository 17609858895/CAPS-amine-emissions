"""Download the ten public source workbooks and verify their published checksums."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent
out=ROOT/'carbon_capture_zenodo_5153417/raw';out.mkdir(parents=True,exist_ok=True)
for item in json.loads((ROOT/'source_data.json').read_text())['files']:
    p=out/item['filename']
    if not p.exists():
        request=urllib.request.Request(item['source_url'],headers={'User-Agent':'CAPS-reproducibility/1.0'})
        with urllib.request.urlopen(request) as response:p.write_bytes(response.read())
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual!=item['sha256']:raise ValueError('Checksum mismatch: '+p.name)
    print(p.name,'verified')
