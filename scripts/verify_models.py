#!/usr/bin/env python3
"""Static verification of bundled TFLite sizes/SHA256 and golden vector layouts.
Only standard Python libraries; not Android inference or ground-truth testing.
"""
from pathlib import Path
import hashlib, json, sys, xml.etree.ElementTree as ET

R = Path(__file__).resolve().parents[1]
A = R / 'android/app/src/main/assets'
fail = []
verified = 0

for subdir, filename, format_names in [
    ('validator', 'stageA_deployment_config.json', ['INT8']),
    ('multiclass', 'model_config.json', ['FP32', 'FP16', 'INT8']),
]:
    folder = A/subdir
    cfg = json.loads((folder/filename).read_text())
    for fmt in format_names:
        spec = cfg['models'][fmt]
        model = folder / spec['filename']
        if not model.is_file():
            fail.append(f'{subdir}/{fmt}: missing {model.name}'); continue
        actual_len = model.stat().st_size
        sha = hashlib.sha256()
        with model.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                sha.update(chunk)
        actual_sha = sha.hexdigest()
        if actual_len != int(spec['size_bytes']):
            fail.append(f'{subdir}/{fmt}: size {actual_len}, expected {spec["size_bytes"]}')
        if actual_sha.lower() != spec['sha256'].lower():
            fail.append(f'{subdir}/{fmt}: SHA256 mismatch')
        else:
            verified += 1
        print(f'{subdir}/{fmt}: {model.name} | {actual_len} bytes | SHA256 {actual_sha}')

for folder, count, size in [('validator',8,160*160*3*4),('multiclass',4,224*224*3*4)]:
    cases=list((A/folder/'golden_inputs').glob('*.bin'))
    if len(cases) != count:
        fail.append(f'{folder}: found {len(cases)} golden files, expected {count}')
    for p in cases:
        if p.stat().st_size != size:
            fail.append(f'{folder}: {p.name} wrong bytes ({p.stat().st_size}, expected {size})')
    json.loads((A/folder/'golden_cases.json').read_text())
    print(f'{folder}: {len(cases)}/{count} golden inputs with expected byte size')

for rel in ['android/app/src/main/res/layout/activity_main.xml','android/app/src/main/AndroidManifest.xml']:
    ET.parse(R/rel)
    print(f'{rel}: XML parsed')

print(f'Bundled model hashes matched: {verified}/4')
if fail:
    print('FAILED:',*fail,sep='\n - ')
    sys.exit(1)
print('PASS: static integrity/layout checks (not inference correctness)')
