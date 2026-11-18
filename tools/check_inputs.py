#!/usr/bin/env python3
"""Check study input files against their recorded hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',nargs='+',required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    args=ap.parse_args()
    data=json.loads((ROOT/'INPUTS.json').read_text())
    panels=json.loads((ROOT/'PANELS.json').read_text())
    unknown=set(args.panel)-set(panels)
    if unknown:ap.error('Unknown panel IDs: '+', '.join(sorted(unknown)))
    if not args.data_root.is_dir():ap.error(f'--data-root is not a directory: {args.data_root}')
    if len({panels[x]['study'] for x in args.panel})!=1:
        ap.error('Use one study workspace per invocation.')
    selected=[e for e in data['inputs'] if set(e['panels']) & set(args.panel)
]
    if not selected:
        print('No analysis inputs are listed for these panels; nothing to check.')
        return 0
    failures=0
    for e in selected:
        p=args.data_root/e['relative_path']
        if not p.is_file():
            print('MISSING',e['relative_path']);failures+=1;continue
        if e.get('content_sha256'):
            from download_genomics import content_digest
            good=content_digest(p)==e['content_sha256']
            print('MATCH' if good else 'MISMATCH',e['relative_path'],'(content)')
            failures+=not good;continue
        if not e.get('sha256'):
            print('UNVERIFIED (no reference hash)',e['relative_path']);failures+=1;continue
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        good=h.hexdigest()==e['sha256']
        print('MATCH' if good else 'MISMATCH',e['relative_path'])
        failures+=not good
    print(f'{len(selected)} listed inputs checked; {failures} missing, mismatched or unauthenticated.')
    return 2 if failures else 0

if __name__=='__main__':raise SystemExit(main())
