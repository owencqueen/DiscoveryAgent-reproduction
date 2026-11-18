#!/usr/bin/env python3
"""Download the ACS housing files used by the economics analyses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(entry, root):
    destination = root / entry['relative_path']
    if destination.exists():
        if digest(destination) != entry['sha256']:
            raise ValueError(f'Existing file has a different hash: {destination}')
        print(f'Already verified: {destination}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = entry['upstream_public_archive_url']
    print(f'Downloading {url}', flush=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / 'source.zip'
        request = Request(url, headers={'User-Agent': 'DiscoveryAgent-reproduction'})
        with urlopen(request, timeout=120) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        extracted = temporary / destination.name
        with ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist()
                       if Path(name).name == destination.name]
            if len(members) != 1:
                raise ValueError(f'Expected one {destination.name} in {url}')
            with bundle.open(members[0]) as source, extracted.open('wb') as output:
                shutil.copyfileobj(source, output)
        if digest(extracted) != entry['sha256']:
            raise ValueError(f'Downloaded {destination.name} differs from the recorded input hash')
        extracted.replace(destination)
    print(f'Verified: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'outputs/economics')
    parser.add_argument('--panel', nargs='+', choices=['E1', 'E6'], default=['E1', 'E6'])
    args = parser.parse_args()
    entries = json.loads((ROOT/'INPUTS.json').read_text())['inputs']
    try:
        for entry in entries:
            if entry['study'] == 'economics' and set(entry['panels']) & set(args.panel):
                download(entry, args.data_root.resolve())
    except Exception as error:
        parser.exit(1, f'Download failed: {error}\n')


if __name__ == '__main__':
    main()
