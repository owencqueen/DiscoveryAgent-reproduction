#!/usr/bin/env python3
"""Download the public GW-CD4 DE results and restore the v1.0 metadata used in the paper.

The Virtual Cells Platform replaced GWCD4i.DE_stats.h5ad on 2026-05-28. The
differential expression layers are unchanged, but the new file reorders rows and
revises .obs and .varm (for example, `offtarget_flag` was removed). The case
study used the original v1.0 file, so this tool combines the public layers with
the bundled v1.0 metadata and checks the result against the original content.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
H5AD = 'data/GWCD4i.DE_stats.h5ad'
ROWS_PER_BLOCK = 2000


def entry():
    entries = json.loads((ROOT/'INPUTS.json').read_text())['inputs']
    return next(e for e in entries if e['study'] == 'genomics' and e['relative_path'] == H5AD)


def read_elem(element):
    try:
        from anndata.io import read_elem as read
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem as read
    return read(element)


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _frame_bytes(frame):
    import numpy as np
    import pandas as pd
    yield '\x1f'.join(map(str, frame.index)).encode()
    for name in frame.columns:
        column = frame[name]
        yield f'\x1e{name}\x1e'.encode()
        if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
            dtype = '<f8' if pd.api.types.is_float_dtype(column) else '<i8'
            yield np.ascontiguousarray(column.to_numpy(), dtype=dtype).tobytes()
        else:
            yield '\x1f'.join(map(str, column)).encode()


def content_digest(path):
    """SHA-256 over the layers, obs, var and varm of an H5AD, independent of HDF5 layout."""
    import h5py
    import numpy as np
    h = hashlib.sha256()
    with h5py.File(path, 'r') as f:
        for name in sorted(f['layers']):
            layer = f['layers'][name]
            h.update(f'{name}|{layer.shape}|{layer.dtype.str}\n'.encode())
            for start in range(0, layer.shape[0], ROWS_PER_BLOCK):
                block = layer[start:start + ROWS_PER_BLOCK]
                h.update(np.ascontiguousarray(block, dtype=block.dtype.newbyteorder('<')).tobytes())
        frames = [('obs', read_elem(f['obs'])), ('var', read_elem(f['var']))]
        frames += [(f'varm/{k}', read_elem(f['varm'][k])) for k in sorted(f['varm'])]
    for name, frame in frames:
        h.update(f'\n{name}\n'.encode())
        for chunk in _frame_bytes(frame):
            h.update(chunk)
    return h.hexdigest()


def download(url, destination, expected_bytes):
    """Stream `url` to `destination`, resuming a partial download."""
    done = destination.stat().st_size if destination.exists() else 0
    if done == expected_bytes:
        return
    if done > expected_bytes:
        raise ValueError(f'{destination} is larger than the public file; delete it and retry')
    headers = {'User-Agent': 'DiscoveryAgent-reproduction'}
    if done:
        headers['Range'] = f'bytes={done}-'
        print(f'Resuming at {done / 1e9:.1f} GB', flush=True)
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        if done and response.status != 206:
            raise ValueError('Server ignored the resume request; delete the partial file and retry')
        with destination.open('ab') as output:
            shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
    if destination.stat().st_size != expected_bytes:
        raise ValueError(f'Incomplete download: {destination}; rerun to resume')


def rebuild(public, metadata, output):
    """Write `output` with the public layers in v1.0 row order and the v1.0 metadata."""
    import anndata as ad
    import h5py
    import pandas as pd
    v1 = ad.read_h5ad(metadata)
    with h5py.File(public, 'r') as f:
        obs, var = read_elem(f['obs']), read_elem(f['var'])
    if not var.index.equals(v1.var.index):
        raise ValueError('Public measured genes differ from v1.0; the public file has changed again')
    key = lambda frame: pd.MultiIndex.from_arrays(
        [frame['target_contrast'].astype(str), frame['culture_condition'].astype(str)])
    public_keys, v1_keys = key(obs), key(v1.obs)
    rows = public_keys.get_indexer(v1_keys)
    if len(obs) != len(v1.obs) or not public_keys.is_unique or (rows < 0).any():
        raise ValueError('Public perturbations differ from v1.0; the public file has changed again')
    v1.write_h5ad(output)
    with h5py.File(public, 'r') as source, h5py.File(output, 'a') as target:
        if 'X' in source and 'X' not in target:  # v1.0 stores an empty X placeholder
            source.copy(source['X'], target, 'X')
        for name in source['layers']:
            print(f'  layer {name}', flush=True)
            layer = source['layers'][name]
            written = target['layers'].create_dataset(name, data=layer[...][rows])
            written.attrs.update(dict(layer.attrs))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='genomics workspace (a copy of inputs/genomics)')
    parser.add_argument('--public-h5ad', type=Path,
                        help='use an already downloaded public GWCD4i.DE_stats.h5ad')
    parser.add_argument('--keep-public', action='store_true',
                        help='keep the downloaded public file after restoring v1.0')
    args = parser.parse_args()
    e = entry()
    release = e['public_release']
    root = args.data_root.resolve()
    output = root/H5AD
    metadata = root/e['v1_metadata']
    try:
        if output.exists():
            print(f'Checking existing {output} (reads 17 GB)', flush=True)
            if content_digest(output) != e['content_sha256']:
                raise ValueError(f'Existing file does not match v1.0 content: {output}')
            print(f'Already verified: {output}')
            return
        if not metadata.is_file() or file_digest(metadata) != e['v1_metadata_sha256']:
            raise ValueError(f'Missing or modified v1.0 metadata: {metadata}; copy inputs/genomics again')
        public = args.public_h5ad or root/'data/downloads'/release['filename']
        if args.public_h5ad is None:
            public.parent.mkdir(parents=True, exist_ok=True)
            print(f'Downloading {release["url"]} ({release["bytes"] / 1e9:.1f} GB)', flush=True)
            download(release['url'], public, release['bytes'])
        print(f'Hashing {public}', flush=True)
        if file_digest(public) != release['sha256']:
            print('Warning: the public file differs from the 2026-05-28 release; '
                  'the final content check will decide whether it is usable.', flush=True)
        temporary = output.with_name(output.name + '.partial')
        temporary.unlink(missing_ok=True)
        print('Restoring v1.0 row order and metadata', flush=True)
        rebuild(public, metadata, temporary)
        print('Verifying restored content', flush=True)
        if content_digest(temporary) != e['content_sha256']:
            raise ValueError(f'Restored file does not match v1.0 content; left at {temporary}')
        temporary.replace(output)
        if args.public_h5ad is None and not args.keep_public:
            public.unlink()
    except Exception as error:
        parser.exit(1, f'Genomics data setup failed: {error}\n')
    print(f'Verified: {output}')


if __name__ == '__main__':
    main()
