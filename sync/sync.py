"""Sync the datastore with a Zenodo record: pull what is only there, push what is only here.

The record holds one zip per case folder (<name>/<hash8>) with the datastore tree inside, and one more with its
model runs (model_results/), which are only pulled with --model-results.
A push makes a new version of the record as a draft; publish it in the Zenodo web page.
The token is read from $ZENODO_TOKEN or sync/.token; without one you can only pull.

python sync.py <datastore> [--record <id>] [--model-results]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib

URL = os.environ.get('ZENODO_URL', 'https://sandbox.zenodo.org')
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.token')
TOKEN = os.environ.get('ZENODO_TOKEN') or (open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else None)


def call(url, method='GET', body=None, ctype='application/json', size=None):
    if method != 'GET' and not TOKEN:
        sys.exit('no token: set ZENODO_TOKEN or put it in sync/.token (without one you can only pull)')
    headers = {'Content-Type': ctype, **({'Authorization': f'Bearer {TOKEN}'} if TOKEN else {})}
    if size:
        headers['Content-Length'] = str(size)
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, body, headers, method=method)) as r:
            return json.loads(r.read() or 'null')
    except urllib.error.HTTPError as e:
        sys.exit(f'{method} {url}: {e.code} {e.read().decode()[:300]}')


class RemoteFile:
    """a file on the web that zipfile can read: every read is a range request"""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if self.pos >= end:
            return b''
        with urllib.request.urlopen(urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end - 1}'})) as r:
            data = r.read()
        self.pos = end
        return data


def crcs(datastore, case):
    """path in the datastore -> crc32, for every file of a case"""
    out = {}
    for path in glob.glob(f'{datastore}/{case}/**', recursive=True):
        if os.path.isfile(path):
            crc = 0
            with open(path, 'rb') as f:
                while chunk := f.read(1 << 20):
                    crc = zlib.crc32(chunk, crc)
            out[os.path.relpath(path, datastore)] = crc
    return out


def labels(paths):
    """what a set of files gives the workflow"""
    found = {'model': '/model_results/', 'surrogate': '/engine.h5', 'uq': '/sensitivity.h5',
             'report': '/report.pdf', 'comparison': '/comparison.pdf'}
    return ', '.join(label for label, part in found.items() if any(part in '/' + p for p in paths))


parser = argparse.ArgumentParser()
parser.add_argument('datastore')
parser.add_argument('--record')
parser.add_argument('--model-results', action='store_true', help='also pull the model runs')
args = parser.parse_args()
ds = args.datastore

# pull: the files only the record has, without the model runs unless asked for
remote, pulled = {}, {}
latest = call(f'{URL}/api/records/{args.record}/versions/latest')['files'] if args.record else []
for f in latest:  # published records are public: no token is sent to the download
    url = f['links']['self']
    remote[f['key']] = {i.filename: i.CRC for i in zipfile.ZipFile(RemoteFile(url, f['size'])).infolist()}
    if f['key'].endswith('_model_results.zip') and not args.model_results:
        continue
    missing = [path for path in remote[f['key']] if not os.path.exists(f'{ds}/{path}')]
    if missing:  # one download of the zip is far fewer requests than one per file
        with tempfile.NamedTemporaryFile() as tmp:
            with urllib.request.urlopen(url) as r:
                shutil.copyfileobj(r, tmp)
            with zipfile.ZipFile(tmp.name) as z:
                for path in missing:
                    z.extract(path, ds)
                    pulled.setdefault('/'.join(path.split('/')[:2]), []).append(path)
if pulled:
    print('Pulled from Zenodo:')
    for case, paths in pulled.items():
        print(f'  {case:<{max(map(len, pulled))}}  {labels(paths)}'.rstrip())

# push: the cases that have files the record does not, or other ones; no model runs here is no change
changed = {}
for case in sorted(os.path.relpath(p, ds) for p in glob.glob(f'{ds}/*/*') if os.path.isdir(p)):
    files = crcs(ds, case)
    model = {path: crc for path, crc in files.items() if '/model_results/' in path}
    for key, part in ((case.replace('/', '_') + '.zip', {p: c for p, c in files.items() if p not in model}),
                      (case.replace('/', '_') + '_model_results.zip', model)):
        if part and part != remote.get(key):
            changed[key] = (case, part)
if not changed:
    print('Nothing to upload.' if pulled else 'The datastore and Zenodo are in sync.')
    sys.exit()

if not TOKEN:
    print('No Zenodo token found, so the datastore was only pulled.')
    print('These local results are not on Zenodo and were not uploaded:')
    for case in sorted({case for case, _ in changed.values()}):
        print(f'  {case}')
    print('To upload them, set ZENODO_TOKEN or save the token in sync/.token.')
    sys.exit()

if args.record:  # Zenodo allows one draft per record
    concept = call(f'{URL}/api/deposit/depositions/{args.record}')['conceptrecid']
    for d in call(f'{URL}/api/deposit/depositions?status=draft'):
        if d['conceptrecid'] == concept and not d['submitted']:
            sys.exit(f"the record has an open draft, publish or delete it first: {d['links']['html']}")
    draft = call(call(f'{URL}/api/deposit/depositions/{args.record}/actions/newversion', 'POST')['links']['latest_draft'])
else:
    draft = call(f'{URL}/api/deposit/depositions', 'POST', {})
old = {f['filename']: f['id'] for f in call(f"{URL}/api/deposit/depositions/{draft['id']}/files")}
with tempfile.TemporaryDirectory() as tmp:
    for key, (case, files) in changed.items():
        if key in old:
            call(f"{URL}/api/deposit/depositions/{draft['id']}/files/{old[key]}", 'DELETE')
        with zipfile.ZipFile(f'{tmp}/{key}', 'w', zipfile.ZIP_DEFLATED) as z:
            for path in files:
                z.write(f'{ds}/{path}', path)
        with open(f'{tmp}/{key}', 'rb') as f:
            call(f"{draft['links']['bucket']}/{key}", 'PUT', f, 'application/octet-stream', os.path.getsize(f'{tmp}/{key}'))
        print(f'pushed {case}  {labels(files)}')
print(f"draft {draft['id']}: publish it at {draft['links']['html']}")
