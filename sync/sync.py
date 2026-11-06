"""Sync the datastore with a Zenodo record: pull what is only there, push what is only here.

The record holds one zip per case folder (<name>/<hash8>) with the datastore tree inside, and one more with its
model runs (model_results/), which are only pulled with --model-results.
A push goes into the open draft of the record, or makes a new version as a draft; publish it in the Zenodo web page.
With a token, a pull reads that draft too: it is newer than the published version.
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


class NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """a redirect may lead to another host: the token is not sent there"""

    def redirect_request(self, *args):
        request = super().redirect_request(*args)
        if request:
            request.headers.pop('Authorization', None)
        return request


OPEN = urllib.request.build_opener(NoAuthRedirect).open


class RemoteFile:
    """a file on the web that zipfile can read: every read is a range request"""

    def __init__(self, url, size, headers=None):
        self.url, self.size, self.headers, self.pos = url, size, headers or {}, 0

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if self.pos >= end:
            return b''
        range = {'Range': f'bytes={self.pos}-{end - 1}'}
        with OPEN(urllib.request.Request(self.url, headers={**self.headers, **range})) as r:
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


def open_draft(record):
    """the open draft of the record, which Zenodo allows only one of, or None; only its owner can see it"""
    if record and TOKEN:
        concept = call(f'{URL}/api/records/{record}')['conceptrecid']  # public: a token of another account can pull too
        drafts = [d for d in call(f'{URL}/api/deposit/depositions?status=draft&size=100') if d['conceptrecid'] == concept and not d['submitted']]
        return call(drafts[0]['links']['self']) if drafts else None


def pull(ds, record, with_model):
    """download what only the record has, from its open draft if there is one, else from the last published version.
    returns the zips there, name -> {path in the datastore: crc32}, whether anything was pulled, and the draft"""
    draft = open_draft(record)
    if draft:  # the zips of a draft are only for its owner: the token is sent
        auth = {'Authorization': f'Bearer {TOKEN}'}
        zips = [(f['filename'], f"{draft['links']['bucket']}/{f['filename']}", int(f['filesize']), auth)
                for f in call(f"{URL}/api/deposit/depositions/{draft['id']}/files")]
    else:  # published records are public: no token is sent to the download
        zips = [(f['key'], f['links']['self'], f['size'], {}) for f in call(f'{URL}/api/records/{record}/versions/latest')['files']] if record else []
    remote, pulled = {}, {}
    for key, url, size, headers in zips:
        if not key.endswith('.zip'):
            continue
        remote[key] = {i.filename: i.CRC for i in zipfile.ZipFile(RemoteFile(url, size, headers)).infolist()}
        if key.endswith('_model_results.zip') and not with_model:
            continue
        missing = [path for path in remote[key] if not os.path.exists(f'{ds}/{path}')]
        if missing:  # one download of the zip is far fewer requests than one per file
            with tempfile.NamedTemporaryFile() as tmp:
                with OPEN(urllib.request.Request(url, headers=headers)) as r:
                    shutil.copyfileobj(r, tmp)
                with zipfile.ZipFile(tmp.name) as z:
                    for path in missing:
                        z.extract(path, ds)
                        pulled.setdefault('/'.join(path.split('/')[:2]), []).append(path)
    if pulled:
        print(f"Pulled from the open draft {draft['id']}:" if draft else 'Pulled from Zenodo:')
        for case, paths in pulled.items():
            print(f'  {case:<{max(map(len, pulled))}}  {labels(paths)}'.rstrip())
    return remote, bool(pulled), draft


def changes(ds, remote):
    """the local zips that differ from the record: name -> (case, {path: crc32}). Missing model runs are no change"""
    changed = {}
    for case in sorted(os.path.relpath(p, ds) for p in glob.glob(f'{ds}/*/*') if os.path.isdir(p)):
        files = crcs(ds, case)
        model = {path: crc for path, crc in files.items() if '/model_results/' in path}
        name = case.replace('/', '_')
        for key, part in ((f'{name}.zip', {p: c for p, c in files.items() if p not in model}), (f'{name}_model_results.zip', model)):
            if part and not part.items() <= remote.get(key, {}).items():
                changed[key] = (case, part)
    return changed


def push(ds, record, changed, remote, draft):
    """upload the changed zips into the open draft, or into a new draft of the record, or of a new record"""
    for key, (case, files) in changed.items():
        if remote.get(key, {}).keys() - files.keys():  # the zip would lose files the datastore never pulled
            sys.exit(f'{case}: pull with --model-results first')
    if draft:
        print(f"Adding to the open draft {draft['id']}.")
    elif record:
        draft = call(call(f'{URL}/api/deposit/depositions/{record}/actions/newversion', 'POST')['links']['latest_draft'])
    else:
        draft = call(f'{URL}/api/deposit/depositions', 'POST', {})
    old = {f['filename']: f['id'] for f in call(f"{URL}/api/deposit/depositions/{draft['id']}/files")}
    with tempfile.TemporaryDirectory() as tmp:
        for key, (case, files) in changed.items():
            with zipfile.ZipFile(f'{tmp}/{key}', 'w', zipfile.ZIP_DEFLATED) as z:
                for path in files:
                    z.write(f'{ds}/{path}', path)
            if key in old:
                call(f"{URL}/api/deposit/depositions/{draft['id']}/files/{old[key]}", 'DELETE')
            with open(f'{tmp}/{key}', 'rb') as f:
                call(f"{draft['links']['bucket']}/{key}", 'PUT', f, 'application/octet-stream', os.path.getsize(f'{tmp}/{key}'))
            print(f'pushed {case}  {labels(files)}')
    print(f"draft {draft['id']}: publish it at {draft['links']['html']}")


parser = argparse.ArgumentParser()
parser.add_argument('datastore')
parser.add_argument('--record')
parser.add_argument('--model-results', action='store_true', help='also pull the model runs')
args = parser.parse_args()

remote, pulled, draft = pull(args.datastore, args.record, args.model_results)
changed = changes(args.datastore, remote)
if not changed:
    print('Nothing to upload.' if pulled else 'The datastore and Zenodo are in sync.')
elif not TOKEN:
    print('No Zenodo token found, so the datastore was only pulled.')
    print('These local results are not on Zenodo and were not uploaded:')
    for case in sorted({case for case, _ in changed.values()}):
        print(f'  {case}')
    print('To upload them, set ZENODO_TOKEN or save the token in sync/.token.')
else:
    push(args.datastore, args.record, changed, remote, draft)
