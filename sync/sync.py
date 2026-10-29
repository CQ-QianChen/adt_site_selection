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


class NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """a redirect may lead to another host: the token is not sent there"""

    def redirect_request(self, *args):
        request = super().redirect_request(*args)
        if request:
            request.headers.pop('Authorization', None)
        return request


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
        with urllib.request.build_opener(NoAuthRedirect).open(urllib.request.Request(self.url, headers={**self.headers, **range})) as r:
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


def pull(ds, record, with_model):
    """download what only the record has; returns the zips of the record: name -> {path in the datastore: crc32}"""
    remote, pulled = {}, {}
    for f in call(f'{URL}/api/records/{record}/versions/latest')['files'] if record else []:
        url = f['links']['self']  # published records are public: no token is sent to the download
        remote[f['key']] = {i.filename: i.CRC for i in zipfile.ZipFile(RemoteFile(url, f['size'])).infolist()}
        if f['key'].endswith('_model_results.zip') and not with_model:
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
    return remote, bool(pulled)


def changes(ds, remote):
    """the local zips that differ from the record: name -> (case, {path: crc32}). Missing model runs are no change"""
    changed = {}
    for case in sorted(os.path.relpath(p, ds) for p in glob.glob(f'{ds}/*/*') if os.path.isdir(p)):
        files = crcs(ds, case)
        model = {path: crc for path, crc in files.items() if '/model_results/' in path}
        name = case.replace('/', '_')
        for key, part in ((f'{name}.zip', {p: c for p, c in files.items() if p not in model}), (f'{name}_model_results.zip', model)):
            if part and part != remote.get(key):
                changed[key] = (case, part)
    return changed


def push(ds, record, changed):
    """upload the changed zips into a draft of the record, a new one unless the record has an open draft (Zenodo allows one)"""
    if record:
        concept = call(f'{URL}/api/deposit/depositions/{record}')['conceptrecid']
        open_drafts = [d for d in call(f'{URL}/api/deposit/depositions?status=draft') if d['conceptrecid'] == concept and not d['submitted']]
        if open_drafts:
            draft = call(open_drafts[0]['links']['self'])
            print(f"Adding to the open draft {draft['id']}.")
        else:
            draft = call(call(f'{URL}/api/deposit/depositions/{record}/actions/newversion', 'POST')['links']['latest_draft'])
    else:
        draft = call(f'{URL}/api/deposit/depositions', 'POST', {})
    old = {f['filename']: f for f in call(f"{URL}/api/deposit/depositions/{draft['id']}/files")}
    with tempfile.TemporaryDirectory() as tmp:
        for key, (case, files) in changed.items():
            if key in old:  # a zip of the draft that already has all these files is left as it is
                draft_zip = RemoteFile(f"{draft['links']['bucket']}/{key}", int(old[key]['filesize']), {'Authorization': f'Bearer {TOKEN}'})
                if files.items() <= {i.filename: i.CRC for i in zipfile.ZipFile(draft_zip).infolist()}.items():
                    print(f'in the draft already: {case}  {labels(files)}')
                    continue
                call(f"{URL}/api/deposit/depositions/{draft['id']}/files/{old[key]['id']}", 'DELETE')
            with zipfile.ZipFile(f'{tmp}/{key}', 'w', zipfile.ZIP_DEFLATED) as z:
                for path in files:
                    z.write(f'{ds}/{path}', path)
            with open(f'{tmp}/{key}', 'rb') as f:
                call(f"{draft['links']['bucket']}/{key}", 'PUT', f, 'application/octet-stream', os.path.getsize(f'{tmp}/{key}'))
            print(f'pushed {case}  {labels(files)}')
    print(f"draft {draft['id']}: publish it at {draft['links']['html']}")


parser = argparse.ArgumentParser()
parser.add_argument('datastore')
parser.add_argument('--record')
parser.add_argument('--model-results', action='store_true', help='also pull the model runs')
args = parser.parse_args()

remote, pulled = pull(args.datastore, args.record, args.model_results)
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
    push(args.datastore, args.record, changed)
