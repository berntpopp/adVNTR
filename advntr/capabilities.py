"""Installed feature discovery bound to runtime bytes, never to a working directory.

Build identity is deterministic for an exact payload, not a promise that different
compilers produce identical binaries. Bytecode caches and generated identity documents
are excluded. A source checkout without an attested build is research-only even when
Git is clean: ignored compiled extensions need not have been built from that revision.
"""
import hashlib
import json
import os
import re
import stat

from advntr import __version__


BUILD_IDENTITY_FILE = '_build_identity.json'
_DIGEST = re.compile(r'^[0-9a-f]{64}\Z')
_REVISION = re.compile(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})\Z')


def canonical_bytes(document):
    """Stable JSON bytes with no timestamps, paths or non-finite numbers added."""
    return json.dumps(document, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def read_document(path):
    """Refuse corrupt or ambiguous present metadata instead of guessing identity."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate identity metadata key')
            result[key] = value
        return result

    def constant(_value):
        raise ValueError('non-finite identity metadata number')

    with open(path, 'rb') as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=constant)


def file_manifest(root, paths):
    """Hash an exact relative file list; symlinks cannot hide another payload."""
    records = []
    for name in sorted(paths):
        if (not isinstance(name, basestring) or '\\' in name or ':' in name
                or name.startswith('/') or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('identity file paths must be normalized relative paths')
        current = root
        for part in name.split('/'):
            current = os.path.join(current, part)
            if os.path.islink(current):
                raise ValueError('identity payload cannot contain symlinks')
        if not stat.S_ISREG(os.stat(current).st_mode):
            raise ValueError('identity payload files must be regular files')
        digest = hashlib.sha256()
        size = 0
        with open(current, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), ''):
                digest.update(chunk)
                size += len(chunk)
        records.append({'path': name, 'size_bytes': size, 'sha256': digest.hexdigest()})
    validate_manifest(records)
    return records


def validate_manifest(records):
    """Validate closed canonical file records used in source and runtime identities."""
    if not isinstance(records, list) or not records:
        raise ValueError('identity requires a non-empty file manifest')
    names = []
    for row in records:
        if not isinstance(row, dict) or set(row) != set(('path', 'size_bytes', 'sha256')):
            raise ValueError('identity manifest record fields differ')
        name = row['path']
        if (not isinstance(name, basestring) or '\\' in name or ':' in name
                or name.startswith('/') or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('identity file paths must be normalized relative paths')
        if (isinstance(row['size_bytes'], bool) or not isinstance(row['size_bytes'], (int, long))
                or row['size_bytes'] < 0):
            raise ValueError('identity file size must be a non-negative integer')
        if not isinstance(row['sha256'], basestring) or not _DIGEST.match(row['sha256']):
            raise ValueError('identity file digest must be lowercase SHA256')
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError('identity manifest paths must be sorted and unique')


def payload_manifest(root):
    """Inventory package Python sources and actual production extension bytes."""
    paths = []
    for package in ('advntr', 'hmm'):
        directory = os.path.join(root, package)
        if not os.path.isdir(directory) or os.path.islink(directory):
            raise ValueError('runtime identity requires regular advntr and hmm package directories')
        for current, directories, filenames in os.walk(directory):
            for child in directories:
                if os.path.islink(os.path.join(current, child)):
                    raise ValueError('identity payload cannot contain symlinks')
            directories[:] = sorted(name for name in directories if name != '__pycache__')
            for name in filenames:
                if name.endswith(('.py', '.so')):
                    paths.append(os.path.relpath(os.path.join(current, name), root).replace(os.sep, '/'))
    return file_manifest(root, paths)


def payload_build_id(manifest, source_identity_sha256=None):
    """SHA256 of the versioned exact runtime payload; metadata never hashes itself."""
    validate_manifest(manifest)
    if source_identity_sha256 is not None and (
            not isinstance(source_identity_sha256, basestring) or not _DIGEST.match(source_identity_sha256)):
        raise ValueError('source identity binding must be a lowercase SHA256 digest or null')
    return hashlib.sha256(canonical_bytes({
        'schema_version': 'advntr-runtime-payload-v1', 'files': manifest,
        'source_identity_sha256': source_identity_sha256})).hexdigest()


def validate_source_identity(document):
    """Check source attestation structure and its non-recursive manifest binding."""
    if (not isinstance(document, dict) or set(document) != set((
            'schema_version', 'source_revision', 'files', 'source_manifest_sha256'))
            or document['schema_version'] != 'advntr-source-identity-v1'):
        raise ValueError('invalid source identity schema')
    revision = document['source_revision']
    if revision is not None and (not isinstance(revision, basestring) or not _REVISION.match(revision)):
        raise ValueError('invalid verified source revision')
    validate_manifest(document['files'])
    if hashlib.sha256(canonical_bytes(document['files'])).hexdigest() != document['source_manifest_sha256']:
        raise ValueError('source identity manifest digest differs')
    return document


def describe_capabilities(package_root=None):
    """Return the closed capability ABI after validating any present build metadata."""
    root = (os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            if package_root is None else os.path.realpath(package_root))
    manifest = payload_manifest(root)
    build_id = payload_build_id(manifest)
    revision = None
    identity_path = os.path.join(root, 'advntr', BUILD_IDENTITY_FILE)
    if os.path.lexists(identity_path):
        if os.path.islink(identity_path):
            raise ValueError('build identity cannot be a symlink')
        identity = read_document(identity_path)
        if (not isinstance(identity, dict) or set(identity) != set((
                'schema_version', 'package_version', 'build_id', 'files', 'source'))
                or identity['schema_version'] != 'advntr-build-identity-v1'
                or identity['package_version'] != __version__):
            raise ValueError('invalid build identity schema or package version')
        validate_manifest(identity['files'])
        source = validate_source_identity(identity['source'])
        build_id = payload_build_id(manifest, hashlib.sha256(canonical_bytes(source)).hexdigest())
        if identity['files'] != manifest or identity['build_id'] != build_id:
            raise ValueError('build identity differs from actual runtime payload')
        revision = source['source_revision']
    return {
        'schema_version': 'advntr-capabilities-v1',
        'package_version': __version__,
        'build_id': build_id,
        'source_revision': revision,
        'capabilities': ['fit-background-installed-v1', 'frameshift-calibration-capture-v1',
                         'frameshift-run-local-policy-v1'],
        'capture_schema_versions': [1],
        'policy_schema_versions': [],
        'background_recipe_ids': ['recipe-v1'],
    }
