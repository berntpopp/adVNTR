"""Strict local adapter for replaying complete frameshift capture-v2 records."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile

from advntr.frameshift_background import load_background_model
from advntr.capabilities import describe_capabilities
from advntr.frameshift_replay_policy import decode_replay_policy


_OPTIONS = frozenset(('capture_root', 'manifest', 'policy', 'background', 'output'))
_SHA256 = frozenset('0123456789abcdef')
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_READ_FLAGS = (os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) |
               getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0))


def _installed_producer():
    described = describe_capabilities()
    return dict((name, described[name]) for name in ('package_version', 'build_id', 'source_revision'))


def _load_renameat2():
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError):
        return None
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    return function


_RENAME_AT2 = _load_renameat2()


def add_replay_arguments(parser):
    parser.add_argument('--capture-root', required=True,
                        help='directory containing exactly the manifest-declared capture files')
    parser.add_argument('--manifest', required=True,
                        help='closed target roster and exact capture-file SHA256 digests')
    parser.add_argument('--policy', required=True,
                        help='closed capture and caller replay policy')
    parser.add_argument('--background', default=None,
                        help='frozen background model, required exactly for exact mode')
    parser.add_argument('--output', required=True,
                        help='new atomically published replay result directory')


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_replay_arguments(parser)
    validate_replay_arguments(parser, argv)
    return parser.parse_args(argv)


def validate_replay_arguments(parser, arguments):
    """Refuse Python 2 argparse abbreviation and repeated policy/asset options."""
    seen = set()
    for token in arguments:
        if token == '--':
            break
        parsed = parser._parse_optional(token)
        if parsed is None or parsed[0] is None:
            continue
        action, _option, _value = parsed
        if action.dest not in _OPTIONS:
            continue
        spelling = token.split('=', 1)[0]
        if spelling not in action.option_strings:
            parser.error('replay options require exact spelling: %s' % token)
        if action.dest in seen:
            parser.error('duplicate replay option: %s' % spelling)
        seen.add(action.dest)


def _reject_constant(value):
    raise ValueError('replay JSON contains non-finite constant %s' % value)


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('replay JSON contains a duplicate field')
        result[key] = value
    return result


def _decode_json(content, label):
    try:
        value = json.loads(content, object_pairs_hook=_strict_object_pairs,
                           parse_constant=_reject_constant)
    except (TypeError, ValueError) as error:
        raise ValueError('%s is not strict JSON: %s' % (label, error))
    if not isinstance(value, dict):
        raise ValueError('%s must contain one JSON object' % label)
    return value


def _metadata(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime, value.st_ctime)


def _read_descriptor(descriptor):
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b''.join(chunks)
        chunks.append(chunk)


def _read_regular(path, label):
    if not isinstance(path, basestring) or not path or not getattr(os, 'O_NOFOLLOW', 0):
        raise ValueError('%s requires a nonempty path and no-follow support' % label)
    try:
        descriptor = os.open(path, _READ_FLAGS)
    except OSError:
        raise ValueError('%s is unreadable or a symlink' % label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('%s must be a regular file' % label)
        content = _read_descriptor(descriptor)
        repeated = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        try:
            named = os.lstat(path)
        except OSError:
            raise ValueError('%s changed while it was read' % label)
        if (_metadata(before) != _metadata(after) or _metadata(before) != _metadata(named)
                or content != repeated or len(content) != before.st_size):
            raise ValueError('%s changed while it was read' % label)
        return content, hashlib.sha256(content).hexdigest()
    finally:
        os.close(descriptor)


def _digest(value, label):
    if not isinstance(value, basestring) or len(value) != 64 or any(char not in _SHA256 for char in value):
        raise ValueError('%s must be a lowercase SHA256 digest' % label)
    return value


def _manifest(document):
    if set(document) != set(('schema_version', 'captures')):
        raise ValueError('replay manifest fields differ from the closed contract')
    if document['schema_version'] != 'advntr-frameshift-replay-manifest-v1':
        raise ValueError('replay manifest schema version is unsupported')
    rows = document['captures']
    if not isinstance(rows, list) or not rows:
        raise ValueError('replay manifest requires a nonempty capture roster')
    checked = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(('key', 'filename', 'sha256', 'vntr_ids')):
            raise ValueError('replay manifest capture fields differ from the closed contract')
        key, filename, vntr_ids = row['key'], row['filename'], row['vntr_ids']
        if not isinstance(key, basestring) or not key or key.strip() != key:
            raise ValueError('replay manifest keys must be nonempty trimmed strings')
        if (not isinstance(filename, basestring) or not filename or filename in ('.', '..')
                or os.path.basename(filename) != filename):
            raise ValueError('replay manifest capture filename must be one safe basename')
        if (not isinstance(vntr_ids, list) or not vntr_ids
                or any(type(item) is not int or item <= 0 for item in vntr_ids)
                or vntr_ids != sorted(set(vntr_ids))):
            raise ValueError('replay manifest vntr_ids must be sorted unique positive integers')
        checked.append({'key': key, 'filename': filename, 'sha256': _digest(row['sha256'], 'capture SHA256'),
                        'vntr_ids': vntr_ids})
    if ([row['key'] for row in checked] != sorted(set(row['key'] for row in checked))
            or len(set(row['filename'] for row in checked)) != len(checked)):
        raise ValueError('replay manifest keys must be sorted unique and filenames distinct')
    return checked


def _load_capture(path, expected_sha256, expected_vntr_ids):
    content, digest = _read_regular(path, 'manifest-declared capture')
    if digest != expected_sha256:
        raise ValueError('manifest-declared capture SHA256 differs from its bytes')
    if not content.endswith(b'\n'):
        raise ValueError('manifest-declared capture ends with a partial JSONL record')
    records = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            raise ValueError('manifest-declared capture contains an empty JSONL record')
        records.append(_decode_json(line, 'manifest-declared capture record %d' % number))
    if not records:
        raise ValueError('manifest-declared capture contains no completed records')
    vntr_ids = []
    for record in records:
        locus = record.get('locus')
        value = locus.get('vntr_id') if isinstance(locus, dict) else None
        if type(value) is not int or value <= 0:
            raise ValueError('capture record locus vntr_id must be a positive integer')
        vntr_ids.append(value)
    if sorted(vntr_ids) != expected_vntr_ids or len(vntr_ids) != len(set(vntr_ids)):
        raise ValueError('capture records do not exactly match the manifest vntr_ids')
    return sorted(records, key=lambda record: record['locus']['vntr_id'])


def _replay_record(evaluator, record, policy_document, background):
    result = evaluator(record, policy_document, background)
    expected = record['locus']['vntr_id']
    if not isinstance(result, dict) or result.get('vntr_id') != expected:
        raise ValueError('replay result VNTR identifier differs from its capture record')
    return {'vntr_id': expected, 'result': result}


def _require_installed_producer(record, installed):
    producer = record.get('producer')
    if (not isinstance(producer, dict)
            or set(producer) != set(('package_version', 'build_id', 'source_revision'))
            or _canonical_content(producer) != _canonical_content(installed)):
        raise ValueError('capture producer differs from the installed replay producer')


def _canonical_content(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode('ascii')
    except (TypeError, ValueError) as error:
        raise ValueError('replay result is not finite JSON: %s' % error)


def _canonical_bytes(value):
    return _canonical_content(value) + b'\n'


def _write_private(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0600)
    try:
        os.fchmod(descriptor, 0600)
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise ValueError('replay output write did not complete')
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source, destination):
    if _RENAME_AT2 is None:
        raise RuntimeError('atomic replay output requires Linux renameat2 RENAME_NOREPLACE')
    ctypes.set_errno(0)
    if _RENAME_AT2(_AT_FDCWD, source, _AT_FDCWD, destination, _RENAME_NOREPLACE) == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise ValueError('replay output already exists')
    if number in (errno.ENOSYS, errno.EINVAL, getattr(errno, 'ENOTSUP', -1)):
        raise RuntimeError('replay output filesystem lacks renameat2 RENAME_NOREPLACE')
    raise OSError(number, os.strerror(number))


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) |
                         getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_output(output, producer):
    if not isinstance(output, basestring) or not output or os.path.basename(output) in ('', '.', '..'):
        raise ValueError('replay output must name a new directory')
    if os.path.lexists(output):
        raise ValueError('replay output already exists')
    if _RENAME_AT2 is None:
        raise RuntimeError('atomic replay output requires Linux renameat2 RENAME_NOREPLACE')
    destination = os.path.abspath(output)
    parent = os.path.dirname(destination)
    if not os.path.isdir(parent) or os.path.islink(parent):
        raise ValueError('replay output parent must be an existing non-symlink directory')
    parent_stat = os.stat(parent)
    identity = (parent_stat.st_dev, parent_stat.st_ino)
    staging = tempfile.mkdtemp(prefix='.%s.' % os.path.basename(output), dir=parent)
    os.chmod(staging, 0700)
    activated = False
    try:
        producer(staging)
        _fsync_directory(staging)
        current_parent = os.stat(parent)
        if (current_parent.st_dev, current_parent.st_ino) != identity:
            raise RuntimeError('replay output parent changed during production')
        _rename_noreplace(staging, destination)
        activated = True
        _fsync_directory(parent)
    finally:
        if not activated:
            shutil.rmtree(staging, ignore_errors=True)


def run(args, evaluator=None):
    """Replay every exact manifest member without genotyping or read access."""
    if evaluator is None:
        from advntr.frameshift_replay import replay_capture
        evaluator = replay_capture

    def produce(staging):
        manifest_bytes, manifest_file_sha = _read_regular(args.manifest, 'replay manifest')
        policy_bytes, policy_file_sha = _read_regular(args.policy, 'replay policy')
        manifest_document = _decode_json(manifest_bytes, 'replay manifest')
        policy_document = _decode_json(policy_bytes, 'replay policy')
        manifest_rows = _manifest(manifest_document)
        replay_policy = decode_replay_policy(policy_document)
        replay_producer = _installed_producer()
        mode = replay_policy.capture.caller_mode
        if mode == 'exact' and args.background is None:
            raise ValueError('exact replay policy requires --background')
        if mode == 'legacy' and args.background is not None:
            raise ValueError('legacy replay policy forbids --background')
        root = os.path.abspath(args.capture_root)
        if not os.path.isdir(root) or os.path.islink(root):
            raise ValueError('replay capture root must be a non-symlink directory')
        root_stat = os.stat(root)
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        if os.path.realpath(args.manifest).startswith(os.path.realpath(root) + os.sep):
            raise ValueError('replay manifest must be outside capture root')
        if set(os.listdir(root)) != set(row['filename'] for row in manifest_rows):
            raise ValueError('replay capture root inventory differs from the exact manifest')
        background, background_sha = None, None
        if args.background is not None:
            background_bytes, background_sha = _read_regular(args.background, 'replay background')
            _decode_json(background_bytes, 'replay background')
            snapshot = os.path.join(staging, '.background.json')
            _write_private(snapshot, background_bytes)
            background = load_background_model(snapshot)
            os.unlink(snapshot)
        results = []
        for row in manifest_rows:
            records = _load_capture(os.path.join(root, row['filename']), row['sha256'], row['vntr_ids'])
            for record in records:
                _require_installed_producer(record, replay_producer)
            results.append({'key': row['key'], 'capture_sha256': row['sha256'],
                            'vntrs': [_replay_record(evaluator, record, policy_document, background)
                                     for record in records]})
        final_root = os.stat(root)
        if (final_root.st_dev, final_root.st_ino) != root_identity:
            raise ValueError('replay capture root changed while evidence was read')
        output = {
            'schema_version': 'advntr-frameshift-replay-output-v1',
            'manifest_file_sha256': manifest_file_sha,
            'manifest_sha256': hashlib.sha256(_canonical_content(manifest_document)).hexdigest(),
            'policy_file_sha256': policy_file_sha,
            'policy_sha256': hashlib.sha256(_canonical_content(policy_document)).hexdigest(),
            'background_file_sha256': background_sha,
            'replay_producer': replay_producer,
            'results': results,
        }
        _write_private(os.path.join(staging, 'replay.json'), _canonical_bytes(output))

    _atomic_output(args.output, produce)
    return 0
