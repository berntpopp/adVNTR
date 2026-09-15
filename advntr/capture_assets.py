"""Verified private asset snapshots for completed calibration-capture records.

The queried SQLite bytes, rather than a mutable source filename, define the model
asset. V2 refuses active journals and WAL-format databases before copying; a
checkpointed rollback-journal source is required. Runtime payload and loaded
background identities are rechecked immediately before a completed capture.
"""
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile

from advntr.capabilities import canonical_bytes, describe_capabilities, read_document
from advntr.frameshift_background import BackgroundModel, load_background_model


_FLAGS = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)


def _metadata(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime, value.st_ctime)


def _sidecars(path):
    if not isinstance(path, basestring) or not path:
        raise ValueError('capture v2 model path must be nonempty text')
    if any(os.path.lexists(path + suffix) for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('capture v2 model has active SQLite sidecars; checkpoint and close writers first')


def _read_hash(descriptor):
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest, size = hashlib.sha256(), 0
    while True:
        data = os.read(descriptor, 1024 * 1024)
        if not data:
            return digest.hexdigest(), size
        digest.update(data)
        size += len(data)


def _open_regular(path):
    if not isinstance(path, basestring) or not path or not getattr(os, 'O_NOFOLLOW', 0):
        raise ValueError('capture v2 assets require a nonempty path and no-follow support')
    try:
        descriptor = os.open(path, _FLAGS)
    except OSError as error:
        raise ValueError('capture v2 asset is unreadable or a symlink: %s' % error)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError('capture v2 assets must be regular files')
    return descriptor


def _snapshot(source, destination, database=False):
    if database:
        _sidecars(source)
    descriptor = _open_regular(source)
    try:
        before = os.fstat(descriptor)
        digest, size, header = hashlib.sha256(), 0, ''
        output = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0600)
        try:
            os.fchmod(output, 0600)
            while True:
                data = os.read(descriptor, 1024 * 1024)
                if not data:
                    break
                if len(header) < 100:
                    header += data[:100 - len(header)]
                digest.update(data)
                size += len(data)
                offset = 0
                while offset < len(data):
                    written = os.write(output, data[offset:])
                    if written <= 0:
                        raise ValueError('capture v2 snapshot write did not complete')
                    offset += written
            os.fsync(output)
        finally:
            os.close(output)
        if database:
            _sidecars(source)
            if len(header) < 100 or header[:16] != 'SQLite format 3\x00' or header[18:20] != '\x01\x01':
                raise ValueError('capture v2 requires a valid rollback-journal SQLite model, not WAL format')
        second_hash, second_size = _read_hash(descriptor)
        after, named = os.fstat(descriptor), os.lstat(source)
        if (not stat.S_ISREG(named.st_mode) or _metadata(before) != _metadata(after)
                or _metadata(before) != _metadata(named) or size != before.st_size
                or second_size != size or second_hash != digest.hexdigest()):
            raise ValueError('capture v2 source changed while its snapshot was copied')
        os.chmod(destination, 0400)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_file(path, expected):
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        digest, size = _read_hash(descriptor)
        if (digest != expected or size != before.st_size or _metadata(before) != _metadata(os.fstat(descriptor))
                or _metadata(before) != _metadata(os.lstat(path))):
            raise ValueError('capture v2 snapshotted asset bytes changed')
    finally:
        os.close(descriptor)


def _producer():
    described = describe_capabilities()
    return dict((name, described[name]) for name in ('package_version', 'build_id', 'source_revision'))


def _background_document(background):
    if type(background) is not BackgroundModel:
        raise ValueError('capture v2 requires the native loaded background implementation')
    return {'schema': 'advntr.frameshift.background', 'version': background.version,
            'provenance': background.provenance, 'default_probability': background.default_probability,
            'states': dict(background.states)}


class CaptureAssets(object):
    """Run-owned snapshots; the source paths never enter the public document."""
    def __init__(self, root, model_path, model_sha256, background_path, background_sha256, background, producer):
        self.root, self.model_path = root, model_path
        self.background_path, self.background = background_path, background
        self.model_sha256, self.background_sha256 = model_sha256, background_sha256
        self._producer = canonical_bytes(producer)
        self._loaded_full = None if background is None else canonical_bytes(_background_document(background))
        semantic = None if background is None else _background_document(background)
        if semantic is not None:
            del semantic['provenance']
        self._loaded = None if semantic is None else canonical_bytes(semantic)
        self._probability_function = BackgroundModel.probability_for
        self._closed = False

    def document(self, background):
        """Recheck the exact runtime/background and bytes queried before completion."""
        if self._closed:
            raise ValueError('capture v2 assets are already closed')
        if background is not self.background:
            raise ValueError('capture v2 background differs from its loaded run asset')
        _verify_file(self.model_path, self.model_sha256)
        _sidecars(self.model_path)
        producer = _producer()
        if canonical_bytes(producer) != self._producer:
            raise ValueError('capture v2 runtime build identity changed during the run')
        if background is not None:
            _verify_file(self.background_path, self.background_sha256)
            if (canonical_bytes(_background_document(background)) != self._loaded_full
                    or getattr(background.probability_for, 'im_func', None) is not self._probability_function.im_func):
                raise ValueError('capture v2 loaded background content or implementation changed')
        return {'producer': producer, 'assets': {
            'model_sha256': self.model_sha256, 'background_sha256': self.background_sha256,
            'loaded_background_sha256': None if self._loaded is None else hashlib.sha256(self._loaded).hexdigest()}}

    def loaded_background_document(self):
        """Return fresh semantic background content, excluding source labels/paths."""
        self.document(self.background)
        return None if self._loaded is None else json.loads(self._loaded)

    def close(self):
        """Remove only this run's temporary asset directory, including on errors."""
        if not self._closed:
            self._closed = True
            shutil.rmtree(self.root)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()


def prepare_capture_assets(model_path, background_path=None):
    """Preflight and snapshot assets before creating a capture destination.

    Returns a context-managed asset owner. The caller uses its model_path and
    loaded background, and closes it after genotyping or any startup failure.
    """
    root = tempfile.mkdtemp(prefix='advntr-capture-assets-')
    os.chmod(root, 0700)
    try:
        model = os.path.join(root, 'model.db')
        model_hash = _snapshot(model_path, model, database=True)
        try:
            connection = sqlite3.connect(model)
            try:
                if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                    raise ValueError('capture v2 model SQLite integrity check failed')
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise ValueError('capture v2 model SQLite integrity check failed: %s' % error)
        snapshot, background_hash, background = None, None, None
        if background_path is not None:
            snapshot = os.path.join(root, 'background.json')
            background_hash = _snapshot(background_path, snapshot)
            read_document(snapshot)  # Duplicate fields/non-finite constants are forbidden in v2.
            background = load_background_model(snapshot)
        assets = CaptureAssets(root, model, model_hash, snapshot, background_hash, background, _producer())
        assets.document(background)
        return assets
    except BaseException:
        shutil.rmtree(root)
        raise
