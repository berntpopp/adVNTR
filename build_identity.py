"""Reproducible source attestations and fresh installed-payload build identities.

An sdist preserves a producer-verified revision only while every build-affecting input
still matches its source attestation. This is integrity under the trusted build-operator
model, not a signature authenticating an arbitrary archive author.
"""
import hashlib
import os
import re
import subprocess
import tempfile

from distutils.command.build import build as _build
from setuptools.command.sdist import sdist as _sdist

from advntr import __version__
from advntr.capabilities import (BUILD_IDENTITY_FILE, canonical_bytes, file_manifest,
                                 payload_build_id, payload_manifest, read_document,
                                 validate_source_identity)


SOURCE_IDENTITY_FILE = '_source_identity.json'
_BUILD_INPUTS = ('setup.py', 'build_config.py', 'build_identity.py', 'MANIFEST.in',
                 'hmm/base.pyx', 'hmm/base.pxd', 'hmm/hmm.pyx', 'hmm/cqueue.pxd',
                 'hmm/_viterbi_fill_core.pxi', 'hmm/queue.c', 'hmm/queue.h')


def source_manifest(root):
    """Hash production package sources and every native/build configuration input."""
    paths = list(_BUILD_INPUTS)
    for package in ('advntr', 'hmm'):
        for directory, subdirs, names in os.walk(os.path.join(root, package)):
            subdirs[:] = sorted(name for name in subdirs if name != '__pycache__')
            for name in names:
                if name.endswith('.py'):
                    paths.append(os.path.relpath(os.path.join(directory, name), root).replace(os.sep, '/'))
    return file_manifest(root, paths)


def _verified_revision(root, manifest):
    # Environment-supplied Git repositories/indexes or alleged revision strings are
    # not evidence about this source directory. Require the real repository and bytes.
    environment = dict((name, value) for name, value in os.environ.items() if not name.startswith('GIT_'))

    def git(*arguments):
        return subprocess.check_output(['git', '-C', root] + list(arguments),
                                       env=environment, stderr=subprocess.PIPE).strip()

    try:
        if os.path.realpath(git('rev-parse', '--show-toplevel')) != os.path.realpath(root):
            return None
        if git('status', '--porcelain', '--untracked-files=normal'):
            return None
        revision = git('rev-parse', 'HEAD')
        if re.match(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})\Z', revision) is None:
            return None
        for record in manifest:
            # Do not strip file bytes: final newlines are part of the source identity.
            original = subprocess.check_output(
                ['git', '-C', root, 'show', revision + ':' + record['path']],
                env=environment, stderr=subprocess.PIPE)
            if len(original) != record['size_bytes'] or hashlib.sha256(original).hexdigest() != record['sha256']:
                return None
        return revision
    except (OSError, subprocess.CalledProcessError):
        return None


def source_identity(root):
    """Resolve verified Git provenance or a matching producer sdist attestation."""
    manifest = source_manifest(root)
    source_path = os.path.join(root, SOURCE_IDENTITY_FILE)
    if os.path.lexists(source_path):
        if os.path.islink(source_path):
            raise ValueError('source identity cannot be a symlink')
        document = validate_source_identity(read_document(source_path))
        if document['files'] != manifest:
            raise ValueError('source identity differs from actual build inputs')
        return document
    return {'schema_version': 'advntr-source-identity-v1',
            'source_revision': _verified_revision(root, manifest),
            'files': manifest,
            'source_manifest_sha256': hashlib.sha256(canonical_bytes(manifest)).hexdigest()}


def write_build_identity(build_root, source):
    """Write non-recursive metadata after Python sources and extensions are built."""
    validate_source_identity(source)
    manifest = payload_manifest(build_root)
    document = {'schema_version': 'advntr-build-identity-v1',
                'package_version': __version__,
                'build_id': payload_build_id(manifest, hashlib.sha256(canonical_bytes(source)).hexdigest()),
                'files': manifest, 'source': source}
    path = os.path.join(build_root, 'advntr', BUILD_IDENTITY_FILE)
    with open(path, 'wb') as handle:
        handle.write(canonical_bytes(document) + '\n')


def build_commands(root, source):
    """Setuptools commands that retain source evidence and attest only fresh outputs."""
    class BuildWithIdentity(_build):
        def finalize_options(self):
            _build.finalize_options(self)
            if not os.path.isdir(self.build_base):
                os.makedirs(self.build_base)
            # Never reuse ignored or stale .so/.py outputs from a previous build.
            # A new private generated directory also avoids deleting user-supplied paths.
            self.build_lib = tempfile.mkdtemp(prefix='advntr-payload-', dir=self.build_base)
            self.force = True

        def run(self):
            _build.run(self)
            if source_manifest(root) != source['files']:
                raise ValueError('source changed during package build')
            write_build_identity(self.build_lib, source)

    class SdistWithIdentity(_sdist):
        def make_release_tree(self, base_dir, files):
            _sdist.make_release_tree(self, base_dir, files)
            if source_manifest(base_dir) != source['files']:
                raise ValueError('sdist omitted or changed a production build input')
            path = os.path.join(base_dir, SOURCE_IDENTITY_FILE)
            with open(path, 'wb') as handle:
                handle.write(canonical_bytes(source) + '\n')

    return {'build': BuildWithIdentity, 'sdist': SdistWithIdentity}
