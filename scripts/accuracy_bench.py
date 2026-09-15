#!/usr/bin/env python
"""Compute accuracy metrics from prepared truth/call JSONL records.

The evaluator lives in ``advntr.background_evaluation`` so the installed
``fit-background`` command and this development benchmark use one implementation.
This script owns only JSONL input and safe report publication outside Git storage.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from advntr.background_evaluation import build_report, mcnemar_exact, wilson_ci


DEFAULT_OUTPUT_DIR = os.path.join('~', '.cache', 'advntr-bench')
OUTPUT_NAME = 'accuracy-report.json'


def load_records(path):
    """Read non-blank JSONL records; validation belongs to ``build_report``."""
    records = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError('%s:%d: invalid JSON: %s'
                                 % (path, line_number, error))
            records.append(record)
    return records


def _repository_boundaries():
    """Return every worktree and the common Git directory, or fail closed."""
    try:
        worktree_output = subprocess.check_output(
            ['git', 'worktree', 'list', '--porcelain', '-z'], cwd=REPO_ROOT)
        common_git = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'], cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('cannot establish repository output boundaries: %s'
                         % error)
    worktrees = [field[len('worktree '):]
                 for field in worktree_output.split('\0')
                 if field.startswith('worktree ')]
    if not worktrees or not common_git:
        raise ValueError('cannot establish repository output boundaries')
    if not os.path.isabs(common_git):
        common_git = os.path.join(REPO_ROOT, common_git)
    boundaries = worktrees + [common_git]
    return tuple(sorted(set(os.path.realpath(path) for path in boundaries)))


def _path_is_within(path, boundary):
    boundary_prefix = boundary.rstrip(os.sep) + os.sep
    return path == boundary or path.startswith(boundary_prefix)


def _resolved_external_path(path):
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    for boundary in _repository_boundaries():
        if _path_is_within(resolved, boundary):
            raise ValueError('output path must be outside Git storage: %s' % path)
    return resolved


def _file_identity(stat_result):
    return stat_result.st_dev, stat_result.st_ino


def _validate_directory_fd(directory_fd):
    """A descriptor pins the inode, not ancestry, so compare both filesystem views."""
    anchored_dir = '/proc/self/fd/%d' % directory_fd
    resolved_dir = _resolved_external_path(anchored_dir)
    try:
        opened_identity = _file_identity(os.fstat(directory_fd))
        resolved_identity = _file_identity(os.stat(resolved_dir))
    except OSError as error:
        raise ValueError('cannot validate opened output directory: %s' % error)
    if opened_identity != resolved_identity:
        raise ValueError('opened output directory moved during validation')
    return anchored_dir, resolved_dir


def _unlink_matching(directory_path, name, identity):
    """Avoid removing an older leaf absent a concurrent same-UID name swap."""
    path = os.path.join(directory_path, name)
    try:
        if _file_identity(os.stat(path)) == identity:
            os.unlink(path)
    except OSError:
        pass


def _best_effort(action, *args):
    try:
        action(*args)
    except BaseException:
        pass


def publish_report(report, output_dir, records_path=None):
    """Atomically publish in a pre-existing, descriptor-validated directory.

    Boundary placement linearizes at the final descriptor-based validation
    after rename; success also requires the following final-file identity check.
    Moves observed by those checks fail, with best-effort descriptor-anchored
    cleanup before close. An open descriptor pins an inode, not its ancestry,
    and stat-then-unlink cannot lock a leaf name. A same-UID process can move the
    directory or swap a leaf after/between checks, including after return, so
    persistent placement and cleanup cannot be guaranteed in that threat model.
    """
    resolved_dir = _resolved_external_path(output_dir)
    if not os.path.exists(resolved_dir):
        raise ValueError('output directory must already exist: %s' % output_dir)
    if not os.path.isdir(resolved_dir):
        raise ValueError('output path is not a directory: %s' % output_dir)
    directory_fd = None
    temporary_fd = None
    temporary_name = None
    temporary_identity = None
    rename_attempted = False
    failure = None
    published_path = None
    try:
        open_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        directory_fd = os.open(resolved_dir, open_flags)
        anchored_dir, _validated_dir = _validate_directory_fd(directory_fd)
        output_path = os.path.join(anchored_dir, OUTPUT_NAME)
        resolved_output = _resolved_external_path(output_path)
        if records_path is not None:
            resolved_records = os.path.realpath(os.path.abspath(
                os.path.expanduser(records_path)))
            if resolved_records == resolved_output:
                raise ValueError('records path collides with final report destination')

        temporary_fd, temporary_path = tempfile.mkstemp(
            prefix='.accuracy-report-', suffix='.json', dir=anchored_dir)
        temporary_name = os.path.basename(temporary_path)
        temporary_identity = _file_identity(os.fstat(temporary_fd))
        _validate_directory_fd(directory_fd)
        handle = os.fdopen(temporary_fd, 'w')
        temporary_fd = None
        with handle:
            json.dump(report, handle, sort_keys=True, indent=2,
                      separators=(',', ': '))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        _validate_directory_fd(directory_fd)
        rename_attempted = True
        os.rename(temporary_path, output_path)
        anchored_dir, validated_dir = _validate_directory_fd(directory_fd)
        if _file_identity(os.stat(output_path)) != temporary_identity:
            raise ValueError('published report identity changed during validation')
        published_path = os.path.join(validated_dir, OUTPUT_NAME)
    except BaseException:
        failure = sys.exc_info()
        if temporary_fd is not None:
            _best_effort(os.close, temporary_fd)
        if directory_fd is not None and temporary_identity is not None:
            anchored_dir = '/proc/self/fd/%d' % directory_fd
            _best_effort(
                _unlink_matching, anchored_dir, temporary_name,
                temporary_identity)
            if rename_attempted:
                _best_effort(
                    _unlink_matching, anchored_dir, OUTPUT_NAME,
                    temporary_identity)
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except BaseException:
                if failure is None:
                    raise
    if failure is not None:
        raise failure[0], failure[1], failure[2]
    return published_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True,
                        help='prepared truth/call records in JSONL format')
    parser.add_argument('--mode', choices=('baseline', 'compare'), required=True)
    parser.add_argument(
        '--out', help='external output directory; must already exist')
    args = parser.parse_args(argv)
    output_dir = (args.out or os.environ.get('ADVNTR_BENCH_OUT')
                  or DEFAULT_OUTPUT_DIR)
    try:
        records = load_records(args.records)
        report = build_report(records, compare=args.mode == 'compare')
        publish_report(report, output_dir, records_path=args.records)
    except (IOError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    sys.exit(main())
