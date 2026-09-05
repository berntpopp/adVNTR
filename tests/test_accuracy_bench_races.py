"""Adversarial publication races for the external accuracy harness."""
import os
import shutil
import sys
import tempfile
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

_saved_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import accuracy_bench
finally:
    sys.dont_write_bytecode = _saved_dont_write_bytecode


class TestPublicationRaces(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix='advntr-accuracy-race-')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def _assert_forbidden_move_stops_at_checkpoint(
            self, owner_name, hook_name, next_os_name=None):
        bench = accuracy_bench
        output_dir = os.path.join(self.tempdir, 'race-output')
        forbidden_root = os.path.join(self.tempdir, 'forbidden')
        moved_output = os.path.join(forbidden_root, 'moved-output')
        os.mkdir(output_dir)
        os.mkdir(forbidden_root)
        owner = getattr(bench, owner_name)
        real_hook = getattr(owner, hook_name)
        real_rename = bench.os.rename
        real_boundaries = bench._repository_boundaries
        real_next = (getattr(bench.os, next_os_name)
                     if next_os_name is not None else None)
        moved = [False]
        next_calls = [0]

        def move_after_hook(*args, **kwargs):
            result = real_hook(*args, **kwargs)
            real_rename(output_dir, moved_output)
            moved[0] = True
            return result

        def forbidden_next_stage(*_args, **_kwargs):
            next_calls[0] += 1
            raise AssertionError('next stage ran after forbidden move')

        bench._repository_boundaries = lambda: (forbidden_root,)
        setattr(owner, hook_name, move_after_hook)
        if next_os_name is not None:
            setattr(bench.os, next_os_name, forbidden_next_stage)
        try:
            with self.assertRaises(ValueError):
                bench.publish_report({'mode': 'baseline'}, output_dir)
        finally:
            setattr(owner, hook_name, real_hook)
            if next_os_name is not None:
                setattr(bench.os, next_os_name, real_next)
            bench._repository_boundaries = real_boundaries
        self.assertTrue(moved[0])
        self.assertEqual(next_calls[0], 0)
        self.assertTrue(os.path.isdir(moved_output))
        self.assertEqual(os.listdir(moved_output), [])

    def test_move_during_temporary_creation_stops_before_fdopen(self):
        self._assert_forbidden_move_stops_at_checkpoint(
            'tempfile', 'mkstemp', 'fdopen')

    def test_move_after_fsync_stops_before_final_rename(self):
        self._assert_forbidden_move_stops_at_checkpoint(
            'os', 'fsync', 'rename')

    def test_move_during_final_rename_is_rolled_back(self):
        self._assert_forbidden_move_stops_at_checkpoint('os', 'rename')

    def test_interrupt_after_rename_leaves_preexisting_directory_empty(self):
        bench = accuracy_bench
        output_dir = os.path.join(self.tempdir, 'interrupted-output')
        os.mkdir(output_dir)
        real_rename = bench.os.rename

        def interrupt_after_rename(*args, **kwargs):
            real_rename(*args, **kwargs)
            raise KeyboardInterrupt('primary interrupt')

        bench.os.rename = interrupt_after_rename
        try:
            with self.assertRaises(KeyboardInterrupt) as raised:
                bench.publish_report({'mode': 'baseline'}, output_dir)
        finally:
            bench.os.rename = real_rename
        self.assertEqual(str(raised.exception), 'primary interrupt')
        self.assertEqual(os.listdir(output_dir), [])

    def test_missing_output_never_reaches_close_time_substitution(self):
        bench = accuracy_bench
        output_dir = os.path.join(self.tempdir, 'missing-output')
        moved_output = os.path.join(self.tempdir, 'moved-output')
        unrelated = os.path.join(self.tempdir, 'unrelated')
        os.mkdir(unrelated)
        real_close = bench.os.close
        real_rename = bench.os.rename
        real_boundaries = bench._repository_boundaries
        substituted = [False]

        def fail_rename(*_args, **_kwargs):
            raise ValueError('synthetic publication failure')

        def substitute_on_close(descriptor):
            real_rename(output_dir, moved_output)
            real_rename(unrelated, output_dir)
            substituted[0] = True
            return real_close(descriptor)

        bench.os.rename = fail_rename
        bench.os.close = substitute_on_close
        bench._repository_boundaries = lambda: (moved_output,)
        try:
            with self.assertRaises(ValueError):
                bench.publish_report({'mode': 'baseline'}, output_dir)
        finally:
            bench.os.rename = real_rename
            bench.os.close = real_close
            bench._repository_boundaries = real_boundaries
        self.assertFalse(substituted[0])
        self.assertTrue(os.path.isdir(unrelated))
        self.assertFalse(os.path.lexists(output_dir))
        self.assertFalse(os.path.lexists(moved_output))

    def test_close_substitution_preserves_unrelated_and_cleans_original(self):
        bench = accuracy_bench
        output_dir = os.path.join(self.tempdir, 'created-output')
        moved_output = os.path.join(self.tempdir, 'moved-output')
        unrelated = os.path.join(self.tempdir, 'unrelated')
        os.mkdir(output_dir)
        os.mkdir(unrelated)
        unrelated_inode = os.stat(unrelated).st_ino
        real_close = bench.os.close
        real_rename = bench.os.rename
        real_boundaries = bench._repository_boundaries
        substituted = [False]

        def fail_rename(*_args, **_kwargs):
            raise ValueError('synthetic publication failure')

        def substitute_on_close(descriptor):
            real_rename(output_dir, moved_output)
            real_rename(unrelated, output_dir)
            substituted[0] = True
            return real_close(descriptor)

        bench.os.rename = fail_rename
        bench.os.close = substitute_on_close
        bench._repository_boundaries = lambda: (moved_output,)
        try:
            with self.assertRaises(ValueError):
                bench.publish_report({'mode': 'baseline'}, output_dir)
        finally:
            bench.os.rename = real_rename
            bench.os.close = real_close
            bench._repository_boundaries = real_boundaries
        self.assertTrue(substituted[0])
        self.assertEqual(os.stat(output_dir).st_ino, unrelated_inode)
        self.assertTrue(os.path.isdir(moved_output))
        self.assertEqual(os.listdir(moved_output), [])

    def test_secondary_cleanup_failures_preserve_primary_and_traceback(self):
        bench = accuracy_bench
        output_dir = os.path.join(self.tempdir, 'exception-output')
        forbidden_root = os.path.join(self.tempdir, 'forbidden')
        os.mkdir(output_dir)
        os.mkdir(forbidden_root)
        primary = RuntimeError('primary publication failure')
        real_dump = bench.json.dump
        real_unlink = bench.os.unlink
        real_close = bench.os.close
        real_boundaries = bench._repository_boundaries
        unlink_calls = [0]
        close_calls = [0]

        def raise_primary(*_args, **_kwargs):
            raise primary

        def raise_secondary_unlink(*_args, **_kwargs):
            unlink_calls[0] += 1
            raise KeyboardInterrupt('secondary unlink failure')

        def close_then_raise_secondary(descriptor):
            close_calls[0] += 1
            real_close(descriptor)
            raise OSError('secondary close failure')

        bench.json.dump = raise_primary
        bench.os.unlink = raise_secondary_unlink
        bench.os.close = close_then_raise_secondary
        bench._repository_boundaries = lambda: (forbidden_root,)
        caught = None
        caught_traceback = None
        try:
            try:
                bench.publish_report({'mode': 'baseline'}, output_dir)
            except BaseException as error:
                caught = error
                caught_traceback = sys.exc_info()[2]
        finally:
            bench.json.dump = real_dump
            bench.os.unlink = real_unlink
            bench.os.close = real_close
            bench._repository_boundaries = real_boundaries
        frame_names = []
        while caught_traceback is not None:
            frame_names.append(caught_traceback.tb_frame.f_code.co_name)
            caught_traceback = caught_traceback.tb_next
        self.assertIs(caught, primary)
        self.assertEqual(str(caught), 'primary publication failure')
        self.assertIn('raise_primary', frame_names)
        self.assertEqual(unlink_calls[0], 1)
        self.assertEqual(close_calls[0], 1)


if __name__ == '__main__':
    unittest.main()
