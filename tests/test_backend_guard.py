"""The pomegranate backend is unsupported and must say so.

Without this guard, USE_ENHANCED_HMM=False imports pomegranate, whose __init__ calls
pyximport.install() and attempts a build at import time -- so the user gets an unrelated
scipy/BLAS gcc error instead of "this mode is not supported". See FORK.md.
"""
import unittest


class TestBackendGuard(unittest.TestCase):
    def test_disabling_enhanced_hmm_raises_a_clear_error(self):
        from advntr.hmm_utils import require_enhanced_hmm
        with self.assertRaises(RuntimeError) as ctx:
            require_enhanced_hmm(False)
        message = str(ctx.exception)
        self.assertIn('USE_ENHANCED_HMM', message)
        self.assertIn('unsupported', message.lower())
        self.assertIn('FORK.md', message)

    def test_enabled_backend_passes(self):
        from advntr.hmm_utils import require_enhanced_hmm
        require_enhanced_hmm(True)  # must not raise

    def test_the_live_backend_really_is_the_enhanced_one(self):
        """If this ever flips, every benchmark and equivalence claim is about other code."""
        from advntr import settings, vntr_finder
        self.assertTrue(settings.USE_ENHANCED_HMM)
        self.assertEqual(vntr_finder.Model.__module__, 'hmm.hmm')


if __name__ == '__main__':
    unittest.main()
