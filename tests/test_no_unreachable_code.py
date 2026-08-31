"""No statements after an unconditional exit inside a function body.

Dead code that sits after a `return` is not merely untidy here: it never runs, so it
is never type-checked by execution and it rots silently. The instance that motivated
this test -- the tail of `get_ru_count_with_coverage_method` in vntr_finder.py --
referenced an `alignment_file` name that does not exist in its scope, so it was a
latent NameError that could never fire. Reviewers reading it reasonably assumed it
was live coverage-bias logic; it was the only reference to coverage_bias.py on any
runnable path.

Scanned by AST rather than by eye so the next one fails the build instead of
accumulating.
"""
import ast
import os
import unittest

# Files whose remaining unreachable blocks are known and deliberately not addressed
# here. Empty by intent: add an entry only with a written reason.
ALLOWED = {}

TERMINATORS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _package_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'advntr')


def _unreachable_statements(tree):
    """Yield (function name, lineno) for the first statement made unreachable by an
    unconditional terminator earlier in the same block."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef,)):
            continue
        for block in _blocks(node):
            for index, statement in enumerate(block[:-1]):
                if isinstance(statement, TERMINATORS):
                    found.append((node.name, block[index + 1].lineno))
                    break
    return found


def _blocks(node):
    """Every statement list reachable inside a function, so a return guarded by an
    `if` is not mistaken for an unconditional one."""
    blocks = []
    for inner in ast.walk(node):
        for field in ('body', 'orelse', 'finalbody'):
            block = getattr(inner, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                blocks.append(block)
    return blocks


class TestNoUnreachableCode(unittest.TestCase):

    def test_no_statement_follows_an_unconditional_exit(self):
        offenders = []
        package = _package_dir()
        for name in sorted(os.listdir(package)):
            if not name.endswith('.py'):
                continue
            path = os.path.join(package, name)
            with open(path) as handle:
                tree = ast.parse(handle.read(), filename=path)
            for function, lineno in _unreachable_statements(tree):
                if ALLOWED.get(name) == function:
                    continue
                offenders.append('%s:%d in %s()' % (name, lineno, function))
        self.assertEqual(
            offenders, [],
            'unreachable code after an unconditional return/raise:\n  '
            + '\n  '.join(offenders))


if __name__ == '__main__':
    unittest.main()
