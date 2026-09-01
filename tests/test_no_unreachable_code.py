"""No statements after an unconditional exit inside a function body.

Dead code that sits after a statically unconditional exit is not merely untidy here: it never runs, so it
is never type-checked by execution and it rots silently. The instance that motivated
this test -- the tail of `get_ru_count_with_coverage_method` in vntr_finder.py --
referenced an `alignment_file` name that does not exist in its scope, so it was a
latent NameError that could never fire. Reviewers reading it reasonably assumed it
was live coverage-bias logic; it was the only reference from `VNTRFinder` and the supported
path. The unsupported plotting module still has its own local import.

Scanned by AST rather than by eye so the next direct terminator, or `if/else` whose
branches both terminate, fails the build instead of accumulating. This is deliberately
not advertised as a complete control-flow graph: constant-condition loops and every
possible `try`/`finally` exit are outside its contract.
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
    """Yield the first statement after a direct or two-branch unconditional exit."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef,)):
            continue
        for block in _blocks(node):
            for index, statement in enumerate(block[:-1]):
                if _statement_terminates(statement):
                    found.append((node.name, block[index + 1].lineno))
                    break
    return found


def _statement_terminates(statement):
    """Whether reaching `statement` guarantees exit from its containing block.

    Direct terminators are obvious. An `if` is terminal only when it has an `else`
    and every reachable route through both branches terminates; a guarded return with
    no `else` therefore remains non-terminal in the enclosing block.
    """
    if isinstance(statement, TERMINATORS):
        return True
    if isinstance(statement, ast.If) and statement.orelse:
        return (_block_terminates(statement.body)
                and _block_terminates(statement.orelse))
    return False


def _block_terminates(block):
    for statement in block:
        if _statement_terminates(statement):
            return True
    return False


def _blocks(node):
    """Every statement list reachable inside a function, so a return guarded by an
    `if` is not mistaken for an unconditional one. Nested function bodies belong to
    their own `FunctionDef` and must not be attributed to the enclosing function."""
    blocks = []

    def visit(inner):
        for _field, value in ast.iter_fields(inner):
            if isinstance(value, list):
                if value and isinstance(value[0], ast.stmt):
                    blocks.append(value)
                for child in value:
                    if isinstance(child, ast.FunctionDef) and child is not node:
                        continue
                    if isinstance(child, ast.AST):
                        visit(child)
            elif isinstance(value, ast.AST):
                if isinstance(value, ast.FunctionDef) and value is not node:
                    continue
                visit(value)

    visit(node)
    return blocks


class TestNoUnreachableCode(unittest.TestCase):

    def _scan(self, source):
        return _unreachable_statements(ast.parse(source))

    def test_a_return_guarded_by_an_if_does_not_exit_the_enclosing_block(self):
        source = ('def guarded(value):\n'
                  '    if value:\n'
                  '        return 1\n'
                  '    consume(value)\n')
        self.assertEqual(self._scan(source), [])

    def test_two_terminating_if_branches_make_the_following_statement_unreachable(self):
        source = ('def both_branches(value):\n'
                  '    if value:\n'
                  '        return 1\n'
                  '    else:\n'
                  '        raise ValueError()\n'
                  '    consume(value)\n')
        self.assertEqual(self._scan(source), [('both_branches', 6)])

    def test_a_nested_function_is_reported_only_under_its_own_name(self):
        source = ('def outer():\n'
                  '    def inner():\n'
                  '        return 1\n'
                  '        consume()\n'
                  '    return inner\n')
        self.assertEqual(self._scan(source), [('inner', 4)])

    def test_no_statement_follows_a_recognised_unconditional_exit(self):
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
