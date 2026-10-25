from __future__ import annotations

import pytest

from wfcllm.gate.data import _make_request
from wfcllm.generation.cpp_structural_rewriter import KeyBlindCppStructuralWindowRewriter
from wfcllm.windowing import get_statement_unit_extractor


def _trajectory(prefix: str, window: str, suffix: str):
    extractor = get_statement_unit_extractor('cpp')
    source = prefix + window + suffix
    units = tuple(extractor.extract(source))
    start, end = len(prefix.encode('utf-8')), len((prefix + window).encode('utf-8'))
    selected = tuple(unit for unit in units if start <= unit.start_byte and unit.end_byte <= end)
    request = _make_request(units, start_index=units.index(selected[0]), window=selected,
                            prompt='Independent authored compiler fixture', source_bytes=source.encode('utf-8'),
                            window_contract_version='cpp-statement-window/v1')
    rewriter = KeyBlindCppStructuralWindowRewriter(extractor=extractor, window_contract_version='cpp-statement-window/v1')
    return rewriter.rewrite_windows(request, candidate_indices=tuple(range(1, 19)))


def test_composition_preserves_all_twelve_existing_variants_before_new_trials():
    window = 'int x = a + 1;\nint y = b + 2;\nreturn x + y;'
    rows = _trajectory('int f(int a,int b){\n', window, '\n}\n')
    expected = []
    for expression in ['a + 1', 'b + 2', 'x + y']:
        for variant in [f'({expression})', f'(({expression}))', f'((({expression})))', f'((void)0, ({expression}))']:
            expected.append(window.replace(expression, variant, 1))
    assert [row.text for row in rows[:12]] == expected
    assert len(rows) == len({row.text for row in rows}) == 18
    assert all(row.parse_status == 'ok' and row.same_parent_scope for row in rows)
    assert all(row.semantic_validation_rule is None and row.semantic_equivalence_certified is None for row in rows)
    assert all('wfcllm:' not in row.text for row in rows)


@pytest.mark.parametrize(('prefix', 'window', 'suffix', 'test'), [
    ('int f(int a,int b){\n', 'int x = a + 1;\nint y = b + 2;\nreturn x + y;', '\n}\n',
     'int main(){for(int a=-3;a<=3;++a)for(int b=-3;b<=3;++b)assert(f(a,b)==a+b+3);}'),
    ('int counter;int f(){\n', 'int x = ++counter;\nint y = ++counter;\nreturn x + y;', '\n}\n',
     'int main(){counter=0;assert(f()==3);assert(counter==2);}'),
    ('int& f(int& a,int& b){\n', 'a += 2;\nb += 3;\nreturn a;', '\n}\n',
     'int main(){int a=1,b=2;int& r=f(a,b);assert(&r==&a);assert(a==3&&b==5);}'),
])
def test_composed_candidates_preserve_authored_behavior_and_evaluation_count(tmp_path, prefix, window, suffix, test):
    from wfcllm.evaluation.cpp_execution import evaluate_cpp_row, require_cpp_toolchain

    try:
        toolchain = require_cpp_toolchain()
    except ValueError as error:
        pytest.skip(str(error))
    # These assertions execute only after the entire fixed trajectory exists.
    rows = _trajectory(prefix, window, suffix)
    assert len(rows) == 18
    for row in rows:
        result = evaluate_cpp_row(final_code=prefix + row.text + suffix,
                                  test_code='#undef NDEBUG\n#include<assert.h>\n' + test, toolchain=toolchain)
        assert result['is_correct'], result.get('error')


def test_single_class_return_keeps_named_return_elision_candidates_only():
    rows = _trajectory('struct Box{};Box f(){Box value;\n', 'return value;', '\n}\n')
    expected = ['return (value);', 'return ((value));', 'return (((value)));'] * 6
    assert [row.text for row in rows] == expected
