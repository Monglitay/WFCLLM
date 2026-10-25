from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest

from wfcllm.gate.data import _make_request
from wfcllm.generation.java_structural_rewriter import KeyBlindJavaStructuralWindowRewriter
from wfcllm.windowing import get_statement_unit_extractor


def _candidates(prefix: str, window: str, suffix: str):
    source = prefix + window + suffix
    extractor = get_statement_unit_extractor('java')
    units = tuple(extractor.extract(source))
    first, last = len(prefix.encode('utf-8')), len((prefix + window).encode('utf-8'))
    selected = tuple(u for u in units if first <= u.start_byte and u.end_byte <= last)
    assert selected
    request = _make_request(units, start_index=units.index(selected[0]), window=selected,
        prompt='Independent authored Java example', source_bytes=source.encode('utf-8'),
        window_contract_version='java-statement-window/v1')
    rewriter = KeyBlindJavaStructuralWindowRewriter(extractor=extractor,
        window_contract_version='java-statement-window/v1')
    candidates = rewriter.rewrite_windows(request, candidate_indices=tuple(range(1, 19)))
    assert all(v.parse_status == 'ok' and v.same_parent_scope and v.unit_count == len(selected) for v in candidates)
    assert all(v.semantic_equivalence_certified is None and v.semantic_validation_rule is None for v in candidates)
    assert all('/*' not in v.text and '//' not in v.text and 'Boolean.TRUE' not in v.text for v in candidates)
    for index, candidate in enumerate(candidates, 1):
        assert rewriter.rewrite_window(request, candidate_index=index) == candidate
        assert rewriter.rewrite(request, candidate_index=index).code == candidate.text
    return request, candidates


def test_java_structure_produces_distinct_scored_statement_combinations() -> None:
    request, candidates = _candidates('class Dev{int f(int a,int b){\n',
        'int x=a*7;\nint y=b-9;\nreturn x+y;', '\n}}\n')
    assert len({v.text for v in candidates}) == 18
    for candidate in candidates:
        completed = (request.completed_prefix + candidate.text).encode('utf-8')
        scored = completed[candidate.parser_spans[0][0]:candidate.parser_spans[-1][1]].decode('utf-8')
        assert scored == candidate.text


def test_java_structure_preserves_authored_types_and_effects(tmp_path: Path) -> None:
    javac, java = shutil.which('javac'), shutil.which('java')
    if not javac or not java:
        pytest.skip('Java native toolchain unavailable')
    cases = [
        ('byte', '', 'return 7;', 'f()==7'),
        ('short', '', 'return 32000;', 'f()==32000'),
        ('char', '', "return 'x';", "f()=='x'"),
        ('int', 'int x', 'return x>0?7:9;', 'f(3)==7&&f(-3)==9'),
        ('int', 'int x', 'return x>>>1;', 'f(-1)==2147483647'),
        ('String', '', 'return "a<b";', 'f().equals("a<b")'),
        ('String', '', 'return null;', 'f()==null'),
        ('boolean', 'int x', 'return x>0;', 'f(3)&&!f(-3)'),
        ('double', 'double x', 'double y=x*0.25;\ndouble z=y+0.625;\nreturn z;', 'f(3.0)==1.375'),
        ('int', 'int[] data', 'int x=data[0]++;\nint y=x+2;\nreturn y;', 'f(new int[]{7})==9'),
        ('String', 'String x', 'String y="开发"+x;\nString z=y.trim();\nreturn z;', 'f("x").equals("开发x")'),
    ]
    sources, assertions = [], []
    for number, (return_type, parameters, window, assertion) in enumerate(cases):
        prefix = f'class Dev{{static {return_type} f({parameters}){{\n'
        request, candidates = _candidates(prefix, window, '\n}}\n')
        for index, candidate in enumerate(candidates):
            name = f'Dev_{number}_{index}'
            sources.append((request.completed_prefix + candidate.text + '\n}}\n').replace('class Dev{', f'class {name}{{', 1))
            qualified = re.sub(r'\bf\(', name + '.f(', assertion)
            assertions.append(f'if(!({qualified}))throw new AssertionError("{name}");')
    sources.append('public class Main{public static void main(String[] args){' + ''.join(assertions) + '}}')
    program = tmp_path / 'Main.java'
    program.write_text('\n'.join(sources), encoding='utf-8')
    for command in ([javac, '-encoding', 'UTF-8', str(program)], [java, '-cp', str(tmp_path), 'Main']):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


def test_java_structure_rejects_another_language_contract() -> None:
    with pytest.raises(ValueError, match='Java window contract'):
        KeyBlindJavaStructuralWindowRewriter(extractor=get_statement_unit_extractor('js'),
            window_contract_version='js-statement-window/v1')
