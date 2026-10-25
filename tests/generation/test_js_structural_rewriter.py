from __future__ import annotations

import pytest

from wfcllm.gate.data import _make_request
from wfcllm.generation.js_structural_rewriter import KeyBlindJavaScriptStructuralWindowRewriter, _line_forms
from wfcllm.windowing import get_statement_unit_extractor


@pytest.mark.parametrize("line", [
    "return\n", "return /a/g;", "return `pre${x}`;", "const x=a,y=b;",
    "const x = 1; // trailing comment", "const x=(missing;", "const x=a/b;",
])
def test_ambiguous_or_unsupported_expressions_are_unavailable(line):
    assert _line_forms(line) == ()


@pytest.mark.parametrize(("prefix", "window", "suffix", "test"), [
    ("function dev(a,b){\n", "const x=a*7;\nconst y=b-9;\nreturn x+y;", "\n}\n",
     "console.assert(dev(3,11)===23);"),
    ("let state=0;function tick(){return ++state;}function dev(){\n",
     "let x=tick();\nlet y=x*4;\nreturn y;", "\n}\n", "console.assert(dev()===4&&state===1);"),
    ("function dev(){\n", "const named=()=>7;\nlet value=named.name;\nreturn value;", "\n}\n",
     'console.assert(dev()==="named");'),
    ("function dev(){let x;\n", "x=function(){};\nreturn x.name;", "\n}\n", 'console.assert(dev()==="x");'),
    ("function dev(){let x=11;\n", 'let value=eval("x+4");\nreturn value;', "\n}\n", "console.assert(dev()===15);"),
    ("function dev(){\n", 'const x="a,b";\nreturn x;', "\n}\n", 'console.assert(dev()==="a,b");'),
    ("function dev(x){\n", "switch(x){", "\ncase 2:return 7;\ndefault:return 0;\n}\n}\n",
     "console.assert(dev(2)===7&&dev(0)===0);"),
    ("let converted=0;function dev(){let x={valueOf(){++converted;return 0;}};\n", "if(x){",
     "\nreturn 1;\n}\nreturn 0;\n}\n", "console.assert(dev()===1&&converted===0);"),
    ("function dev(){\n", "return\n(7);", "\n}\n", "console.assert(dev()===undefined);"),
    ("function dev(){\n", "const x=3,y=4;\nreturn x+y;", "\n}\n", "console.assert(dev()===7);"),
])
def test_fixed_trajectory_preserves_authored_javascript_behavior(prefix, window, suffix, test):
    from wfcllm.evaluation.humanevalpack_execution import evaluate_native_row, require_native_toolchain

    try:
        toolchain = require_native_toolchain("js")
    except ValueError as error:
        pytest.skip(str(error))
    source = prefix + window + suffix
    extractor = get_statement_unit_extractor("js")
    units = tuple(extractor.extract(source))
    first, last = len(prefix.encode("utf-8")), len((prefix + window).encode("utf-8"))
    selected = tuple(u for u in units if first <= u.start_byte and u.end_byte <= last)
    request = _make_request(units, start_index=units.index(selected[0]), window=selected,
        prompt="Independent authored JavaScript example", source_bytes=source.encode("utf-8"),
        window_contract_version="js-statement-window/v1")
    rewriter = KeyBlindJavaScriptStructuralWindowRewriter(extractor=extractor, window_contract_version="js-statement-window/v1")
    trajectory = rewriter.rewrite_windows(request, candidate_indices=tuple(range(1, 19)))
    assert len(trajectory) == 18
    assert all(row.parse_status == "ok" and row.same_parent_scope and row.unit_count == len(selected) for row in trajectory)
    assert all(row.semantic_validation_rule is None and row.semantic_equivalence_certified is None for row in trajectory)
    assert trajectory == tuple(rewriter.rewrite_window(request, candidate_index=i) for i in range(1, 19))
    assert tuple(v.code for v in rewriter.rewrite_many(request, candidate_indices=tuple(range(1, 19)))) == tuple(v.text for v in trajectory)
    for row in trajectory:
        result = evaluate_native_row(final_code=prefix + row.text + suffix, test_code=test, toolchain=toolchain)
        assert result["is_correct"], result.get("error")
