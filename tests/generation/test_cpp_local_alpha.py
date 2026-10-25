from __future__ import annotations

import pytest

from wfcllm.gate.data import RewriteRequest, _make_request
from wfcllm.generation.cpp_structural_rewriter import (
    KeyBlindCppStructuralWindowRewriter,
    _local_alpha_variants,
)
from wfcllm.windowing import get_statement_unit_extractor


def _request(prefix: str, window: str, suffix: str) -> RewriteRequest:
    extractor = get_statement_unit_extractor("cpp")
    source = prefix + window + suffix
    units = tuple(extractor.extract(source))
    first, last = len(prefix.encode("utf-8")), len((prefix + window).encode("utf-8"))
    selected = tuple(u for u in units if first <= u.start_byte and u.end_byte <= last)
    return _make_request(units, start_index=units.index(selected[0]), window=selected,
        prompt="Authored local binding example", source_bytes=source.encode("utf-8"),
        window_contract_version="cpp-statement-window/v1")


@pytest.mark.parametrize(("prefix", "window", "suffix"), [
    ("int f(){\n", "int value=3;", "\nreturn value;\n}\n"),
    ("int f(int value){\n", "int result=value*2;", "\nreturn result;\n}\n"),
    ("#define LABEL(x) #x\nint f(){\n", "int value=2;\nreturn value;", "\n}\n"),
    ("#include <cstdio>\nint f(){\n", "int value=2;\nstd::printf(\"%d\",value);\nreturn value;", "\n}\n"),
    ("int value=3;int f(){\n", "int value=4;\nreturn value+::value;", "\n}\n"),
    ("int f(){\n", "int value=3;\nint result=[&value](){int value=9;return value;}();\nreturn result;", "\n}\n"),
])
def test_alpha_does_not_cross_bindings_or_macro_boundaries(prefix, window, suffix):
    assert _local_alpha_variants(_request(prefix, window, suffix)) == ()


@pytest.mark.parametrize(("prefix", "window", "suffix", "test"), [
    ("struct Box{int value;};int f(){Box box{7};\n",
     "int value=3;\nint total=value+box.value;\nreturn total;", "\n}\n",
     "int main(){assert(f()==10);}"),
    ("int f(){\n", 'int value=4;\nconst char* label="value";\nreturn value+label[0];', "\n}\n",
     "int main(){assert(f()==4+'v');}"),
    ('int state=0;int tick(){return ++state;}int f(){const char* label="开发";\n',
     "int value=tick();\nint total=value*4;\nreturn total;", "\n}\n",
     "int main(){state=0;assert(f()==4);assert(state==1);}"),
])
def test_alpha_preserves_fields_strings_and_side_effects(tmp_path, prefix, window, suffix, test):
    from wfcllm.evaluation.cpp_execution import evaluate_cpp_row, require_cpp_toolchain

    try:
        toolchain = require_cpp_toolchain()
    except ValueError as error:
        pytest.skip(str(error))
    request = _request(prefix, window, suffix)
    rows = KeyBlindCppStructuralWindowRewriter(extractor=get_statement_unit_extractor("cpp"),
        window_contract_version="cpp-statement-window/v1").rewrite_windows(request, candidate_indices=tuple(range(1, 19)))
    assert len(rows) == 18
    assert all(row.parse_status == "ok" and row.same_parent_scope for row in rows)
    assert all(row.semantic_equivalence_certified is None for row in rows)
    assert all("_local" in row.text for row in rows[12:])
    for row in rows[12:]:
        result = evaluate_cpp_row(final_code=prefix + row.text + suffix,
            test_code="#undef NDEBUG\n#include<assert.h>\n" + test, toolchain=toolchain)
        assert result["is_correct"], result.get("error")


def test_alpha_avoids_identifiers_already_present_in_comments():
    request = _request("int f(){\n", "int value=2; // value_local1 reserved\nreturn value;", "\n}\n")
    variants = _local_alpha_variants(request)
    assert len(variants) == 5
    assert all("int value_local1=" not in text for text in variants)
    assert all("// value_local1 reserved" in text for text in variants)
