from __future__ import annotations

import pytest

from wfcllm.generation.window_rewriter import _java_boolean_expression, _java_boolean_return


@pytest.mark.parametrize("expression", [
    '"a<b"', '"a&&b"', 'x > 0 ? 7 : 9', 'x >>> 1', 'x << 1',
    'new java.util.ArrayList<Integer>()', '(x > 0) ? "yes" : "no"',
])
def test_operator_spelling_does_not_certify_a_boolean_result(expression):
    assert _java_boolean_expression(expression) is False


@pytest.mark.parametrize("expression", ["true", "false", "x > 0", "(x > 0)", "x && y", "!x", "x == y"])
def test_boolean_root_expressions_remain_available(expression):
    assert _java_boolean_expression(expression) is True


@pytest.mark.parametrize(("result_type", "parameters", "expression", "assertion"), [
    ("String", "", '"a<b"', 'Dev.f().equals("a<b")'),
    ("int", "int x", "x > 0 ? 7 : 9", "Dev.f(3)==7&&Dev.f(-3)==9"),
    ("int", "int x", "x >>> 1", "Dev.f(-1)==2147483647"),
    ("boolean", "int x", "x > 0", "Dev.f(3)&&!Dev.f(-3)"),
    ("boolean", "boolean x,boolean y", "x && y", "Dev.f(true,true)&&!Dev.f(true,false)"),
])
def test_boolean_return_rules_preserve_authored_java_types(result_type, parameters, expression, assertion):
    from wfcllm.evaluation.humanevalpack_execution import evaluate_native_row, require_native_toolchain

    try:
        toolchain = require_native_toolchain("java")
    except ValueError as error:
        pytest.skip(str(error))
    original = "return " + expression + ";"
    for operator in ["&&", "||"]:
        candidate = _java_boolean_return(original, operator=operator)
        source = f"class Dev{{static {result_type} f({parameters}){{{candidate}}}}}"
        test = "public class Main{public static void main(String[] args){if(!(" + assertion + "))throw new AssertionError();}}"
        result = evaluate_native_row(final_code=source, test_code=test, toolchain=toolchain)
        assert result["is_correct"], result.get("error")
