from __future__ import annotations

from wfcllm.encoder.projection_training import prepare_region_blocks
from wfcllm.gate.production import GateSourceCatalogRecord


def test_cpp_training_views_use_exact_windows_and_keep_source_identity() -> None:
    code = 'int f(int a,int b){\n    int x=a*7;\n    int y=b-9;\n    return x+y;\n}\n'
    record = GateSourceCatalogRecord(source_family='oss_cpp', source_id='authored:cpp-view',
        code=code, repository_id='authored:cpp-repo', task_id=None, function_id=None,
        source_model_id=None, license_id='MIT', contract_or_hard_set=False, prompt='')
    blocks = prepare_region_blocks((record,), language='cpp', max_variants=8, max_perm_len=2)
    assert {block['source'].count(';') for block in blocks} >= {1, 2, 3}
    assert all(block['source_id'] == record.source_id for block in blocks)
    for block in blocks:
        positives = block['positive_variants']
        assert len(positives) == len(set(positives)) <= 8
        assert block['source'] not in positives
        assert all(text.count(';') == block['source'].count(';') for text in positives)
    returning = next(b for b in blocks if b['source'] == 'return x+y;')
    assert 'return (x+y);' in returning['positive_variants']
