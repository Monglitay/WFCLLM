from wfcllm.extract.param_resolution import resolve_extract_lsh_params


def test_extract_prefers_embedded_metadata_over_stale_config():
    record = {"watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75}}
    ext_cfg = {"lsh_d": 3, "lsh_gamma": 0.5}
    resolved = resolve_extract_lsh_params(record, ext_cfg)
    assert resolved == (4, 0.75)
