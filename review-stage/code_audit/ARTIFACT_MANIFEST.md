# Artifact Manifest Summary

Generated: 2026-06-08T11:02:46.050380+00:00
Root: /root/autodl-tmp/WFCLLM
Entries: 362

## Role Counts
- artifact: 1
- calibration: 3
- config: 13
- details: 27
- diagnostics: 144
- encoder_checkpoint: 11
- model: 35
- negative: 7
- result: 33
- run_state: 1
- summary: 27
- token_channel_artifact: 4
- watermarked: 56

## Global Notes
- run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'

## Mismatch / Missing Entries
- data/negative_corpus.jsonl (negative): exists=True; base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'
- data/negative_corpus_llm.jsonl (negative): exists=True; base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'
- data/results/humaneval_full_cap9_details.jsonl (details): exists=True; run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- data/results/humaneval_full_cap9_summary.json (summary): exists=True; run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- data/watermarked/humaneval_20260523_130234.jsonl (watermarked): exists=True; run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- data/models/codet5-base/tokenizer.json (model): exists=False; missing expected artifact

## Key Artifacts
- data/run_state.json: role=run_state, size=1111, rows=None, sha256=c231bf07f7c558449a4e2c884146df31f2198cf7a95c30246c6a2e05eb1dd08f
- data/models/encoder/best_model.pt: role=encoder_checkpoint, size=222067430, rows=None, sha256=6e340d66edcb8684504ae87e5d55425d9214ac00d16ac247198ea55abd0e3f5a
  Provenance: run_state.encoder.best_model_path
- data/checkpoints/encoder/encoder_epoch9.pt: role=encoder_checkpoint, size=227617024, rows=None, sha256=02cd9feaf288eb86253e33995f01407fc58e8388b90e1d2c8d2f970ed85a97dc
- data/models/token-channel/model.pt: role=token_channel_artifact, size=486015815, rows=None, sha256=9a67295a41b0edea5175ef7e5629d57d0977070325db7da33bf587d64ddfc85e
- data/models/token-channel/metadata.json: role=token_channel_artifact, size=638, rows=None, sha256=a5e09e70f9a2703ee56d11be859bb28ec0684888cd5c553441b60be84321380a
- data/watermarked/humaneval_20260523_130234.jsonl: role=watermarked, size=564800, rows=55, sha256=6a9f976a00ba63457844defb0596d45f2ac18badf5338255407b0ccaf782832e
  Provenance: run_state.watermark.output_file
  Mismatch: run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- data/watermarked/humaneval_full_cap9.jsonl: role=watermarked, size=1616308, rows=164, sha256=5e6ef96b308f51de9c93e9ea57f9374b108354f51eb59706e41677f42d43d548
- data/results/humaneval_full_cap9_details.jsonl: role=details, size=122449, rows=141, sha256=5ac56be5fe60ea67a82cfcf5af9fbf2d537eb9e56d86514edf3956ce521a5b83
  Provenance: run_state.extract.details_file
  Mismatch: run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- data/results/humaneval_full_cap9_summary.json: role=summary, size=1279, rows=None, sha256=312f74ec8806dac0d8ecba036a5e59265ddb377a9ae0542aa973291ae152667c
  Provenance: run_state.extract.summary_file; summary meta input_file=data/watermarked/humaneval_full_cap9.jsonl
  Mismatch: run_state pairing mismatch: watermark output stem 'humaneval_20260523_130234' != extract details input stem 'humaneval_full_cap9'
- data/negative_corpus.jsonl: role=negative, size=122283, rows=164, sha256=3f05cc2e453d32c7c67ad69f92959388373b946aafb73e8027e64ccdee17c7e0
  Mismatch: base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'
- data/negative_corpus_llm.jsonl: role=negative, size=322517, rows=164, sha256=f33e7eb91406516ee91c95158cbe9fa2b00b1ac12d7244d4b1344cf450f27bf2
  Provenance: run_state.generate-negative.output_file
  Mismatch: base_config extract.calibration_corpus 'data/negative_corpus.jsonl' differs from run_state generate-negative output 'data/negative_corpus_llm.jsonl'
