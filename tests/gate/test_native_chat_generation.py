from __future__ import annotations

from pathlib import Path
import json

import pytest

from wfcllm.gate.production import (
    LocalHFProgramGenerator,
    _extract_native_chat_code,
    _native_chat_program_prompt,
)


@pytest.mark.parametrize(('language', 'sample_id', 'code'), [
    ('Java', 'Java/7', 'class Solution { int wrong(int x) { return x; } }'),
    ('JavaScript', 'JavaScript/7', 'const wrong = x => x;\nconsole.log(wrong(1));'),
])
def test_native_chat_retains_raw_response_and_never_repairs_wrong_code(language, sample_id, code):
    import torch

    response = f'```{language.lower()}\n{code}\n```\nUnrelated prose.'
    prompt = 'PUBLIC PREFIX'

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True)
            assert messages[0]['role'] == 'user'
            text = messages[0]['content']
            assert prompt in text and language in text
            assert 'Preserve its exact function, class and parameter names' in text
            return '<rendered-native-prompt>'

        def __call__(self, text, **kwargs):
            assert text == '<rendered-native-prompt>'
            return dict(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [7, 8]
            return response

    class Model:
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            assert kwargs['do_sample'] is False and kwargs['max_new_tokens'] == 512
            return torch.tensor([[10, 11, 7, 8]])

    runtime = object.__new__(LocalHFProgramGenerator)
    runtime.model, runtime.tokenizer = Model(), Tokenizer()
    runtime.device, runtime.seed = 'cpu', 7
    runtime.max_new_tokens, runtime.temperature, runtime.top_p = 512, 0.0, 0.95
    runtime.is_encoder_decoder = False
    runtime.program_prompt_mode = 'humanevalpack_chat'
    assert runtime.generate_program(prompt=prompt, sample_id=sample_id) == code + '\n'
    assert runtime.model.calls == 1
    trace = runtime.last_program_generation_trace
    assert trace['model_response'] == response
    assert trace['model_input_token_ids'] == [10, 11]
    assert trace['model_continuation_token_ids'] == [7, 8]
    assert trace['sample_id'] == sample_id


@pytest.mark.parametrize(('text', 'language', 'expected'), [
    ('bad syntax', 'Java', 'bad syntax\n'),
    ('```java\nclass A {', 'Java', 'class A {\n'),
    ('```python\nx=1\n```\nprose', 'Java', '```python\nx=1\n```\nprose\n'),
    ('```js\nconst f=()=>0;\n```\n```js\nconst f=()=>1;\n```', 'JavaScript', 'const f=()=>0;\n'),
    ('const f=()=>0; console.log(f());', 'JavaScript', 'const f=()=>0; console.log(f());\n'),
])
def test_native_response_unwrapping_preserves_failures_and_first_candidate(text, language, expected):
    assert _extract_native_chat_code(text, language) == expected


def test_native_chat_missing_template_fails_closed():
    with pytest.raises(ValueError, match='chat template'):
        _native_chat_program_prompt(object(), 'prefix', 'Java')


def test_native_chat_rejects_other_languages_before_inference():
    runtime = object.__new__(LocalHFProgramGenerator)
    runtime.program_prompt_mode = 'humanevalpack_chat'
    with pytest.raises(ValueError, match='sample ID'):
        runtime.generate_program(prompt='prefix', sample_id='CPP/7')


@pytest.mark.parametrize('language', ['java', 'js'])
def test_native_chat_profiles_are_exact_public_overlays(language):
    from wfcllm.cli.experiment_preflight import validate_public_full_config_overlay

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / f'configs/wfcllm/experiments/{language}_humanevalpack_full.json').read_text(encoding='utf-8'))
    validate_public_full_config_overlay(config, language, 'humanevalpack', 'full')
    config['generation']['prompt_mode'] = 'completion'
    with pytest.raises(ValueError, match='canonical overlay'):
        validate_public_full_config_overlay(config, language, 'humanevalpack', 'full')
