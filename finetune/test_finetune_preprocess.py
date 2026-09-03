import torch

from finetune_deepseekcoder import IGNORE_INDEX, preprocess


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = torch.tensor([input_ids], dtype=torch.long)


class _CharacterTokenizer:
    """Small deterministic tokenizer used to exercise truncation boundaries."""

    model_max_length = 6
    pad_token_id = 0
    truncation_side = "right"

    def __call__(self, text, return_tensors=None, padding=None, max_length=None, truncation=False):
        input_ids = [ord(char) - ord("a") + 1 for char in text]
        if truncation and len(input_ids) > max_length:
            if self.truncation_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]
        return _Encoding(input_ids)


def test_short_examples_keep_the_original_source_mask():
    tokenizer = _CharacterTokenizer()

    result = preprocess(["ab"], ["xy"], tokenizer)

    assert result["input_ids"][0].tolist() == [1, 2, 24, 25]
    assert result["labels"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 24, 25]
    assert tokenizer.truncation_side == "right"


def test_long_source_keeps_response_tokens_and_masks_only_retained_source():
    tokenizer = _CharacterTokenizer()

    result = preprocess(["abcdefgh"], ["xy"], tokenizer)

    # The left-truncated sequence contains the final four source tokens and
    # both response tokens; the response must remain trainable.
    assert result["input_ids"][0].tolist() == [5, 6, 7, 8, 24, 25]
    assert result["labels"][0].tolist() == [IGNORE_INDEX] * 4 + [24, 25]
    assert any(label != IGNORE_INDEX for label in result["labels"][0].tolist())

    # A language-model loss over this sample is well-defined; the previous
    # all-IGNORE mask produced a NaN mean loss for the same boundary.
    logits = torch.zeros(1, 6, 26)
    shifted_labels = result["labels"][0][1:].unsqueeze(0)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert torch.isfinite(loss)


def test_response_longer_than_context_is_still_supervised():
    tokenizer = _CharacterTokenizer()

    result = preprocess(["xy"], ["abcdefghij"], tokenizer)

    # If the response itself fills the context, no source token remains to be
    # masked, but all retained response labels must stay active.
    assert result["input_ids"][0].tolist() == [5, 6, 7, 8, 9, 10]
    assert result["labels"][0].tolist() == [5, 6, 7, 8, 9, 10]
