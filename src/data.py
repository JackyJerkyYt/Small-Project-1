"""Data preparation: build datasets for SFT with or without chat template, with masking options."""

import warnings

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from tasks.base import Sample


def format_with_chat_template(
    tokenizer,
    question: str,
    answer: str | None = None,
    extra_chat_template_kwargs: dict | None = None,
) -> str:
    """Format using the model's chat template.
    If answer is None, format as a prompt only (for generation)."""
    messages = [{"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    kwargs = extra_chat_template_kwargs or {}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(answer is None),
        **kwargs,
    )


def format_without_chat_template(
    question: str,
    answer: str | None = None,
    eos_token: str | None = None,
) -> str:
    """Simple concatenation: 'Question: ... Answer: ...'
    
    The prompt always ends with 'Answer: ' (with trailing space) so that
    tokenization is consistent between prompt-only and full text.
    """
    prompt = f"Question: {question}\nAnswer: "
    if answer is not None:
        text = prompt + answer
        if eos_token:
            text += eos_token
        return text
    return prompt


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        samples: list of (question, answer) Sample objects
        tokenizer: HuggingFace tokenizer
        use_chat_template: whether to apply the model's chat template
        mask_question: if True, loss is computed only on the answer tokens
        max_seq_length: maximum sequence length
        extra_chat_template_kwargs: extra kwargs for tokenizer.apply_chat_template
    """

    def __init__(
        self,
        samples: list[Sample],
        tokenizer,
        use_chat_template: bool,
        mask_question: bool,
        max_seq_length: int = 1024,
        extra_chat_template_kwargs: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.mask_question = mask_question
        self.max_seq_length = max_seq_length
        self.extra_chat_template_kwargs = extra_chat_template_kwargs or {}
        self.data = self._prepare(samples)

    def _prepare(self, samples: list[Sample]) -> list[dict]:
        prepared = []
        mismatch_count = 0
        total_fallback_count = 0

        for s in samples:
            if self.use_chat_template:
                full_text = format_with_chat_template(
                    self.tokenizer, s.question, s.answer,
                    self.extra_chat_template_kwargs,
                )
                prompt_text = format_with_chat_template(
                    self.tokenizer, s.question, None,
                    self.extra_chat_template_kwargs,
                )
            else:
                full_text = format_without_chat_template(
                    s.question, s.answer, eos_token=self.tokenizer.eos_token
                )
                prompt_text = format_without_chat_template(s.question, None)

            full_ids = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=False,
            )
            input_ids = full_ids["input_ids"]
            attention_mask = full_ids["attention_mask"]

            if self.mask_question:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    add_special_tokens=False,
                )["input_ids"]
                prompt_len = len(prompt_ids)

                if prompt_ids != input_ids[:prompt_len]:
                    mismatch_count += 1
                    for i in range(min(len(prompt_ids), len(input_ids)), 0, -1):
                        if prompt_ids[:i] == input_ids[:i]:
                            prompt_len = i
                            break
                    else:
                        prompt_len = 0
                        total_fallback_count += 1

                labels = [-100] * prompt_len + input_ids[prompt_len:]
            else:
                labels = list(input_ids)

            prepared.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        if self.mask_question and mismatch_count > 0:
            print(f"Tokenization boundary mismatches: {mismatch_count}/{len(samples)} "
                  f"(adjusted prompt_len for these samples)")
        if total_fallback_count > 0:
            warnings.warn(
                f"{total_fallback_count}/{len(samples)} samples had completely misaligned "
                f"tokenization — masking was disabled for these (prompt_len=0, full loss applied)."
            )

        return prepared

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class CausalLMDataCollator:
    """Data collator for causal language modeling that properly pads labels with -100.
    
    This collator:
    - Pads input_ids with pad_token_id (right-padding for training)
    - Pads labels with -100 (so padded positions are ignored in loss)
    - Pads attention_mask with 0
    """
    tokenizer: object
    padding: bool = True
    max_length: int | None = None
    
    def __call__(self, features: list[dict]) -> dict:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_labels = [f["labels"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        
        max_len = max(len(ids) for ids in batch_input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for input_ids, labels, attention_mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
            padding_length = max_len - len(input_ids)
            
            if padding_length > 0:
                padded_input_ids.append(input_ids + [pad_token_id] * padding_length)
                padded_labels.append(labels + [-100] * padding_length)
                padded_attention_mask.append(attention_mask + [0] * padding_length)
            else:
                padded_input_ids.append(input_ids[:max_len])
                padded_labels.append(labels[:max_len])
                padded_attention_mask.append(attention_mask[:max_len])
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

