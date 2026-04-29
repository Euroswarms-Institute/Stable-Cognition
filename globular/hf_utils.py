"""Hugging Face loading helpers with conservative fallbacks."""

from __future__ import annotations

from typing import Any, Dict, Optional


def safe_load_tokenizer(model_name_or_path: str, **kwargs: Any):
    """Load a tokenizer, falling back around common fast-tokenizer issues.

    Some transformers/tokenizers combinations raise errors around Mistral regex
    patching, such as duplicate ``fix_mistral_regex`` keyword arguments. The
    slow tokenizer path avoids that code path for affected environments.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    except TypeError as exc:
        message = str(exc)
        if "fix_mistral_regex" not in message and "_patch_mistral_regex" not in message:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("fix_mistral_regex", None)
        fallback_kwargs["use_fast"] = False
        return AutoTokenizer.from_pretrained(model_name_or_path, **fallback_kwargs)


def safe_load_causal_lm(
    model_name_or_path: str,
    *,
    device_map: Optional[str] = None,
    torch_dtype: Any = None,
    low_cpu_mem_usage: bool = True,
    **kwargs: Any,
):
    """Load a causal LM with sensible defaults and fallback behavior."""
    from transformers import AutoModelForCausalLM

    load_kwargs: Dict[str, Any] = dict(kwargs)
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    load_kwargs.setdefault("low_cpu_mem_usage", low_cpu_mem_usage)

    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError:
        # Older transformers versions may not support some newer kwargs.
        load_kwargs.pop("low_cpu_mem_usage", None)
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except ImportError as exc:
        # device_map requires accelerate in many transformers versions. Fall
        # back to a plain load rather than failing entirely.
        if "accelerate" not in str(exc).lower():
            raise
        load_kwargs.pop("device_map", None)
        load_kwargs.pop("low_cpu_mem_usage", None)
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)


def ensure_tokenizer_padding(tokenizer):
    """Ensure generation has a usable pad token."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
