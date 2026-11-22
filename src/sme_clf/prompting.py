"""
src/sme_clf/prompting.py
Single source of truth for training, evaluation, and inference prompting.
Enforces the Gemma 3 chat template and strict one-BOS token policy.
"""

import hashlib
import re
from typing import List, Dict, Optional, Tuple, Any

PROMPT_VERSION = "v4s"

CATEGORIES_17: List[str] = [
    "business_operations",
    "capital_structure",
    "company_information",
    "declarations",
    "disclosures_and_definitions",
    "financial_information",
    "financial_statements",
    "industry_information",
    "ipo_details",
    "legal_and_regulatory",
    "litigation",
    "objects_of_issue",
    "other",
    "promoters_and_management",
    "qualitative_factors",
    "related_party_transactions",
    "risk_factors",
]

# Exact short V3 instruction (15 tokens under Gemma tokenizer)
SHORT_INSTRUCTION: str = "Classify the following SME IPO prospectus section into the most appropriate document category."

# Standard Gemma 3 prompt template representation
PROMPT_TEMPLATE: str = (
    "<bos><start_of_turn>user\n"
    "{instruction}\n\n"
    "{content}<end_of_turn>\n"
    "<start_of_turn>model\n"
    "{target}<end_of_turn>\n"
)

# Deterministic SHA-256 hash of the instruction and template
PROMPT_HASH: str = hashlib.sha256(
    (SHORT_INSTRUCTION + "\n" + PROMPT_TEMPLATE).encode("utf-8")
).hexdigest()

def normalize_for_match(text: str) -> str:
    """Normalizes text for robust heading duplication matching."""
    if not text:
        return ""
    text = text.lower()
    # Strip Section prefixes like 'SECTION IV - ' or 'CHAPTER 2:'
    text = re.sub(r'^(?:section|chapter)\s+[ivxlcdm0-9]+\s*[:\-–—.]\s*', '', text)
    # Remove all non-alphanumeric characters and collapse spaces
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.strip()

def strip_leading_heading(body: str, heading: str) -> str:
    """
    Removes the body's first line(s) if their normalized text equals the normalized heading.
    Phase 1 measured that about 94.5% of examples repeat the heading verbatim or near-verbatim
    as the first line(s) of the body.
    """
    if not body or not heading:
        return body if body else ""

    norm_h = normalize_for_match(heading)
    if not norm_h:
        return body.strip()

    lines = body.split("\n")
    # Check if first line matches
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        norm_line = normalize_for_match(line)
        # Check direct equality or line is a prefix of heading / heading is prefix of line
        if norm_line == norm_h or (len(norm_h) > 5 and norm_line.startswith(norm_h)):
            idx += 1
            # Continue checking if multi-line heading split occurred
            continue
        elif len(norm_h) > 10 and norm_h.startswith(norm_line):
            # First line is first half of heading, check if next line completes it
            if idx + 1 < len(lines):
                combined = normalize_for_match(line + " " + lines[idx + 1].strip())
                if combined == norm_h or combined.startswith(norm_h):
                    idx += 2
                    continue
            break
        else:
            break

    remaining = "\n".join(lines[idx:]).strip()
    return remaining if remaining else body.strip()

def build_prompt_content(heading: str, body: str, variant: str = "full") -> str:
    """
    Builds the user turn content:
    - 'full': Heading line + body without duplicated heading.
    - 'no_heading': Heading line removed AND duplicate leading heading stripped from body.
    """
    body_clean = strip_leading_heading(body, heading)
    heading_clean = heading.strip()

    if variant == "no_heading":
        return f"Text:\n{body_clean}"
    elif variant == "full":
        if heading_clean:
            return f"Heading: {heading_clean}\n\nText:\n{body_clean}"
        else:
            return f"Text:\n{body_clean}"
    else:
        raise ValueError(f"Unknown variant '{variant}'. Must be 'full' or 'no_heading'.")

def build_messages(heading: str, body: str, target: Optional[str] = None, variant: str = "full") -> List[Dict[str, str]]:
    """Builds conversational message dictionary for chat templates."""
    content = f"{SHORT_INSTRUCTION}\n\n{build_prompt_content(heading, body, variant=variant)}"
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": target.strip()})
    return messages

def build_prompt_text(heading: str, body: str, variant: str = "full") -> str:
    """Builds user prompt text without assistant generation prompt."""
    return f"{SHORT_INSTRUCTION}\n\n{build_prompt_content(heading, body, variant=variant)}"

def build_full_text(heading: str, body: str, target: str, tokenizer: Any, variant: str = "full") -> str:
    """
    Renders the complete training sequence using the tokenizer's chat template.
    Guarantees exactly one <bos> in text output.
    """
    messages = build_messages(heading, body, target=target, variant=variant)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return rendered

def build_full_ids(heading: str, body: str, target: str, tokenizer: Any, variant: str = "full") -> List[int]:
    """
    Builds full token IDs with EXACTLY ONE <bos> at index 0.
    Uses add_special_tokens=False on rendered chat template to prevent duplicate BOS tokens.
    """
    rendered = build_full_text(heading, body, target, tokenizer, variant=variant)
    tokens = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    assert tokens[0] == tokenizer.bos_token_id, f"First token must be BOS ({tokenizer.bos_token_id}), got {tokens[0]}"
    assert tokens.count(tokenizer.bos_token_id) == 1, f"Expected exactly 1 BOS token, found {tokens.count(tokenizer.bos_token_id)}"
    return tokens

def build_prompt_ids(heading: str, body: str, tokenizer: Any, variant: str = "full") -> List[int]:
    """
    Builds prompt token IDs up to the assistant generation prompt with EXACTLY ONE <bos> at index 0.
    """
    messages = build_messages(heading, body, target=None, variant=variant)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    assert tokens[0] == tokenizer.bos_token_id, f"First token must be BOS ({tokenizer.bos_token_id}), got {tokens[0]}"
    assert tokens.count(tokenizer.bos_token_id) == 1, f"Expected exactly 1 BOS token, found {tokens.count(tokenizer.bos_token_id)}"
    return tokens

