"""
src/sme_clf/predictors.py
Predictor interface and implementations for SME IPO section classification:
- BaselinePredictor: Abstract base class for non-LLM baselines (TF-IDF, heuristic rules)
- LLMPredictor: Gemma-3 classifier supporting:
  (a) Constrained scoring (exact log-probability over 17 candidate completions)
  (b) Diagnostic greedy text generation
"""

import math
import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.sme_clf.prompting import (
    CATEGORIES_17,
    build_prompt_text,
    build_messages,
)

class BaselinePredictor(ABC):
    @abstractmethod
    def predict(self, heading: str, body: str) -> Dict[str, Any]:
        """Returns dict with 'pred', 'confidence', 'score_vector'"""
        pass

class LLMPredictor:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_seq_len: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.categories = CATEGORIES_17
        self.end_of_turn_id = 106 # <end_of_turn>

        # Pre-tokenize all 17 candidate continuations: "{category}<end_of_turn>"
        self.candidate_tokens = {}
        for cat in self.categories:
            # Tokenize category + <end_of_turn>
            toks = self.tokenizer.encode(cat, add_special_tokens=False) + [self.end_of_turn_id]
            self.candidate_tokens[cat] = toks

    def score_candidates_constrained(self, prompt_text: str) -> Tuple[str, float, List[float], Dict[str, float]]:
        """
        Computes exact sequence log-probability for all 17 candidate label continuations:
        P(c_1, ..., c_k | prompt) = sum_{j=1}^k log P(c_j | prompt, c_1, ..., c_{j-1})
        Returns (predicted_label, max_prob, normalized_score_vector_17, raw_log_probs_dict).
        """
        messages = [{"role": "user", "content": prompt_text}]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = self.tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len - 32
        )["input_ids"]
        prompt_len = len(prompt_ids)

        raw_log_probs = {}

        for cat, c_tokens in self.candidate_tokens.items():
            full_seq = prompt_ids + c_tokens
            seq_tensor = torch.tensor([full_seq], device=self.device)
            with torch.no_grad():
                logits = self.model(seq_tensor, use_cache=False).logits
                log_probs = F.log_softmax(logits, dim=-1)

            lp_sum = 0.0
            for idx, tok in enumerate(c_tokens):
                pos = prompt_len + idx - 1
                lp_sum += log_probs[0, pos, tok].item()

            raw_log_probs[cat] = lp_sum

        # Softmax normalization across the 17 categories
        max_lp = max(raw_log_probs.values())
        exp_vals = [math.exp(raw_log_probs[cat] - max_lp) for cat in self.categories]
        sum_exp = sum(exp_vals)
        score_vector = [round(v / sum_exp, 6) for v in exp_vals]

        best_idx = int(np.argmax(score_vector))
        pred_label = self.categories[best_idx]
        confidence = score_vector[best_idx]

        return pred_label, confidence, score_vector, raw_log_probs

    def score_candidates_naive(self, prompt_text: str) -> Tuple[str, Dict[str, float]]:
        """
        Reference implementation: computes full forward pass on prompt + candidate continuation
        without KV caching, to verify mathematical equivalence.
        """
        messages = [{"role": "user", "content": prompt_text}]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len - 32
        )["input_ids"]

        raw_log_probs = {}
        for cat, c_tokens in self.candidate_tokens.items():
            full_seq = prompt_ids + c_tokens
            seq_tensor = torch.tensor([full_seq], device=self.device)
            with torch.no_grad():
                logits = self.model(seq_tensor).logits
                log_probs = F.log_softmax(logits, dim=-1)

            # Sum log probabilities of the candidate tokens
            lp_sum = 0.0
            for idx, tok in enumerate(c_tokens):
                # The prediction for full_seq[len(prompt_ids) + idx] is at position len(prompt_ids) + idx - 1
                pos = len(prompt_ids) + idx - 1
                lp_sum += log_probs[0, pos, tok].item()
            raw_log_probs[cat] = lp_sum

        best_cat = max(raw_log_probs.keys(), key=lambda c: raw_log_probs[c])
        return best_cat, raw_log_probs

    def generate_diagnostic(self, prompt_text: str, max_new_tokens: int = 16) -> Tuple[str, bool]:
        """
        Unconstrained greedy generation to measure invalid generation rate.
        Returns (predicted_text, is_valid_label).
        """
        messages = [{"role": "user", "content": prompt_text}]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len - 32
        )["input_ids"]
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.end_of_turn_id,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        gen_tokens = output_ids[0][len(prompt_ids):].tolist()
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().split("\n")[0]

        is_valid = (gen_text in self.categories)
        return gen_text, is_valid
