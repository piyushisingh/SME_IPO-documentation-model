# Actual Batch Mask & Tokenization Verification

**Date:** 2026-11-21

## 1. Questions Addressed
1. **Is the unmasked span exactly label tokens + `<end_of_turn>` (id 106)?**
   Yes. The unmasked span contains the label tokens followed by token `106` (`<end_of_turn>`) and token `107` (`\n`).

2. **Does TRL append an extra `<eos>` (id 1), `<bos>`, or newline token anywhere?**
   - Token `1` (`<eos>`) is **NOT** appended (0 occurrences in unmasked span).
   - No extra `<bos>` token is appended.
   - Token `107` (`\n`) is present after `106` as part of Gemma-3's turn separator standard.

3. **How many BOS tokens does the sequence contain?**
   **Exactly 1 BOS token** (token ID 2) at index 0.

## 2. Batch Examples Sampled

### Example 0
- First 6 input tokens: `[2, 105, 2364, 107, 4335, 1891]`
- BOS count: `1`
- Unmasked token IDs: `[2763, 56785, 236779, 624, 236779, 90906, 106, 107]`
- Unmasked decoded: `'disclosures_and_definitions<end_of_turn>\n'`

### Example 1
- First 6 input tokens: `[2, 105, 2364, 107, 4335, 1891]`
- BOS count: `1`
- Unmasked token IDs: `[2763, 56785, 236779, 624, 236779, 90906, 106, 107]`
- Unmasked decoded: `'disclosures_and_definitions<end_of_turn>\n'`

### Example 2
- First 6 input tokens: `[2, 105, 2364, 107, 4335, 1891]`
- BOS count: `1`
- Unmasked token IDs: `[2763, 56785, 236779, 624, 236779, 90906, 106, 107]`
- Unmasked decoded: `'disclosures_and_definitions<end_of_turn>\n'`

