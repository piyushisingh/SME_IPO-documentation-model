# Task 0a: Mask-Mechanism Verification via Real SFTTrainer Path

**Companion JSON:** [`reports/phase5/mask_check_real_path.json`](file:///c:/SMEIPO%20Model/reports/phase5/mask_check_real_path.json)

### SFTTrainer Configuration & Collator

- **Dataset Format:** JSON with 'prompt' and 'completion' fields (conversational prompt-completion)
- **Real Collator Class:** `trl.trainer.sft_trainer.DataCollatorForLanguageModeling`
- **Masking Field:** `SFTConfig(completion_only_loss=True)`

### Key Forensic Answers

1. **Unmasked Span:** Unmasked tokens consist of the exact label tokens followed by `[106, 107]` (`<end_of_turn>\n`). Trailing newline (107) **is trained** as part of the turn terminator.
2. **EOS (Token ID 1) Appendage:** Token 1 is **NOT** appended anywhere (`eos_present = False`).
3. **BOS Count:** Exactly **1 BOS token** (token ID 2) at sequence index 0 (`all_single_bos = True`).
4. **Consistency with Phase 3B:** Identical to Phase 3B findings; no unexpected trained tokens.

### Inspected Examples Summary

| Example | Expected Label | Unmasked Token IDs | Decoded Span | Single BOS | EOS Present |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `disclosures_and_definitions` | `[12921, 236779, 624, 236779, 155215, 106, 107]` | `'legal_and_regulatory<end_of_turn>\n'` | `True` | `False` |
| 1 | `disclosures_and_definitions` | `[26447, 236779, 13308, 106, 107]` | `'ipo_details<end_of_turn>\n'` | `True` | `False` |
| 2 | `disclosures_and_definitions` | `[23019, 236779, 68382, 106, 107]` | `'business_operations<end_of_turn>\n'` | `True` | `False` |
