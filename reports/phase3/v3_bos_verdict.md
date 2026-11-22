# V3 Double-BOS Empirical Verdict

**Verdict:** `V3 trained with a double BOS: NO`

**Evidence:**
TRL's SFTTrainer natively tokenizes conversational prompt/completion datasets via apply_chat_template(tokenize=True), which produces [2, 105, 2364, 107, 4335, 1891] (exactly 1 BOS token). The double-BOS bug was confined to the standalone audit script data/audit_v3_chat_template.py which manually rendered text and then called tokenizer() with add_special_tokens=True.

### First 6 Tokens across 3 V3 Training Examples:
- Example 0: `[2, 105, 2364, 107, 4335, 1891]` (BOS count: 1)
- Example 1: `[2, 105, 2364, 107, 4335, 1891]` (BOS count: 1)
- Example 2: `[2, 105, 2364, 107, 4335, 1891]` (BOS count: 1)

**Implication for Evaluation:**
When evaluating checkpoint v3, prompts should be formatted with standard SINGLE BOS: <bos><start_of_turn>user
...<end_of_turn>
<start_of_turn>model
.
