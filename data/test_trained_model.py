import torch
from unsloth import FastLanguageModel

MODEL_PATH = "checkpoints/phase1/final"

print("=" * 60)
print("Loading trained SME IPO model...")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": (
            "Is there any pending litigation against the company?\n\n"
            "Company: Acme Textiles Ltd.\n"
            "Forum: Delhi High Court.\n"
            "Amount: Rs 12,00,000.\n"
            "Status: Pending."
        ),
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=150,
        temperature=0.2,
    )

response = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True,
)

print("\n--- MODEL OUTPUT ---")
print(response)
print("\n--- END OUTPUT ---")