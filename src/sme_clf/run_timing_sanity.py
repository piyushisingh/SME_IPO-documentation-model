"""
src/sme_clf/run_timing_sanity.py
Task 0b: Timing Sanity & Throttle Investigation:
- 16-longest-example benchmark at 1024 and 1536
- 1 warm-up step excluded, 6 timed steps each
- Alternating order: [1024_run1, 1536_run1, 1024_run2, 1536_run2]
- Logs nvidia-smi metrics every second via GPUMonitor
- Checks Windows power plan and battery/AC status
- Saves results to reports/phase5/timing_sanity.json and .md
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.sme_clf.gpu_monitor import GPUMonitor

DEFAULT_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--unsloth--gemma-3-1b-it-unsloth-bnb-4bit" / "snapshots" /
    "bfae90ad90f41a9be81bd68255d733bb80c97159"
)

def get_system_power_status() -> Dict[str, Any]:
    power_plan = "UNKNOWN"
    battery_status = "UNKNOWN"
    try:
        res = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        if res.returncode == 0:
            power_plan = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["powershell", "-Command", "Get-CimInstance -ClassName Win32_Battery | Select-Object -Property BatteryStatus, EstimatedChargeRemaining | ConvertTo-Json"],
            capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            battery_status = json.loads(res.stdout)
    except Exception:
        pass

    return {
        "active_power_plan": power_plan,
        "battery_info": battery_status
    }

def main():
    print("=" * 70)
    print("TASK 0b: RUNNING TIMING SANITY & HARDWARE BENCHMARK")
    print("=" * 70)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # Load 16 longest examples from manifests
    def get_longest_16(cap_dir: Path) -> List[Dict[str, Any]]:
        manifest_path = cap_dir / "manifest_train.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        sorted_records = sorted(records, key=lambda r: r["total_tokens"], reverse=True)[:16]
        eids = set(r["example_id"] for r in sorted_records)
        train_path = cap_dir / "train.jsonl"
        chosen = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["example_id"] in eids:
                    chosen.append(row)
        return chosen

    rows_1024 = get_longest_16(WORKSPACE_ROOT / "runs" / "data_v4_1024")
    rows_1536 = get_longest_16(WORKSPACE_ROOT / "runs" / "data_v4_1536")

    print(f"Loaded {len(rows_1024)} longest 1024 examples and {len(rows_1536)} longest 1536 examples.")

    power_info = get_system_power_status()

    runs_to_execute = [
        {"name": "1024_run1", "cap": 1024, "rows": rows_1024},
        {"name": "1536_run1", "cap": 1536, "rows": rows_1536},
        {"name": "1024_run2", "cap": 1024, "rows": rows_1024},
        {"name": "1536_run2", "cap": 1536, "rows": rows_1536},
    ]

    timing_dir = WORKSPACE_ROOT / "runs" / "_phase5" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = []

    for run_cfg in runs_to_execute:
        run_name = run_cfg["name"]
        cap = run_cfg["cap"]
        rows = run_cfg["rows"]

        print(f"\n--- Starting Benchmark Run: {run_name} (cap={cap}) ---")
        csv_path = timing_dir / f"gpu_metrics_{run_name}.csv"
        monitor = GPUMonitor(output_csv=csv_path, interval=1.0)
        monitor.start()

        # Write dataset to tmp file
        tmp_data_file = timing_dir / f"data_{run_name}.jsonl"
        with open(tmp_data_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Load fresh model for this cap
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(DEFAULT_SNAPSHOT),
            max_seq_length=cap,
            load_in_4bit=True,
            local_files_only=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        dataset = load_dataset("json", data_files=str(tmp_data_file), split="train")

        out_run_dir = timing_dir / f"run_out_{run_name}"
        out_run_dir.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=str(out_run_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=7, # 1 warmup + 6 timed
            learning_rate=1e-4,
            save_strategy="no",
            logging_steps=1,
            optim="adamw_8bit",
            seed=42,
            fp16=False,
            bf16=True,
            report_to="none",
            max_length=cap,
            completion_only_loss=True,
            dataloader_num_workers=0,
            dataset_num_proc=1,
        )

        class StepTimingCallback:
            def __init__(self):
                self.step_times = []
                self.last_time = None

            def on_step_begin(self, args, state, control, **kwargs):
                self.last_time = time.perf_counter()

            def on_step_end(self, args, state, control, **kwargs):
                if self.last_time is not None:
                    step_duration = time.perf_counter() - self.last_time
                    self.step_times.append(step_duration)

        from transformers import TrainerCallback
        class HFCallbackWrapper(TrainerCallback):
            def __init__(self, cb):
                self.cb = cb
            def on_step_begin(self, args, state, control, **kwargs):
                self.cb.on_step_begin(args, state, control, **kwargs)
            def on_step_end(self, args, state, control, **kwargs):
                self.cb.on_step_end(args, state, control, **kwargs)

        timer_cb = StepTimingCallback()
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            callbacks=[HFCallbackWrapper(timer_cb)],
        )

        trainer.train()
        monitor.stop()

        all_steps = timer_cb.step_times
        warmup_step = all_steps[0] if all_steps else None
        timed_steps = all_steps[1:7] if len(all_steps) >= 7 else all_steps[1:]

        mean_step_time = round(float(np.mean(timed_steps)), 3) if timed_steps else 0.0
        min_step_time = round(float(min(timed_steps)), 3) if timed_steps else 0.0
        max_step_time = round(float(max(timed_steps)), 3) if timed_steps else 0.0

        gpu_sum = monitor.get_summary()

        run_summary = {
            "run_name": run_name,
            "max_seq_length": cap,
            "warmup_step_seconds": round(warmup_step, 3) if warmup_step else None,
            "timed_step_count": len(timed_steps),
            "timed_step_times": [round(t, 3) for t in timed_steps],
            "mean_step_time_seconds": mean_step_time,
            "min_step_time_seconds": min_step_time,
            "max_step_time_seconds": max_step_time,
            "gpu_hardware": gpu_sum
        }
        benchmark_results.append(run_summary)
        print(f"Run {run_name} Mean Step Time: {mean_step_time}s (min: {min_step_time}s, max: {max_step_time}s)")
        print(f"GPU Temp: {gpu_sum.get('temp_c_mean')} C | Power: {gpu_sum.get('power_w_mean')} W | Throttling: {gpu_sum.get('throttling_active')}")

        # Cleanup memory
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)

    # Recompute projected 3-epoch training time from 1024 runs
    # 3 epochs on 3,786 examples with grad_accum=8 => 1,419 optimization steps
    # Each optimization step is 8 micro-batches of batch_size=1
    # Average micro-step time across 1024 runs:
    times_1024 = [r["mean_step_time_seconds"] for r in benchmark_results if r["max_seq_length"] == 1024]
    avg_step_1024 = float(np.mean(times_1024))
    # Full training has 3,786 * 3 = 11,358 total forward passes (micro-steps)
    projected_total_seconds = avg_step_1024 * 11358
    projected_total_hours = round(projected_total_seconds / 3600.0, 3)

    final_report = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "power_and_battery_status": power_info,
        "runs": benchmark_results,
        "summary_comparison": {
            "cap_1024_mean_step_seconds": round(float(np.mean(times_1024)), 3),
            "cap_1536_mean_step_seconds": round(float(np.mean([r["mean_step_time_seconds"] for r in benchmark_results if r["max_seq_length"] == 1536])), 3),
            "projected_3_epoch_hours_at_1024": projected_total_hours
        }
    }

    rep_dir = WORKSPACE_ROOT / "reports" / "phase5"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_json = rep_dir / "timing_sanity.json"
    with open(rep_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    rep_md = rep_dir / "timing_sanity.md"
    with open(rep_md, "w", encoding="utf-8") as f:
        f.write("# Task 0b: Timing Sanity & Hardware Throttle Benchmark\n\n")
        f.write(f"**Companion JSON:** [`reports/phase5/timing_sanity.json`](file:///c:/SMEIPO%20Model/reports/phase5/timing_sanity.json)\n\n")
        f.write("### System Power & Throttle Status\n\n")
        f.write(f"- **Active Power Plan:** `{power_info['active_power_plan']}`\n")
        f.write(f"- **Battery Info:** `{power_info['battery_info']}`\n\n")
        f.write("### Alternating Benchmark Runs (1 Warmup Excluded, 6 Timed Steps Each)\n\n")
        f.write("| Run Name | Cap | Mean Step (s) | Min Step (s) | Max Step (s) | Mean SM Clock (MHz) | Mean Temp (C) | Mean Power (W) | Throttling Active |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in benchmark_results:
            gpu = r["gpu_hardware"]
            f.write(f"| `{r['run_name']}` | {r['max_seq_length']} | **{r['mean_step_time_seconds']}** | {r['min_step_time_seconds']} | {r['max_step_time_seconds']} | {gpu.get('sm_clock_mean')} | {gpu.get('temp_c_mean')} | {gpu.get('power_w_mean')} | `{gpu.get('throttling_active')}` |\n")
        f.write("\n### 3-Epoch Training Projection\n\n")
        f.write(f"- **Average 1024 Step Time (Longest Sequences):** `{final_report['summary_comparison']['cap_1024_mean_step_seconds']} s`\n")
        f.write(f"- **Average 1536 Step Time (Longest Sequences):** `{final_report['summary_comparison']['cap_1536_mean_step_seconds']} s`\n")
        f.write(f"- **Projected 3-Epoch Training Time (11,358 steps):** **{projected_total_hours} hours**\n\n")

    print(f"Timing sanity saved to: {rep_json} and {rep_md}")

if __name__ == "__main__":
    main()

