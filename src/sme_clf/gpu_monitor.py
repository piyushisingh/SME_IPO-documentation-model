"""
src/sme_clf/gpu_monitor.py
Monitors GPU hardware metrics via nvidia-smi every second:
- SM clock, memory clock
- GPU temperature
- Power draw
- Throttle reasons
- GPU & Memory utilization
- Memory used
Saves output to CSV and provides a context manager / thread.
"""

import time
import subprocess
import threading
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any

class GPUMonitor(threading.Thread):
    def __init__(self, output_csv: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.output_csv = Path(output_csv)
        self.interval = interval
        self.running = False
        self.records: List[Dict[str, Any]] = []

    def run(self):
        self.running = True
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        query = (
            "timestamp,clocks.current.sm,clocks.current.memory,"
            "temperature.gpu,power.draw,clocks_throttle_reasons.active,"
            "utilization.gpu,utilization.memory,memory.used,memory.total"
        )
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits"
        ]

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "sm_clock_mhz", "mem_clock_mhz",
                "temp_c", "power_w", "throttle_reasons",
                "gpu_util_pct", "mem_util_pct", "mem_used_mib", "mem_total_mib"
            ])
            
            while self.running:
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    line = res.stdout.strip()
                    if line:
                        parts = [p.strip() for p in line.split(",")]
                        writer.writerow(parts)
                        f.flush()
                        self.records.append({
                            "timestamp": parts[0],
                            "sm_clock_mhz": float(parts[1]) if parts[1].replace(".","").isdigit() else parts[1],
                            "mem_clock_mhz": float(parts[2]) if parts[2].replace(".","").isdigit() else parts[2],
                            "temp_c": float(parts[3]) if parts[3].replace(".","").isdigit() else parts[3],
                            "power_w": float(parts[4]) if parts[4].replace(".","").isdigit() else parts[4],
                            "throttle_reasons": parts[5],
                            "gpu_util_pct": float(parts[6]) if parts[6].replace(".","").isdigit() else parts[6],
                            "mem_util_pct": float(parts[7]) if parts[7].replace(".","").isdigit() else parts[7],
                            "mem_used_mib": float(parts[8]) if parts[8].replace(".","").isdigit() else parts[8],
                            "mem_total_mib": float(parts[9]) if parts[9].replace(".","").isdigit() else parts[9],
                        })
                except Exception:
                    pass
                time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.join(timeout=3.0)

    def get_summary(self) -> Dict[str, Any]:
        if not self.records:
            return {}
        temps = [r["temp_c"] for r in self.records if isinstance(r["temp_c"], (int, float))]
        powers = [r["power_w"] for r in self.records if isinstance(r["power_w"], (int, float))]
        sm_clocks = [r["sm_clock_mhz"] for r in self.records if isinstance(r["sm_clock_mhz"], (int, float))]
        mem_clocks = [r["mem_clock_mhz"] for r in self.records if isinstance(r["mem_clock_mhz"], (int, float))]
        throttles = list(set(r["throttle_reasons"] for r in self.records))

        return {
            "record_count": len(self.records),
            "temp_c_min": min(temps) if temps else None,
            "temp_c_max": max(temps) if temps else None,
            "temp_c_mean": round(sum(temps)/len(temps), 1) if temps else None,
            "power_w_min": min(powers) if powers else None,
            "power_w_max": max(powers) if powers else None,
            "power_w_mean": round(sum(powers)/len(powers), 1) if powers else None,
            "sm_clock_min": min(sm_clocks) if sm_clocks else None,
            "sm_clock_max": max(sm_clocks) if sm_clocks else None,
            "sm_clock_mean": round(sum(sm_clocks)/len(sm_clocks), 1) if sm_clocks else None,
            "mem_clock_mean": round(sum(mem_clocks)/len(mem_clocks), 1) if mem_clocks else None,
            "throttle_reasons_observed": throttles,
            "throttling_active": any(t not in ("0x0000000000000000", "None", "") for t in throttles)
        }

