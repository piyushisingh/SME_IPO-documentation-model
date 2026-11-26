# Task 0b: Timing Sanity & Hardware Throttle Benchmark

**Companion JSON:** [`reports/phase5/timing_sanity.json`](file:///c:/SMEIPO%20Model/reports/phase5/timing_sanity.json)

### System Power & Throttle Status

- **Active Power Plan:** `Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)`
- **Battery Info:** `UNKNOWN`

### Alternating Benchmark Runs (1 Warmup Excluded, 6 Timed Steps Each)

| Run Name | Cap | Mean Step (s) | Min Step (s) | Max Step (s) | Mean SM Clock (MHz) | Mean Temp (C) | Mean Power (W) | Throttling Active |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1024_run1` | 1024 | **1.86** | 1.086 | 5.338 | 1221.5 | 55.0 | 14.5 | `True` |
| `1536_run1` | 1536 | **1.409** | 1.381 | 1.445 | 1442.2 | 62.5 | 29.3 | `True` |
| `1024_run2` | 1024 | **1.275** | 1.197 | 1.358 | 1294.5 | 63.0 | 20.8 | `True` |
| `1536_run2` | 1536 | **1.434** | 1.367 | 1.504 | 1488.8 | 65.8 | 28.7 | `True` |

### 3-Epoch Training Projection

- **Average 1024 Step Time (Longest Sequences):** `1.567 s`
- **Average 1536 Step Time (Longest Sequences):** `1.421 s`
- **Projected 3-Epoch Training Time (11,358 steps):** **4.945 hours**

