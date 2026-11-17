# Phase 1 — Landmark Extraction QC Report

**MediaPipe version:** 0.10.14  |  **stride:** 2  |  **max_seq_len:** 256


## Split: train

| Metric | Value |
|--------|-------|
| Total clips | 100 |
| Succeeded | 100 |
| Failed | 0 (0.0%) |
| Flagged (avg hand det < 30%) | 1 (1.0%) |
| Left hand detection rate (mean) | 0.857 |
| Right hand detection rate (mean) | 0.860 |
| Seq length mean | 104.6 frames |
| Seq length median | 98.0 frames |
| Seq length p95 | 234.1 frames |
| Seq length max | 256 frames |
| Truncated (> max_seq_len) | 5 |

**Gate 1 status:** ✅ PASS (fail rate < 5%)


## Split: val

| Metric | Value |
|--------|-------|
| Total clips | 50 |
| Succeeded | 50 |
| Failed | 0 (0.0%) |
| Flagged (avg hand det < 30%) | 1 (2.0%) |
| Left hand detection rate (mean) | 0.828 |
| Right hand detection rate (mean) | 0.888 |
| Seq length mean | 58.0 frames |
| Seq length median | 45.0 frames |
| Seq length p95 | 122.1 frames |
| Seq length max | 177 frames |
| Truncated (> max_seq_len) | 0 |

**Gate 1 status:** ✅ PASS (fail rate < 5%)


## Split: test

| Metric | Value |
|--------|-------|
| Total clips | 50 |
| Succeeded | 50 |
| Failed | 0 (0.0%) |
| Flagged (avg hand det < 30%) | 0 (0.0%) |
| Left hand detection rate (mean) | 0.849 |
| Right hand detection rate (mean) | 0.935 |
| Seq length mean | 91.0 frames |
| Seq length median | 84.0 frames |
| Seq length p95 | 226.9 frames |
| Seq length max | 256 frames |
| Truncated (> max_seq_len) | 1 |

**Gate 1 status:** ✅ PASS (fail rate < 5%)
