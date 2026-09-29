# Steering evaluation: qwen2.5-0.5b_phase4

Model `qwen2.5-0.5b` (revision `7ae557604adf67be50417f59c2c2f167def9a775`), code `7a745e08bf21`, 48 held-out prompts (test split), greedy decoding of 80 tokens, layer distribution `bell`. Intervals are 95% bootstrap intervals over prompts (or ARC items); differences are paired with the baseline.

## Language and fluency

| Condition | Dose ratio (target / achieved peak) | Arabic-script outputs, English prompts | …English neutral prompts | Arabic-script outputs, Arabic prompts | Degenerate outputs | Distinct-2 | ΔNLL under unsteered model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | – | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.96 [0.95, 0.98] | n/a |
| raw_mean_r0.1 | 0.1 / 0.112 [0.111, 0.112] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.98 [0.98, 0.99] | +0.36 [+0.29, +0.44] |
| centered_r0.05 | 0.05 / 0.054 [0.054, 0.055] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.98] | +0.19 [+0.13, +0.26] |
| centered_r0.1 | 0.1 / 0.111 [0.110, 0.112] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.06 [0.00, 0.12] | 0.88 [0.84, 0.92] | +1.11 [+0.93, +1.28] |
| centered_r0.2 | 0.2 / 0.202 [0.201, 0.204] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.92 [0.79, 1.00] | 0.81 [0.71, 0.92] | 0.86 [0.78, 0.95] | +1.71 [+1.39, +2.03] |
| centered_en_control_r0.1 | 0.1 / 0.099 [0.099, 0.099] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.95] | +0.29 [+0.24, +0.34] |
| rag_only (RAG) | – | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.97 [0.96, 0.98] | +0.02 [-0.06, +0.09] |
| rag_centered_r0.1 (RAG) | 0.1 / 0.107 [0.107, 0.108] | 0.08 [0.00, 0.21] | 0.00 [0.00, 0.00] | 0.17 [0.04, 0.33] | 0.21 [0.10, 0.33] | 0.79 [0.72, 0.85] | +0.83 [+0.57, +1.09] |

## Capability, thematic proxy and transport

| Condition | ARC-Easy accuracy | ΔARC vs baseline | Δ thematic proxy | Δρ | Δholonomy |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.61 [0.51, 0.71] | n/a | n/a | n/a | n/a |
| raw_mean_r0.1 | 0.55 [0.45, 0.65] | -0.060 [-0.130, +0.010] | +0.016 [-0.013, +0.041] | -0.0267 [-0.0279, -0.0255] | -0.2217 [-0.2294, -0.2145] |
| centered_r0.05 | 0.59 [0.50, 0.69] | -0.020 [-0.070, +0.030] | +0.082 [+0.042, +0.130] | -0.0103 [-0.0109, -0.0098] | -0.0953 [-0.0996, -0.0914] |
| centered_r0.1 | 0.46 [0.36, 0.56] | -0.150 [-0.250, -0.060] | +0.317 [+0.254, +0.385] | -0.0240 [-0.0256, -0.0226] | -0.2074 [-0.2206, -0.1953] |
| centered_r0.2 | 0.40 [0.31, 0.50] | -0.210 [-0.320, -0.090] | +0.307 [+0.246, +0.366] | -0.0584 [-0.0610, -0.0554] | -0.4629 [-0.4961, -0.4317] |
| centered_en_control_r0.1 | 0.56 [0.46, 0.66] | -0.050 [-0.110, +0.010] | +0.025 [-0.000, +0.052] | +0.0162 [+0.0158, +0.0167] | +0.1618 [+0.1582, +0.1654] |
| rag_only (RAG) | 0.61 [0.51, 0.71] | +0.000 [+0.000, +0.000] | +0.124 [+0.084, +0.164] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| rag_centered_r0.1 (RAG) | 0.46 [0.36, 0.56] | -0.150 [-0.250, -0.060] | +0.330 [+0.272, +0.394] | -0.0240 [-0.0256, -0.0226] | -0.2074 [-0.2206, -0.1953] |

Arabic-script outputs: share of outputs whose letters are mostly Arabic script. The thematic proxy is an embedding contrast, not a rating; see the rating sheet. Capability and transport depend only on the hooks, so RAG conditions repeat the matching non-RAG values.
