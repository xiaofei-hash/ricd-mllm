# LLaVA-1.5-7B / AMBER / RiCD L-N4

This directory contains the reviewer-facing experiment configuration and
source manifest for the 1,004-image AMBER generative split. The
configuration records the generative LLaVA setting: `lambda_l = 1.0`,
`lambda_s = 0.5`, `tau_l = 0.3`,
`tau_v = 0.3`, `beta = 0.3` (`theta_apc` in the compatibility API), and
language-proxy `k = 10` (`top_k_proxy` in code). The separate
`generation.top_k` field is a sampling control, not the RiCD method parameter
`k`.

