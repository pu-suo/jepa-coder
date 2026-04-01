# JEPA-Coder

## Project Overview
This is a novel research architecture for code generation. Read docs/jepa_coder_architecture_v2.md 
for the full architecture specification.

## Critical Rules
- DO NOT substitute any custom component with a standard library equivalent
- DO NOT add fallback paths or simplifications without asking
- Every model component has an exact specification in the docs/ folder
- When implementing a component, read the relevant contract document FIRST
- All tensor shapes must match the specifications exactly
- All verification tests must pass before proceeding

## Document Hierarchy
1. docs/jepa_coder_architecture_v2.md — Overall architecture (read first)
2. docs/contract_1_sst_loop.md — SST training loop specification
3. docs/contract_2_vq_module.md — VQ module specification  
4. docs/contract_3_talker_interface.md — Talker interface specification

## Implementation Order
Build and test components in this exact order:
1. models/normalization.py
2. models/vq.py
3. models/ema_encoder.py
4. models/reasoner.py (depends on 1)
5. Verification tests for 1-4
6. training/sst.py (depends on all of 1-4)
7. data/ pipeline
8. training/pretrain.py
9. models/talker.py
10. training/train_talker.py
11. evaluation/inference.py