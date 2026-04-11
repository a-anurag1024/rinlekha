# pipeline — Phase 1 data pipeline scripts for RinLekha
#
# Modules:
#   samplers.py          — FieldSampler classes and PROFILE_SCHEMA
#   rules.py             — Derived field rules, underwriting policy rules, decline triggers
#   profile_generator.py — Borrower profile generation (Ray parallel)
#   memo_synthesizer.py  — Credit memo synthesis via OpenAI API (Ray parallel)  [Phase 1 step 2]
#   quality_reviewer.py  — Automated structural QA on generated memos           [Phase 1 step 3]
#   dataset_builder.py   — Format conversion and HuggingFace Dataset push       [Phase 1 step 4]
