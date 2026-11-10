# UC.2 — ROM Generator, ROM Module Template & Weight Audit v2
Status: done
Phase/Gate: UC / HC     Tags: [rtl, verification]     Estimate: 2h
Depends on: UB.4         Spec refs: docs/part2/02 §6, §7; docs/part2/06 §A7

## Goal
Implement:
1. Parameterized synchronous ROM template `rtl/lib/rom_sync.v` with `$readmemh`, single-cycle read, and zero write ports.
2. Generator `backend/rom_generator.py` for emitting model-specific ROM wrapper modules.
3. Verification tool `verification/weight_audit_v2.py` enforcing immutable weights, interface conformance (zero write ports), and cryptographic hash matching of ROM hex contents against `address_map.json` and quantized model tensors.
4. Comprehensive tests in `tests/test_weight_audit_v2.py` including negative controls (injected write port, altered hex, corrupted hash).

## Acceptance (executable)
- [x] `rtl/lib/rom_sync.v` implemented, lints clean in `iverilog -g2005 -Wall` and `verilator --lint-only -Wall`.
- [x] `backend/rom_generator.py` generates clean ROM wrapper modules.
- [x] `verification/weight_audit_v2.py` verifies zero write ports and byte-level matching of hex images against tensors.
- [x] Negative controls: audit fails on an injected write port, fails on altered hex image, fails on hash mismatch.
- [x] `tests/test_weight_audit_v2.py` passes and is added to `make check2`.

## Evidence (after)
- `rtl/lib/rom_sync.v` created, verified with zero write ports.
- `backend/rom_generator.py` created to emit ROM verilog files with exact width and depth.
- `verification/weight_audit_v2.py` implemented `audit_rom_module_source` and `audit_rom_hex_images`.
- `tests/test_weight_audit_v2.py` covers valid modules, generated modules, and negative controls for write ports, runtime writes, and tampered hex contents.
- `make check2` passes 109/109 tests.
