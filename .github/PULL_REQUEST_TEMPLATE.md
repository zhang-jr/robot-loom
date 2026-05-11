## Change Type

<!-- Check all that apply -->

- [ ] `feat` — new feature / new Tool / new Skill
- [ ] `fix` — bug fix
- [ ] `safety` — SafetyEnvelope / audit changes
- [ ] `tool` — Tool / ToolAdapter changes
- [ ] `skill` — Skill / SkillManifest changes
- [ ] `embodiment` — EmbodimentAdapter / robot form factor
- [ ] `refactor` — refactor (no behavior change)
- [ ] `test` — test additions
- [ ] `docs` — documentation
- [ ] `chore` — build / dependencies / CI

---

## Summary

<!-- 1-3 sentences: what changed and why -->

---

## Architecture Checklist

<!-- Confirm none of the core constraints are violated -->

- [ ] No heavy model inference (VLA / perception / grasp) inside harness — server is external
- [ ] New Tool implements the `Tool` Protocol; schema serializable to MCP tool definition
- [ ] New Skill has a valid `SkillManifest` (name / version / safety_class / required_tools); hotswap returns `SwapHandle`
- [ ] Skill calls capabilities via `tools.get(...)` — no direct connections to external servers
- [ ] All hardware-bound tool calls pass through `SafetyEnvelope.check()`; `SafetyEnvelopeViolation` is never caught and suppressed
- [ ] No `print()` debugging — structured logging via `observability.tracer`
- [ ] No `try/except Exception: pass` silently swallowing errors
- [ ] Single file ≤ 1000 lines; nesting ≤ 3 levels

---

## Test Coverage

- [ ] Hardware tests are marked `@pytest.mark.hardware` (skipped in CI by default)
- [ ] New Tool covers all four error paths: schema violation / backend unreachable / timeout / cancelled
- [ ] `SafetyEnvelope` is **not** mocked — safety tests use real validation logic
- [ ] New Skill end-to-end test validated on real hardware or high-fidelity sim (or marked `pending_hardware_validation`)

---

## Related Links

<!-- Issue, external server docs, etc. -->

- Issue / PR:
