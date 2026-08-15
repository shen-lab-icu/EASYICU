# Canonical9 owner activation packet

Date: 2026-07-22 EDT  
Task: `FIG2-CANONICAL9-GATE/OWNER-ACTIVATION-PACKET`
Commit: `4ad31c8 docs(figure2): add canonical9 owner activation packet`

`benchmarks/figure2_canonical9/canonical9_owner_activation_packet.md` is a
non-executable owner checklist for the existing P4 real-run authorization
gate. It records no patient data, mapping contents, credentials, model output,
or inferred clinical decision.

It requires one selected data route (fresh native export or controlled identity
bridge), frozen source/identity evidence, E2/H2/H3 clinical and methods review,
and the full P4 operator-freeze pins. It explicitly preserves:

- identity bridge `real_run_authorized=false`;
- exact ordered full-nine `aware` arm only;
- no cross-run memory, development sample, resume, writer probe, or output reuse;
- final P4 comparison against the actual command line before any Provider,
  Docker, or data activity.

No real run was requested or started by preparing this packet.
