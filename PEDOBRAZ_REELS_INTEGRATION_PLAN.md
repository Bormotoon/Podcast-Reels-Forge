## Plan: Integrate Reels Forge Into PedObraz Orchestrator

PedObraz remains the control plane and resource governor; Podcast Reels Forge becomes a managed stage-domain executed through PedObraz queueing, placement, leasing, retries, and operator UI. The integration should preserve PedObraz contracts and avoid direct local process ownership conflicts by replacing Forge-local lifecycle control with PedObraz-managed execution context.

**Steps**
1. Phase 1 - Discovery Baseline and Contract Freeze
1.1. Freeze current runtime contracts for both projects: stage graphs, queue semantics, lease keys, retry classes, artifact contracts, model endpoints, and service ownership boundaries.
1.2. Produce a mapping table from Forge stages to PedObraz-compatible stages with explicit queue/resource assignment.
1.3. Define hard constraints: PedObraz owns scheduling, queues, retries, and resource leases; Forge domain workers must not self-orchestrate global resources.

2. Phase 2 - Unified Domain Model in PedObraz (depends on 1)
2.1. Add a pipeline-domain discriminator in PedObraz intake and pipeline item metadata so items are routed either to news domain or reels domain without branching by ad hoc flags.
2.2. Extend stage registry and placement contract for reels stages, including queue, host role, resource key, timeout, max attempts, and expected model/backend.
2.3. Extend stage ordering and transition policy to support a second stage graph under the same conveyor framework.
2.4. Keep apply_stage_output and repository transition mechanics shared to avoid duplicating retry/publication state logic.

3. Phase 3 - Resource Governance Unification (depends on 2)
3.1. Introduce explicit resource keys for reels-heavy assets (example: llama.cpp lane, ffmpeg lane, subtitle-render lane, transcription lane) and map them to lease policy in PedObraz.
3.2. Replace Forge-local host ownership assumptions with PedObraz placement assertions and lease acquisition around heavy stages.
3.3. Add GPU admission policy for reels workloads that cooperates with existing GPU 5060 admission rules, preserving time-critical bypass for news and preventing starvation.
3.4. Consolidate model/server ownership: llama-server lifecycle should be controlled by PedObraz worker role, not by nested per-item subprocess ownership inside Forge orchestration.

4. Phase 4 - Execution Adapter Layer (depends on 2 and 3)
4.1. Build a PedObraz reels stage executor adapter that invokes Forge stage primitives directly (preferred) or via stable CLI contract (fallback), with strict StageExecutionInput/Output conformity.
4.2. Split Forge orchestration into reusable stage functions under a domain API so PedObraz can call transcribe, analyze, and video processing independently.
4.3. Ensure adapter returns canonical artifacts, provider/model metadata, metrics, and retryable/non-retryable failure classes for PedObraz policy engine.
4.4. Preserve idempotency by epoch using PedObraz stage run history and deterministic artifact hashes.

5. Phase 5 - Storage, Artifacts, and Config Unification (depends on 4)
5.1. Define shared runtime path rules and canonical artifact schema for reels outputs (transcript, moments, reels, subtitles, previews) under PedObraz runtime roots.
5.2. Unify environment and runtime settings via PedObraz config/state store; eliminate duplicate source-of-truth knobs for queues, endpoints, and model routing.
5.3. Standardize logging and metrics namespaces so reels stages appear in PedObraz operator views, queue snapshots, and alerting.

6. Phase 6 - Unified Control Plane and Operator UX (depends on 5)
6.1. Build a single Operations Hub in PedObraz admin for both domains (PedObrazNews and ReelsForge) with a domain switcher and combined global status strip.
6.2. Extend admin endpoints/pages to show reels pipeline items, stage runs, queue bottlenecks, lease waits, and per-stage artifact links.
6.3. Add operator controls for reels-specific retries, forced reruns, dead-letter requeue, and stage-level overrides.
6.4. Add temporal/task-queue diagnostics for newly added reels queues using existing queue ownership snapshots.

6A. Unified Admin Control Center
6A.1. Add domain health cards for News and Reels with SLA/SLO signal, queue depth, oldest-item age, retry storm indicator, dead-letter trend, and worker heartbeat consistency.
6A.2. Add a global Operational Mode switch with explicit policies: Balanced, News Priority, Reels Priority, Reels Exclusive Window.
6A.3. Add unified queue and lease views across both domains with live claim rate, lease wait duration, stage throughput, and head-of-line blocker diagnostics.
6A.4. Add per-domain and cross-domain deployment drift panels (git SHA, build ID, worker role version mismatch, stale pollers).

6B. Mandatory Pause/Resume PedObrazNews for Resource Reallocation
6B.1. Implement News runtime state machine: Running, PauseRequested, Draining, Paused, ResumeRequested, Resuming.
6B.2. Implement pause modes:
6B.2.1. Soft Pause: stop new News intake, allow in-flight stages to finish.
6B.2.2. Drain Pause: stop intake and new claims for News, finish only already-running work.
6B.2.3. Hard Pause: stop intake and claims, allow controlled cancel/terminate actions for selected workflows with safe lease release.
6B.3. Implement Resume flow with preflight checks, queue ownership restoration, and warmup concurrency caps for first minutes after resume.
6B.4. Ensure Pause News can automatically activate Reels Priority or Reels Exclusive Window policies to free constrained resources for Reels workloads.
6B.5. Log every pause/resume transition as auditable admin actions with operator identity, reason, timestamps, and affected queues/resources.

6C. Cross-Domain Resource Scheduler Panel
6C.1. Introduce resource budgets per domain (GPU, CPU, RAM, IO, concurrency for heavy stages).
6C.2. Add burst and cooldown limits to avoid starvation and oscillation when switching priorities.
6C.3. Add contention analyzer showing which stage/domain is holding scarce resources and who is waiting beyond threshold.
6C.4. Add one-click actions: Shift Capacity to Reels, Return Capacity to News, Freeze intake for domain, Release blocked lane by policy.

6D. Unified Error Center and Incident Operations
6D.1. Add one error inbox for News and Reels with grouping by fingerprint, stage, provider, queue, host, and recurrence rate.
6D.2. Add incident actions: acknowledge, silence window, bulk retry, move to dead-letter review, escalate.
6D.3. Add correlation of errors with resource pressure (GPU OOM/VRAM pressure, network/transient storms, provider degradation, poller gaps).
6D.4. Add runbook links and suggested remediation hints per top error group.

6E. Unified Statistics and Resource Monitoring
6E.1. Add multi-domain throughput dashboard: per-stage throughput, success/failure/retry/dead-letter ratios, and queue latency percentiles (p50/p95/p99).
6E.2. Add host and domain resource timelines: GPU utilization and VRAM, CPU/RAM/IO, queue wait vs resource pressure overlays.
6E.3. Add SLA/SLO board with freshness/completion/error-budget metrics and anti-noise alert thresholds.
6E.4. Add exportable incident and capacity reports for operator postmortems and planning.

7. Phase 7 - Migration and Cutover Strategy (depends on 6)
7.1. Start with shadow mode: PedObraz creates reels pipeline items and executes dry-run adapters without publishing outputs.
7.2. Enable canary on one queue/host profile with strict concurrency caps and lease observability.
7.3. Gradually move Forge entrypoint traffic to PedObraz orchestrator and disable direct autonomous Forge runs in production.
7.4. Keep rollback path: feature flag to route reels tasks back to standalone Forge execution until confidence gates pass.
7.5. Run canary and cutover only through Operations Hub controls; disallow ad hoc host-level toggles in production.
7.6. Make Pause News checkpoint mandatory before enabling Reels Exclusive Window in production cutover playbooks.
7.7. Add atomic rollback action in Operations Hub to restore News priority profile, re-enable News intake/claims, and disable Reels-exclusive policy.

8. Phase 8 - Validation and Hardening (depends on 7)
8.1. Contract tests: stage input/output schema, transition correctness, idempotency, retry/dead-letter policy, placement assertions.
8.2. Resource contention tests: mixed news+reels queue pressure on 5060/1060 lanes, lease fairness, starvation checks, admission-gate behavior.
8.3. Chaos tests: llama-server restarts, transient network failures, ffmpeg failures, subtitle pipeline exceptions, watchdog recovery.
8.4. Throughput/latency benchmarks before and after integration with SLO acceptance thresholds.
8.5. Pause/Resume reliability tests for Soft/Drain/Hard pause modes with lease safety and workflow consistency checks.
8.6. Reels Priority and Reels Exclusive Window tests under mixed load with deterministic resource reclaim guarantees.
8.7. Unified observability tests: resource metrics completeness, error inbox aggregation fidelity, and alert signal quality.
8.8. Operator workflow acceptance tests: one-click pause/resume, capacity shift, incident ack/retry/escalation, and audit trail completeness.

**Relevant files**
- /home/borm/VibeCoding/PedObraz/news/pipeline/stage_executor.py - Central stage dispatch, retry classification, artifact recording, and transition application; primary insertion point for reels domain execution.
- /home/borm/VibeCoding/PedObraz/news/pipeline/repository.py - Intake item creation, claim logic, queue ordering, stage advancement; needed for multi-domain routing and claim fairness.
- /home/borm/VibeCoding/PedObraz/news/pipeline/registry.py - Stage-to-queue/resource/timeout/max-attempt contract definitions.
- /home/borm/VibeCoding/PedObraz/news/pipeline/execution_placement_contract.py - Host/queue/model contract enforcement and placement assertions.
- /home/borm/VibeCoding/PedObraz/news/pipeline/resource_lease.py - Shared lease mechanism for scarce compute resources.
- /home/borm/VibeCoding/PedObraz/news/pipeline/gpu5060_admission.py - Admission guard for heavy GPU lane; must be extended for mixed domain fairness.
- /home/borm/VibeCoding/PedObraz/news/pipeline/worker_settings.py - Worker queue filters and execution owner gates.
- /home/borm/VibeCoding/PedObraz/news/automation/runtime.py - Runtime control loop and service-level orchestration context.
- /home/borm/VibeCoding/PedObraz/news/automation/temporal_admin.py - Queue ownership diagnostics and workflow control interfaces.
- /home/borm/VibeCoding/PedObraz/news/docs/pipeline/conveyor-architecture.md - Canonical conveyor runtime constraints and load profile.
- /home/borm/VibeCoding/PedObraz/news/docker-compose.yml - Control-plane service topology and environment contracts.
- /home/borm/VibeCoding/Podcast Reels Forge/podcast_reels_forge/pipeline.py - Current standalone orchestration to be decomposed into PedObraz-managed stage calls.
- /home/borm/VibeCoding/Podcast Reels Forge/podcast_reels_forge/stages/analyze_stage.py - Reels LLM staged analysis behavior and tunable concurrency.
- /home/borm/VibeCoding/Podcast Reels Forge/podcast_reels_forge/stages/transcribe_stage.py - Transcription quality/performance stage contract.
- /home/borm/VibeCoding/Podcast Reels Forge/podcast_reels_forge/scripts/video_processor.py - Heavy video render/export path and subtitle burn flow.
- /home/borm/VibeCoding/Podcast Reels Forge/podcast_reels_forge/utils/burned_subtitles.py - Subtitle rendering and artifact generation logic.
- /home/borm/VibeCoding/Podcast Reels Forge/podcast_reels_forge/utils/llama_cpp_service.py - Local llama.cpp lifecycle assumptions to be shifted under PedObraz ownership.
- /home/borm/VibeCoding/Podcast Reels Forge/config.yaml - Existing resource tuning knobs to be remapped into PedObraz runtime settings.
- /home/borm/VibeCoding/Podcast Reels Forge/start_forge.py - Standalone entrypoint to retain only for local dev/fallback mode.

**Verification**
1. Build an integration matrix test suite that validates queue-to-stage placement, lease acquisition/release, retry policy, and artifacts for both domains.
2. Run mixed-load benchmark with concurrent news and reels items, measuring queue wait, stage latency, GPU utilization, and failed retries per stage.
3. Execute failover drills: kill llama-server, force transient DB/network errors, and verify PedObraz retry/dead-letter outcomes remain deterministic.
4. Validate operator workflows: inspect queues, cancel/terminate/rerun pipelines, and review artifacts from admin pages and Temporal diagnostics.
5. Confirm rollback switch by toggling reels domain off in PedObraz and resuming standalone Forge path without data loss.
6. Validate Pause News and Resume News operational flow, including resource reclaim time and post-resume warmup behavior.
7. Validate unified monitoring dashboards and per-domain resource accounting against host-level telemetry ground truth.
8. Validate unified error center triage workflow end-to-end, including deduplication, suppression windows, and escalation.

**Acceptance Criteria: Unified Admin and Resource Control**
1. Operator can manage PedObrazNews and ReelsForge from one admin surface without host-level manual intervention.
2. Operator can pause PedObrazNews (Soft/Drain/Hard) to free constrained resources and run ReelsForge workloads predictably.
3. Operator can resume PedObrazNews with preflight checks and warmup guards, with no data loss or orphaned pipeline state.
4. Resource contention decisions are policy-driven (Balanced/News Priority/Reels Priority/Reels Exclusive) and fully observable.
5. Errors across both domains are visible in one triage inbox with actionable grouping and remediation workflow.
6. Every control action (pause/resume/policy switch/retry/escalation) is recorded in auditable history with operator, reason, and effect.

**Decisions**
- PedObraz is the only orchestrator and resource owner in production.
- Reels Forge becomes a stage-domain module, not a peer orchestrator.
- Resource arbitration stays lease-first with explicit stage placement contracts.
- Stage execution should prefer in-process adapter calls over shell subprocess orchestration where possible.
- Included scope: orchestration, resource governance, contracts, observability, migration.
- Excluded scope: redesign of editorial logic/prompt semantics in either domain.

**Further Considerations**
1. Integration mode recommendation: start with adapter-through-CLI for fast parity, then migrate hot stages to direct function calls for lower overhead and better observability.
2. GPU fairness recommendation: keep news time-critical bypass, but reserve bounded reels capacity windows to avoid permanent starvation during spikes.
3. Config source recommendation: migrate Forge performance knobs into PedObraz runtime settings with namespaced keys and one authoritative admin control path.

## Implementation Backlog (Epic -> Feature -> Story)

### Epic E0 - Foundation and Contract Baseline (Priority P0)
Goal: freeze contracts and remove ambiguity before active integration work.

Feature E0.F1 - Contract inventory and compatibility matrix (P0)
1. Story E0.F1.S1: Capture current stage graphs for PedObrazNews and ReelsForge.
2. Story E0.F1.S2: Capture queue and lease contracts (queue names, resource keys, timeout, max attempts).
3. Story E0.F1.S3: Build a compatibility matrix for artifacts, retries, and stage outcomes.
4. Story E0.F1.S4: Approve and freeze the contract document with version tag.

Feature E0.F2 - Integration acceptance baseline (P0)
1. Story E0.F2.S1: Define acceptance gates for pause/resume, unified admin, and resource arbitration.
2. Story E0.F2.S2: Define observability minimum set (metrics, logs, events, audit fields).
3. Story E0.F2.S3: Define rollback invariants and recovery playbook entry criteria.

### Epic E1 - Multi-Domain Pipeline Model in PedObraz (Priority P0)
Goal: make PedObraz explicitly domain-aware while preserving common transition mechanics.

Feature E1.F1 - Domain discriminator and routing model (P0)
1. Story E1.F1.S1: Add pipeline domain discriminator to intake and pipeline item metadata.
2. Story E1.F1.S2: Route items by domain without ad hoc flags.
3. Story E1.F1.S3: Add domain-aware validation for stage transitions.

Feature E1.F2 - Stage registry and placement extension (P0)
1. Story E1.F2.S1: Register Reels stages in stage registry with queue/resource/timeout/attempt contracts.
2. Story E1.F2.S2: Extend execution placement contract assertions for Reels queues and model lanes.
3. Story E1.F2.S3: Add migration-safe fallback behavior for unknown stage/domain pairs.

### Epic E2 - Reels Execution Adapter and Stage API Extraction (Priority P0)
Goal: execute Reels stages under PedObraz orchestration with canonical stage IO.

Feature E2.F1 - Adapter layer for stage execution (P0)
1. Story E2.F1.S1: Implement Reels adapter in stage executor with strict StageExecutionInput/Output conformity.
2. Story E2.F1.S2: Normalize retryable vs non-retryable error mapping.
3. Story E2.F1.S3: Emit canonical artifacts and provider/model attribution.

Feature E2.F2 - Forge orchestration decomposition (P0)
1. Story E2.F2.S1: Extract reusable stage-level API functions from Forge orchestrator.
2. Story E2.F2.S2: Keep CLI adapter as temporary compatibility path.
3. Story E2.F2.S3: Add idempotency guards via deterministic artifact hashes and stage-run history.

### Epic E3 - Unified Resource Governance and Scheduling (Priority P0)
Goal: enforce fair, observable, policy-driven resource arbitration across News and Reels.

Feature E3.F1 - Resource key and lease policy extension (P0)
1. Story E3.F1.S1: Add resource keys for transcribe, llm analyze, video render, subtitle render.
2. Story E3.F1.S2: Map keys to lease pools and queue admission limits.
3. Story E3.F1.S3: Add lease fairness checks and starvation watchdog rules.

Feature E3.F2 - Cross-domain scheduler policies (P0)
1. Story E3.F2.S1: Implement policy profiles: Balanced, News Priority, Reels Priority, Reels Exclusive Window.
2. Story E3.F2.S2: Add burst/cooldown controls for policy switches.
3. Story E3.F2.S3: Add contention analyzer signals and operator hints.

### Epic E4 - Unified Admin Operations Hub (Priority P0)
Goal: control both domains from one place with operational safety and auditability.

Feature E4.F1 - Unified dashboard and domain control (P0)
1. Story E4.F1.S1: Add domain switcher and global status strip.
2. Story E4.F1.S2: Add unified queue, lease, and throughput views.
3. Story E4.F1.S3: Add unified deployment drift panel.

Feature E4.F2 - Mandatory Pause/Resume News controls (P0)
1. Story E4.F2.S1: Implement News state machine: Running, PauseRequested, Draining, Paused, ResumeRequested, Resuming.
2. Story E4.F2.S2: Implement pause modes Soft, Drain, Hard with clear preconditions/postconditions.
3. Story E4.F2.S3: Implement resume preflight and warmup concurrency caps.
4. Story E4.F2.S4: Add one-click actions to shift capacity between domains.

Feature E4.F3 - Unified error center and incident ops (P1)
1. Story E4.F3.S1: Aggregate error events by fingerprint, stage, provider, queue, host.
2. Story E4.F3.S2: Add incident actions: ack, silence, retry set, escalate.
3. Story E4.F3.S3: Correlate top errors with resource pressure and queue health.

### Epic E5 - Data, Config, and Observability Unification (Priority P0)
Goal: one source of truth for config and consistent observability across domains.

Feature E5.F1 - Config unification (P0)
1. Story E5.F1.S1: Migrate Forge tuning knobs into PedObraz settings namespace.
2. Story E5.F1.S2: Remove duplicated runtime knobs and enforce precedence rules.
3. Story E5.F1.S3: Add config audit events for operational changes.

Feature E5.F2 - Metrics and monitoring (P0)
1. Story E5.F2.S1: Add multi-domain throughput and latency metrics (p50/p95/p99).
2. Story E5.F2.S2: Add host and domain resource metrics (GPU/VRAM, CPU/RAM/IO).
3. Story E5.F2.S3: Add SLA/SLO and error-budget board with anti-noise alerts.

### Epic E6 - Migration, Canary, and Rollback Automation (Priority P0)
Goal: perform cutover safely through admin-controlled procedures.

Feature E6.F1 - Controlled migration path (P0)
1. Story E6.F1.S1: Enable shadow mode through Operations Hub only.
2. Story E6.F1.S2: Enable canary with strict queue and concurrency caps.
3. Story E6.F1.S3: Require Pause News checkpoint before Reels Exclusive Window.

Feature E6.F2 - Atomic rollback (P0)
1. Story E6.F2.S1: Add one action to restore News priority profile and disable Reels-exclusive mode.
2. Story E6.F2.S2: Re-enable News intake/claims with warmup.
3. Story E6.F2.S3: Validate post-rollback consistency and queue recovery.

### Epic E7 - Verification and Hardening (Priority P0)
Goal: prove reliability under mixed load, failures, and operator workflows.

Feature E7.F1 - Contract and behavior tests (P0)
1. Story E7.F1.S1: Validate stage IO schema and transition correctness for both domains.
2. Story E7.F1.S2: Validate idempotency and retry/dead-letter rules.
3. Story E7.F1.S3: Validate placement and queue ownership assertions.

Feature E7.F2 - Operational reliability tests (P0)
1. Story E7.F2.S1: Test pause/resume reliability for Soft/Drain/Hard modes.
2. Story E7.F2.S2: Test mixed-load contention on 5060/1060 with policy switches.
3. Story E7.F2.S3: Test chaos scenarios (model server restarts, transient DB/network, ffmpeg/subtitle failures).

Feature E7.F3 - Operator acceptance tests (P1)
1. Story E7.F3.S1: Validate one-click operations for pause/resume/capacity shift.
2. Story E7.F3.S2: Validate incident workflows in unified error center.
3. Story E7.F3.S3: Validate audit completeness for all control actions.

## Iteration Roadmap

### Iteration 1 (P0)
1. E0 foundation and acceptance baseline.
2. E1 multi-domain model and contract extensions.
3. E2 adapter skeleton and initial stage API extraction.

Exit criteria:
1. Domain-aware intake and stage registry are live behind feature flags.
2. Reels adapter can run dry-run stage execution with canonical output.

### Iteration 2 (P0)
1. E3 unified resource governance and policy engine.
2. E4 unified Operations Hub baseline.
3. E4 mandatory Pause/Resume News controls.

Exit criteria:
1. Operator can pause News and switch capacity to Reels from one admin panel.
2. Policy modes affect scheduling and lease behavior as expected.

### Iteration 3 (P0)
1. E5 config and observability unification.
2. E6 migration/canary/rollback automation.
3. E7 core reliability test suite.

Exit criteria:
1. Shadow and canary operate via Operations Hub only.
2. Atomic rollback flow is validated end-to-end.

### Iteration 4 (P1/P2 hardening)
1. E4 unified error center advanced workflows.
2. E7 operator acceptance and non-functional hardening.
3. UX refinements and reporting polish.

Exit criteria:
1. Unified incident workflows are production-ready.
2. Monitoring and audit reports satisfy operator/postmortem requirements.

## Priority Summary
1. P0: E0, E1, E2, E3, E4.F1-F2, E5, E6, E7.F1-F2.
2. P1: E4.F3, E7.F3, iteration-4 hardening and advanced incident UX.
3. P2: optional UX/reporting enhancements that do not affect control-plane safety.

## Dependency Chain (Critical Path)
1. E0 -> E1 -> E2 -> E3 -> E4.F2 -> E6 -> E7.
2. E5 runs in parallel after E2, but must finish before production cutover gate.
3. E4.F3 can start after E5 telemetry baseline is available.
