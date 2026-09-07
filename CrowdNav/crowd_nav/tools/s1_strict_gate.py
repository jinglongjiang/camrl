#!/usr/bin/env python3
"""s1_strict_gate.py: the single CLI entry point for S1's stages
(guide.md section 13, Order S1-0/S1-0R, 2026-08-04).

Every stage below is a thin dispatcher onto ``crowd_nav.bayesian_brne.
s1_protocol`` -- this file must never reimplement registry parsing,
identity checks, or any statistical logic itself (guide.md: "统计逻辑不得
散落在CLI脚本里").

All frozen S1 stages are implemented. Each delegates to ``s1_pipeline``;
scientific and identity rules remain centralized in ``s1_protocol``.

The repo root is ALWAYS derived from this file's own on-disk location
(``crowd_nav.bayesian_brne.s1_protocol.repo_root()``) -- never a hardcoded
host path. Order S1-0R (R0R-1) fixed a real bug where this file hardcoded
one developer machine's absolute checkout path, which does not exist at
all on the 4090 training host (a different user account, a different
directory tree); code synced there and run as-is would have silently
read/written the wrong location.

Usage:
    python3 -m crowd_nav.tools.s1_strict_gate \\
      --registry crowd_nav/configs/s1_strict_registry.json \\
      --stage preflight|collect_necessity|fit_ac|select_k|fit_controls|necessity|collect_audit|audit|promote \\
      [--protocol-spec-path /path/to/guide.md]   # local-only, optional; see s1_protocol.freeze_protocol_spec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from crowd_nav.bayesian_brne import s1_protocol as sp
from crowd_nav.bayesian_brne import s1_pipeline as pipeline


def _output_root(repo_root: Path, registry: dict) -> Path:
    return repo_root / "runs" / "bayesian_brne" / registry["experiment_id"]


_ROLE_REPORT_BULK_FIELDS = ("initial_state_hashes", "episode_seeds", "episode_ids")


def _compact_role_report(report: dict) -> dict:
    """Drop the large per-episode identity lists (kept in-memory for the
    cross-role disjoint check) from what actually gets written to
    preflight_manifest.json -- the *_sha256 summary fields already make
    the full lists' content auditable/comparable without embedding
    thousands of entries in a manifest meant to be read by a human."""
    return {k: v for k, v in report.items() if k not in _ROLE_REPORT_BULK_FIELDS}


def _run_preflight(registry_path: str, protocol_spec_path: str = None, invocation_mode: str = "direct") -> int:
    """Order S1-0's real stage, hardened by Order S1-0R's independent
    audit: load+validate the registry, freeze it (or verify it has not
    changed since freezing), build/verify a SCAFFOLD source manifest
    (soft-drift-tolerant on the S1 CLI/protocol files themselves, hard on
    everything else), strictly verify the train/selection data roles
    against ``data_io.load_episode`` (not a raw ``np.load``), scan every
    existing SM-BRNE episode for a seed collision against the three fresh
    data roles, and (on first run only) create + restore-verify the
    one-time rollback archive. Writes frozen_registry.json/
    source_manifest_s1_0_scaffold.json/preflight_manifest.json/
    status.json/controller.log under the registry's own experiment output
    root, whose location is ALWAYS ``<repo_root>/runs/bayesian_brne/
    <experiment_id>`` -- repo_root computed from this file's own location,
    never a literal path."""
    log_lines = []

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}")

    repo_root = sp.repo_root()
    verdict = "PASS"
    reason = ""
    registry = None
    out_root = None
    try:
        log("[preflight] repo_root resolved to:", str(repo_root))
        log("[preflight] loading and validating registry:", registry_path)
        registry = sp.load_registry(registry_path)
        registry_hash = sp.registry_content_sha256(registry)
        out_root = _output_root(repo_root, registry)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "data_manifests").mkdir(parents=True, exist_ok=True)

        stage_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        sp.update_status_atomic(
            str(out_root), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
            invocation_mode=invocation_mode, stage_pid=os.getpid(), current_stage="preflight",
            stage_status="RUNNING", stage_started_at=stage_started_at, stage_finished_at=None,
        )

        frozen_registry_path = out_root / "frozen_registry.json"
        if frozen_registry_path.exists():
            with open(frozen_registry_path) as f:
                frozen_registry = json.load(f)
            frozen_hash = sp.registry_content_sha256(frozen_registry)
            if frozen_hash != registry_hash:
                raise sp.RegistryError(
                    f"registry content changed since S1-0 froze it: frozen_sha256={frozen_hash} "
                    f"current_sha256={registry_hash} -- guide.md 13 forbids modifying pre-registered "
                    "constants after freezing"
                )
            log("[preflight] frozen_registry.json already exists and matches current registry content -- OK")
        else:
            with open(frozen_registry_path, "w") as f:
                json.dump(registry, f, indent=2, sort_keys=True)
                f.write("\n")
            log("[preflight] froze registry ->", str(frozen_registry_path), "sha256=", registry_hash)

        # R0R-4: this is a SCAFFOLD snapshot (the S1-0 rollback baseline),
        # never a "method lock" -- the three S1 CLI/protocol files are
        # expected to keep changing through S1-1..S1-BUILD, so drift on
        # exactly those paths is soft (reported, not fatal) until a real
        # method_lock.json exists.
        method_lock_path = out_root / "method_lock.json"
        scaffold_path = out_root / "source_manifest_s1_0_scaffold.json"
        if method_lock_path.exists():
            method_manifest_path = out_root / "method_source_manifest.json"
            if not method_manifest_path.exists():
                raise sp.PreflightError("method_lock.json exists but method_source_manifest.json is missing")
            method_lock = json.loads(method_lock_path.read_text())
            if method_lock.get("method_source_manifest_sha256") != sp.provenance.sha256_file(method_manifest_path):
                raise sp.PreflightError("method_lock.json does not match method_source_manifest.json")
            method_manifest = json.loads(method_manifest_path.read_text())
            drift_report = sp.verify_source_manifest_unchanged(
                str(repo_root), method_manifest, soft_paths=frozenset(),
            )
            source_manifest = method_manifest
            log("[preflight] method lock present: final method source manifest is unchanged -- OK")
        else:
            log("[preflight] method lock absent: verifying development scaffold with explicit soft paths ...")
            source_manifest = sp.build_source_manifest(str(repo_root))
            if source_manifest["missing_files"]:
                raise sp.PreflightError(f"source manifest references missing files: {source_manifest['missing_files']}")
            if scaffold_path.exists():
                with open(scaffold_path) as f:
                    frozen_source_manifest = json.load(f)
                drift_report = sp.verify_source_manifest_unchanged(
                    str(repo_root), frozen_source_manifest,
                    soft_paths=sp.SOURCE_MANIFEST_SOFT_PATHS_BEFORE_METHOD_LOCK,
                )
                if drift_report["soft_changed"]:
                    log("[preflight] expected development drift:", drift_report["soft_changed"])
                log("[preflight] scaffold hard-tracked hashes unchanged -- OK")
            else:
                with open(scaffold_path, "w") as f:
                    json.dump(source_manifest, f, indent=2, sort_keys=True)
                    f.write("\n")
                log("[preflight] froze scaffold source manifest ->", str(scaffold_path))
                drift_report = {"soft_changed": [], "hard_changed": []}

        # The repo-COMMITTED protocol spec (sp.COMMITTED_PROTOCOL_SPEC_REL_PATH)
        # is tracked in SOURCE_MANIFEST_FILES, so its hash is ALREADY
        # hard-verified above by build_source_manifest/
        # verify_source_manifest_unchanged, unconditionally, on every host
        # -- including the 4090, which never passes --protocol-spec-path
        # at all (guide.md A1 item 5: "4090即使没有本地guide.md，也必须
        # 同步并校验仓库内已冻结的protocol spec/hash，不能跳过就算通过").
        # --protocol-spec-path is an ADDITIONAL, LOCAL-ONLY consistency
        # check: does the live guide.md still agree with what is
        # committed? It never auto-corrects a mismatch -- only the
        # explicit, separately-invoked amend_committed_protocol_spec may.
        if protocol_spec_path:
            log("[preflight] verifying live guide.md still matches the committed protocol spec:", protocol_spec_path)
            guide_text = Path(protocol_spec_path).read_text()
            sp.verify_committed_protocol_spec_matches_live_guide(str(repo_root), guide_text)
            log("[preflight] live guide.md matches committed", sp.COMMITTED_PROTOCOL_SPEC_REL_PATH, "-- OK")
            log("[preflight] freezing/verifying local runs/-audit-trail protocol spec snapshot ...")
            protocol_spec_result = sp.freeze_protocol_spec(str(out_root), guide_text)
            log("[preflight] frozen_protocol_spec.md (audit trail only):", protocol_spec_result)
        else:
            protocol_spec_result = None
            log("[preflight] --protocol-spec-path not given -- skipping the LOCAL live-guide.md consistency "
                "check (expected/fine on the 4090 host, which has no local guide.md); the committed "
                f"{sp.COMMITTED_PROTOCOL_SPEC_REL_PATH} hash is still hard-verified above regardless")

        log("[preflight] strictly verifying train/selection data roles via data_io.load_episode ...")
        train_report = sp.verify_formal_data_role(str(repo_root), "train", registry)
        log(f"[preflight] train: {train_report['n_files']} files, seed_counts={train_report['seed_counts']}")
        selection_report = sp.verify_formal_data_role(str(repo_root), "selection", registry)
        log(f"[preflight] selection: {selection_report['n_files']} files, seed_counts={selection_report['seed_counts']}")
        sp.verify_train_selection_cross_disjoint(train_report, selection_report)
        log("[preflight] train/selection initial_state_hash cross-check: disjoint -- OK")

        log("[preflight] scanning all existing SM-BRNE episodes under runs/ for seed collisions ...")
        scan = sp.scan_existing_episode_identities([str(repo_root / "runs")])
        log(f"[preflight] scanned={scan.n_files_scanned} matched={scan.n_files_matched} "
            f"skipped_non_episode={scan.n_files_skipped_non_episode}")
        disjoint_report = sp.check_seed_disjoint(registry, scan)
        log("[preflight] seed collision check: PASS (no overlap) --", disjoint_report["reserved_seeds"])

        rollback_manifest_path = out_root / "rollback" / "sm_brne_s1_0_rollback_manifest.json"
        if rollback_manifest_path.exists():
            with open(rollback_manifest_path) as f:
                rollback_manifest = json.load(f)
            log("[preflight] rollback archive already exists at", rollback_manifest["archive_path"],
                "-- not regenerating (one-time S1-0 snapshot); re-verifying restorability ...")
            if not sp.verify_rollback_archive(str(repo_root), rollback_manifest):
                raise sp.PreflightError("existing rollback archive failed restore verification (tar hash or content mismatch)")
            log("[preflight] existing rollback archive restore-verified: OK")
        else:
            log("[preflight] creating one-time S1-0 rollback archive ...")
            rollback_manifest = sp.build_rollback_archive(str(repo_root), str(out_root / "rollback"))
            log(f"[preflight] rollback archive: {rollback_manifest['n_files']} files, "
                f"sha256={rollback_manifest['archive_sha256']}")
            log("[preflight] verifying rollback archive restores correctly ...")
            if not sp.verify_rollback_archive(str(repo_root), rollback_manifest):
                raise sp.PreflightError("rollback archive failed restore verification immediately after creation")
            log("[preflight] rollback archive restore-verified: OK")

        preflight_manifest = {
            "experiment_id": registry["experiment_id"],
            "registry_path": str(registry_path),
            "registry_sha256": registry_hash,
            "source_manifest_kind": "method" if method_lock_path.exists() else "scaffold",
            "source_manifest_sha256": sp.provenance.sha256_file(
                out_root / "method_source_manifest.json" if method_lock_path.exists() else scaffold_path
            ),
            "scaffold_source_drift": drift_report,
            "protocol_spec_freeze": protocol_spec_result,
            "train_role_report": _compact_role_report(train_report),
            "selection_role_report": _compact_role_report(selection_report),
            "seed_disjoint_report": disjoint_report,
            "rollback_manifest_path": str(rollback_manifest_path),
            "rollback_archive_sha256": rollback_manifest["archive_sha256"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verdict": "PASS",
        }
        sp.update_status_atomic(
            str(out_root), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
            invocation_mode=invocation_mode, stage_pid=None, current_stage="preflight",
            stage_status="PASS", stage_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), returncode=0,
        )
    except (sp.RegistryError, sp.SeedCollisionError, sp.PreflightError, sp.DataRoleIntegrityError) as exc:
        verdict = "PRECHECK_FAIL"
        reason = str(exc)
        log("[preflight] PRECHECK_FAIL:", reason)
        preflight_manifest = {
            "experiment_id": registry["experiment_id"] if registry else None,
            "registry_path": str(registry_path),
            "verdict": "PRECHECK_FAIL",
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if out_root is None:
            # Registry itself never even loaded -- no real experiment_id
            # or output root exists yet; fall back to a diagnostics-only
            # location (not the real per-experiment status.json, which
            # requires a genuine identity to protect).
            out_root = repo_root / "runs" / "bayesian_brne" / "s1_preflight_failures"
            out_root.mkdir(parents=True, exist_ok=True)
        else:
            sp.update_status_atomic(
                str(out_root), experiment_id=registry["experiment_id"], registry_sha256=sp.registry_content_sha256(registry),
                invocation_mode=invocation_mode, stage_pid=None, current_stage="preflight",
                stage_status="PRECHECK_FAIL", stage_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), returncode=1,
            )

    with open(out_root / "preflight_manifest.json", "w") as f:
        json.dump(preflight_manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    controller_log_path = out_root / "controller.log"
    with open(controller_log_path, "a") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n[s1_strict_gate] stage=preflight verdict={verdict}")
    print(f"[s1_strict_gate] manifest: {out_root / 'preflight_manifest.json'}")
    return 0 if verdict == "PASS" else 1


def _run_stage(registry_path: str, stage: str, invocation_mode: str) -> int:
    registry = sp.load_registry(registry_path)
    out = _output_root(sp.repo_root(), registry)
    registry_hash = sp.registry_content_sha256(registry)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    sp.update_status_atomic(
        str(out), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
        invocation_mode=invocation_mode, stage_pid=os.getpid(), current_stage=stage,
        stage_status="RUNNING", stage_started_at=started, stage_finished_at=None,
    )
    try:
        pipeline._method_manifest(registry)
        if stage == "collect_necessity":
            result = pipeline.collect_role(registry, "necessity_id")
        elif stage == "fit_ac":
            result = pipeline.fit_grid(registry, pipeline.VARIANT_ACTION_CONDITIONED, registry["primary_k_candidates"])
        elif stage == "select_k":
            result = pipeline.select_models(registry)
        elif stage == "fit_controls":
            selection = json.loads((out / "selection_report.json").read_text())
            if selection.get("status") != "SELECTED":
                result = {"status": "SKIPPED", "reason": "selection_not_passed"}
            else:
                K = int(selection["selected_k"])
                pipeline.build_shuffle(registry)
                result = {
                    "self_only": pipeline.fit_grid(registry, pipeline.VARIANT_SELF_ONLY, [K]),
                    "constrained_action_shuffle": pipeline.fit_grid(registry, pipeline.VARIANT_SHUFFLE, [K]),
                }
        elif stage == "necessity":
            result = pipeline.run_necessity(registry)
        elif stage == "collect_audit":
            result = pipeline.unlock_and_collect_audit(registry)
        elif stage == "audit":
            result = pipeline.run_audit(registry)
        elif stage == "promote":
            result = pipeline.promote(registry)
        else:
            raise pipeline.S1PipelineError(f"unknown stage {stage}")
        stage_status = str(result.get("status", result.get("verdict", "PASS"))) if isinstance(result, dict) else "PASS"
        sp.update_status_atomic(
            str(out), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
            invocation_mode=invocation_mode, stage_pid=None, current_stage=stage,
            stage_status=stage_status, stage_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), returncode=0,
        )
        print(f"[s1_strict_gate] stage={stage} status={stage_status}", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "stage": stage, "status": "INCONCLUSIVE", "error_type": type(exc).__name__,
            "reason": str(exc), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _atomic_path = out / f"{stage}_failure.json"
        pipeline._atomic_json(_atomic_path, failure)
        sp.update_status_atomic(
            str(out), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
            invocation_mode=invocation_mode, stage_pid=None, current_stage=stage,
            stage_status="INCONCLUSIVE", stage_finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), returncode=1,
        )
        print(f"[s1_strict_gate] stage={stage} INCONCLUSIVE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--stage", required=True,
        choices=["preflight", "collect_necessity", "fit_ac", "select_k", "fit_controls",
                 "necessity", "collect_audit", "audit", "promote"],
    )
    parser.add_argument(
        "--protocol-spec-path", default=None,
        help="Local-only, optional: path to guide.md, whose section 13 gets frozen/verified into "
             "frozen_protocol_spec.md. Omit on hosts (e.g. the 4090) that have no local guide.md.",
    )
    parser.add_argument(
        "--invocation-mode", default="direct", choices=list(sp.VALID_INVOCATION_MODES),
        help="Set to 'queue' only by run_s1_queue.py's own subprocess call -- never pass this by hand.",
    )
    parser.add_argument(
        "--lock-method", action="store_true",
        help="After a successful preflight, atomically freeze the completed S1-BUILD source manifest.",
    )
    args = parser.parse_args()

    if args.stage == "preflight":
        rc = _run_preflight(args.registry, args.protocol_spec_path, args.invocation_mode)
        if rc == 0 and args.lock_method:
            registry = sp.load_registry(args.registry)
            lock = pipeline.create_method_lock(registry)
            print("[s1_strict_gate] method lock created/verified:", lock)
        return rc
    if args.lock_method:
        parser.error("--lock-method is only valid with --stage preflight")
    return _run_stage(args.registry, args.stage, args.invocation_mode)


if __name__ == "__main__":
    sys.exit(main())
