"""Poker44 miner — ONNX MLP chunk-level scorer."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.miner_heuristics import (
    score_chunk_gen9model20,
    get_chunk_scorer_startup_check,
    chunk_payload_is_legacy,
    filter_other_actions_from_chunk,
)
from poker44.validator.synapse import DetectionSynapse


FORCED_VALIDATOR_HOTKEYS = {
    "5GgnyzhZ6ozkdnQumwRuEaULggvMr2np4SS3N7eDCMMrXoMC",
}

EXTRA_ALLOWED_VALIDATOR_HOTKEYS = {
    "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD",
}


class Miner(BaseMinerNeuron):
    """Poker44 ONNX MLP miner."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 ONNX MLP Miner started")

        chunk_scorer = "gen9model20"
        bt.logging.info("[init] POKER44_CHUNK_SCORER=gen9model20 (hardcoded)")
        bt.logging.info("[init] Chunk scorer active: gen9model20 ONNX MLP.")

        scorer_check = get_chunk_scorer_startup_check(chunk_scorer)
        if scorer_check.get("active"):
            details = scorer_check.get("details") or {}
            if scorer_check.get("ok"):
                bt.logging.info(
                    "[init] Chunk scorer startup check: ok "
                    f"scorer={scorer_check.get('scorer')} details={details}",
                )
            else:
                bt.logging.error(
                    "[init] Chunk scorer startup check: FAILED "
                    f"scorer={scorer_check.get('scorer')} "
                    f"error={scorer_check.get('error')} details={details}",
                )

        bt.logging.info(f"Axon created: {self.axon}")
        bt.logging.info(f"Build timestamp: {datetime.now(timezone.utc).isoformat()}")
        self._project_root = Path(__file__).resolve().parent.parent
        repo_root = Path(__file__).resolve().parents[1]
        try:
            _git_commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
        except Exception:
            _git_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "")
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44_gen9model20",
                "model_version": "9",
                "framework": "onnx-mlp",
                "license": "MIT",
                "repo_url": "https://github.com/tomkaba/poker44-miner-gen9v2",
                "repo_commit": _git_commit,
                "notes": "Poker44 ONNX MLP 20-feature chunk-level classifier.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on public human corpus and offline-generated bot chunks. "
                    "No validator-private data used."
                ),
                "training_data_sources": ["hands_generator/human_hands/poker_hands_combined.json.gz"],
                "private_data_attestation": "This miner does not train on validator-private human data.",
                "data_attestation": (
                    "This miner does not train on validator-private human data."
                ),
            },
        )

        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Model manifest: {self.model_manifest}")
        bt.logging.info(
            "Manifest source: "
            f"repo_url={self.model_manifest.get('repo_url', '')} "
            f"repo_commit={self.model_manifest.get('repo_commit', '')}"
        )

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk using gen9 ONNX model."""
        chunks: List[List[dict]] = synapse.chunks or []

        def _preview(seq, limit=8):
            if len(seq) <= limit:
                return seq
            return [*seq[:limit], "..."]

        chunk_sizes = [len(chunk or []) for chunk in chunks]
        bt.logging.debug(f"[miner] Received {len(chunks)} chunk(s); first sizes={_preview(chunk_sizes)}")

        remove_other_flag = os.getenv("REMOVE_OTHER", "0").strip().lower()
        remove_other_enabled = remove_other_flag in ("1", "true", "yes")

        if remove_other_enabled:
            filtered_chunks = []
            total_removed = 0
            for idx, chunk in enumerate(chunks):
                filtered_chunk, removed_count = filter_other_actions_from_chunk(chunk)
                filtered_chunks.append(filtered_chunk)
                total_removed += removed_count
                if removed_count > 0:
                    bt.logging.debug(f"[miner] chunk[{idx}] removing {removed_count} 'other' entries")
            chunks = filtered_chunks
            if total_removed > 0:
                bt.logging.debug(f"[miner] Total 'other' entries removed: {total_removed}")
            bt.logging.info(f"[REMOVE_OTHER] filter ran: removed={total_removed} chunks={len(chunks)}")
            chunk_sizes = [len(chunk or []) for chunk in chunks]

        processing_start = time.perf_counter()
        scores = []
        chunk_routes = []

        for index, chunk in enumerate(chunks):
                score, route = score_chunk_gen9model20(chunk)
            scores.append(score)
            chunk_routes.append(route)
            bt.logging.debug(
                f"[miner] chunk[{index}] size={len(chunk or [])} "
                f"route={route} score={float(score):.6f}"
            )

        processing_elapsed = time.perf_counter() - processing_start
        avg_ms = (processing_elapsed / max(1, len(chunks))) * 1000.0
        bt.logging.debug(
            "[miner] batch processing timing: "
            f"chunks={len(chunks)} total_seconds={processing_elapsed:.6f} "
            f"avg_ms_per_chunk={avg_ms:.3f}"
        )
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        bt.logging.debug(
            f"[miner] Responding with scores={_preview(scores)} "
            f"predictions={_preview(synapse.predictions)}"
        )
        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks.")

        source_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")
        self._append_request_log(
            validator_hotkey=source_hotkey,
            chunk_sizes=chunk_sizes,
            scores=scores,
            chunk_routes=chunk_routes,
            predictions=synapse.predictions,
            chunks=chunks,
        )

        return synapse

    @staticmethod
    def _flag_enabled(config_section, attr, default=None):
        value = getattr(config_section, attr, default) if config_section else default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _allowed_validator_hotkeys(self) -> set[str]:
        cfg = getattr(self.config, "blacklist", None)
        allowed = set(FORCED_VALIDATOR_HOTKEYS) | set(EXTRA_ALLOWED_VALIDATOR_HOTKEYS)

        def _normalize(value) -> set[str]:
            if value is None:
                return set()
            if isinstance(value, (list, tuple, set)):
                iterable = value
            else:
                iterable = str(value).split(",")
            return {str(item).strip() for item in iterable if str(item).strip()}

        allowed |= _normalize(getattr(cfg, "forced_validator_hotkey", None))
        allowed |= _normalize(getattr(cfg, "forced_validator_hotkeys", None))
        allowed |= _normalize(getattr(cfg, "extra_validator_hotkeys", None))
        return allowed

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        allow_non_registered = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "allow_non_registered",
            False,
        )
        force_validator_permit = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "force_validator_permit",
            True,
        )
        allowed_hotkeys = self._allowed_validator_hotkeys()

        if synapse.dendrite.hotkey in allowed_hotkeys:
            bt.logging.debug(f"Allowing validator hotkey {synapse.dendrite.hotkey}")
            return False, "Validator allowlist"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            if not allow_non_registered:
                bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
                return True, "Unrecognized hotkey"
            bt.logging.debug(
                f"Allowing unregistered hotkey {synapse.dendrite.hotkey} "
                "because allow_non_registered=True"
            )
            return False, "Unregistered hotkey allowed"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if force_validator_permit and not self.metagraph.validator_permit[uid]:
            bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)

    def _get_log_path(self) -> Path:
        uid = getattr(self, "uid", None)
        suffix = uid if uid is not None else "unknown"
        return self._project_root / f"miner_{suffix}.log"

    def _full_logging_enabled(self) -> bool:
        cfg = getattr(self.config, "logging", None)
        config_flag = getattr(cfg, "disable_full_logs", False)
        env_flag = os.getenv("POKER44_DISABLE_FULL_LOGS", "false").strip().lower()
        env_disable = env_flag in {"1", "true", "yes", "on"}
        return not (config_flag or env_disable)

    def _append_request_log(
        self,
        validator_hotkey,
        chunk_sizes,
        scores,
        chunk_routes,
        predictions,
        chunks,
    ) -> None:
        if not self._full_logging_enabled():
            bt.logging.debug("Full request logging disabled; skipping miner log entry.")
            return
        entry = {
            "timestamp": time.time(),
            "validator_hotkey": validator_hotkey,
            "miner_hotkey": getattr(self.wallet.hotkey, "ss58_address", "unknown"),
            "chunk_count": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "chunk_routes": chunk_routes,
            "scores": scores,
            "predictions": predictions,
            "chunks": chunks,
        }
        try:
            log_path = self._get_log_path()
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as log_error:
            bt.logging.warning(f"Failed to append miner request log: {log_error}")


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]} "
                "| Scorer: gen9model20"
            )
            time.sleep(5 * 60)
