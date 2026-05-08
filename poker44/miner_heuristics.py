"""Poker44 miner scoring module."""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np

_KNOWN_ATYPES: Set[str] = {"bet", "call", "check", "fold", "raise"}
_LEGACY_PLACEHOLDERS: Set[str] = {"", "action"}
_EPS = 1e-9

LEGACY_ACTION_PLACEHOLDERS: Set[str] = _LEGACY_PLACEHOLDERS


def _extract_actor_seat(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def _hand_has_modern_actions(hand: dict) -> bool:
    for action in hand.get("actions") or []:
        at = str(action.get("action_type") or "").strip().lower()
        if at and at not in _LEGACY_PLACEHOLDERS:
            return True
        if _extract_actor_seat(action.get("actor_seat")) > 0:
            return True
        if float(action.get("normalized_amount_bb") or 0.0) != 0.0:
            return True
        if float(action.get("amount") or 0.0) != 0.0:
            return True
    return False


def chunk_payload_is_legacy(chunk: List[dict]) -> bool:
    if not chunk:
        return True
    for hand in chunk:
        if isinstance(hand, dict) and _hand_has_modern_actions(hand):
            return False
    return True


def filter_other_actions_from_chunk(chunk: List[dict]) -> Tuple[List[dict], int]:
    if not chunk:
        return chunk, 0
    removed = 0
    out: List[dict] = []
    for item in chunk:
        if isinstance(item, dict) and isinstance(item.get("actions"), list):
            orig = item.get("actions") or []
            kept = [a for a in orig if not (isinstance(a, dict) and str(a.get("action_type") or "").strip().lower() == "other")]
            n = len(orig) - len(kept)
            removed += n
            out.append(dict(item, actions=kept) if n else item)
            continue
        if isinstance(item, dict) and str(item.get("action_type") or "").strip().lower() == "other":
            removed += 1
            continue
        out.append(item)
    return out, removed


def _cfeats(chunk: List[dict]) -> Dict[str, float]:
    _ac: Counter = Counter()
    _sc: Counter = Counter()
    _pc: Counter = Counter()
    _aph: List[float] = []
    _rbb: List[float] = []
    _bbb: List[float] = []
    _pv: List[float] = []
    _plc: List[float] = []
    _sd: List[float] = []
    _shd = 0

    for hand in chunk:
        _acts = hand.get("actions") or []
        _out = hand.get("outcome") or {}
        _pls = hand.get("players") or []
        _sts = hand.get("streets") or []
        _aph.append(float(len(_acts)))
        _plc.append(float(len(_pls)))
        _sd.append(float(len(_sts)))
        _pv.append(float(_out.get("total_pot") or 0.0))
        if bool(_out.get("showdown")):
            _shd += 1
        for a in _acts:
            at = str(a.get("action_type") or "other")
            if at not in _KNOWN_ATYPES:
                continue
            _ac[at] += 1
            _sc[str(a.get("street") or "unknown")] += 1
            _pc[str(a.get("actor_seat") or "?")] += 1
            bb = float(a.get("normalized_amount_bb") or 0.0)
            if at == "raise":
                _rbb.append(bb)
            elif at == "bet":
                _bbb.append(bb)

    _n = len(chunk)
    _ta = sum(_ac.values())
    _d = max(1, _ta)

    def _m(v: List[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    def _s(v: List[float]) -> float:
        if len(v) < 2:
            return 0.0
        m = sum(v) / len(v)
        return float(math.sqrt(sum((x - m) ** 2 for x in v) / len(v)))

    def _h(c: Counter) -> float:
        t = sum(c.values())
        if t <= 0:
            return 0.0
        return float(-sum((v / t) * math.log(v / t + _EPS) for v in c.values()))

    return {
        "action_entropy": _h(_ac),
        "actions_per_hand_mean": _m(_aph),
        "actions_per_hand_std": _s(_aph),
        "actions_total": float(_ta),
        "actor_entropy": _h(_pc),
        "aggression_ratio": ((_ac.get("raise", 0) + _ac.get("bet", 0)) / max(1, _ac.get("call", 0) + _ac.get("check", 0))),
        "bet_bb_mean": _m(_bbb),
        "bet_bb_std": _s(_bbb),
        "bet_ratio": _ac.get("bet", 0) / _d,
        "call_ratio": _ac.get("call", 0) / _d,
        "check_ratio": _ac.get("check", 0) / _d,
        "chunk_size": float(_n),
        "fold_ratio": _ac.get("fold", 0) / _d,
        "players_mean": _m(_plc),
        "players_std": _s(_plc),
        "pot_mean": _m(_pv),
        "pot_std": _s(_pv),
        "raise_bb_mean": _m(_rbb),
        "raise_bb_std": _s(_rbb),
        "raise_ratio": _ac.get("raise", 0) / _d,
        "showdown_rate": _shd / max(1, _n),
        "street_depth_mean": _m(_sd),
        "street_depth_std": _s(_sd),
        "street_entropy": _h(_sc),
    }


def _xcount(chunk: List[dict]) -> Dict[str, int]:
    _r: Dict[str, int] = {"a": 0, "b": 0, "c": 0}
    for hand in chunk:
        _acts = hand.get("actions") or []
        _seen: Set[object] = set()
        _fa = _fb = _fc = False
        for a in _acts:
            _seat = a.get("actor_seat")
            _at = (a.get("action_type") or "").lower()
            if _at == "fold":
                _seen.add(_seat)
            elif _seat in _seen:
                _fa = True
            if _at == "raise" and a.get("raise_to") is None:
                _fb = True
            if _at == "call" and a.get("call_to") is None:
                _fc = True
        if _fa:
            _r["a"] += 1
        if _fb:
            _r["b"] += 1
        if _fc:
            _r["c"] += 1
    return _r


_SES = None
_PROF: Optional[dict] = None
_LERR: Optional[str] = None
_PS = 8.0


def _load_gen9model20() -> bool:
    global _SES, _PROF, _LERR
    if _SES is not None:
        return True
    if _LERR is not None:
        return False
    import json as _j
    try:
        import onnxruntime as ort
    except ImportError as exc:
        _LERR = str(exc)
        return False
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pp = os.environ.get("POKER44_GEN9_PROFILE", os.path.join(_base, "models", "gen9_model20_profile.json"))
    _mp = os.environ.get("POKER44_GEN9_MODEL",   os.path.join(_base, "models", "gen9_model20.onnx"))
    try:
        with open(_pp) as f:
            _PROF = _j.load(f)
        _SES = ort.InferenceSession(_mp)
        return True
    except Exception as exc:
        _LERR = str(exc)
        return False


def _fvec(chunk: List[dict]) -> np.ndarray:
    g = _cfeats(chunk)
    _xc = _xcount(chunk)
    _hc = max(1, len(chunk))
    return np.array([
        (_xc["a"] / _hc) * _PS,
        (_xc["b"] / _hc) * _PS,
        (_xc["c"] / _hc) * _PS,
        min(g["chunk_size"] / 40.0, 2.0),
        g["action_entropy"],
        min(g["actions_per_hand_mean"] / 20.0, 2.0),
        g["fold_ratio"],
        g["raise_ratio"],
        g["call_ratio"],
        g["check_ratio"],
        min(g["aggression_ratio"] / 3.0, 2.0),
        g["showdown_rate"],
        min(g["pot_mean"] / 200.0, 2.0),
        min(g["raise_bb_mean"] / 10.0, 2.0),
        min(g["street_depth_mean"] / 4.0, 2.0),
        g["actor_entropy"],
        min(g["players_mean"] / 6.0, 2.0),
        g["street_entropy"],
        min(g["actions_per_hand_std"] / 10.0, 2.0),
        g["bet_ratio"],
    ], dtype=np.float32)


def score_chunk_gen9model20(chunk: List[dict]) -> Tuple[float, str]:
    if not _load_gen9model20():
        return 0.5, "gen9_load_error"
    if not chunk:
        return 0.5, "gen9_empty"
    try:
        raw = _SES.run(None, {"input": _fvec(chunk).reshape(1, -1)})[0]
        score = float(np.clip(raw.flat[0], 0.0, 1.0))
    except Exception:
        return 0.5, "gen9_predict_error"
    return round(score, 6), "gen9model20"


def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    _norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {"scorer": _norm, "active": _norm == "gen9model20", "ok": True, "error": None, "details": {}}
    if _norm == "gen9model20":
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _pp = Path(os.environ.get("POKER44_GEN9_PROFILE", os.path.join(_base, "models", "gen9_model20_profile.json")))
        _mp = Path(os.environ.get("POKER44_GEN9_MODEL",   os.path.join(_base, "models", "gen9_model20.onnx")))
        info["details"] = {"profile_path": str(_pp), "profile_exists": _pp.exists(), "model_path": str(_mp), "model_exists": _mp.exists()}
        ok = _load_gen9model20()
        info["ok"] = ok
        if not ok:
            info["error"] = _LERR
    return info
