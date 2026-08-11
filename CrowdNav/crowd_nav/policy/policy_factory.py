# -*- coding: utf-8 -*-
"""
policy_factory.py — robust registry for IL + Mamba + (SARL/CADRL/ORCA)

Why this rewrite
- Solve KeyError('mamba'): No longer rely on external HAS_MAMBA switch; dynamically discover policies based on "if importable, then register" principle.
- Compatible with multiple class names/file names: Mamba class in your repo might be called MambaRL_SARL / MambaRL / MambaPolicy; try all of them here.
- Register unified aliases: 'mamba', 'mamba_rl', 'mamba_vl' etc. all map to the same implementation, avoiding command line/config differences.
- Give **readable reasons** for failures (import errors will be written to logging.warning), instead of silent failures.

Drop‑in: Replace crowd_nav/policy/policy_factory.py
"""
from __future__ import annotations
import logging

# -----------------------------
# Safe import helpers
# -----------------------------

def _try_import_mamba():
    mod = None; cls = None; err = None
    # Relative import, ensure availability within package
    try:
        from . import mamba_rl as mod  # type: ignore
    except Exception as e:
        err = e
        return None, None, err
    # Try multiple class names
    for name in ("MambaRL_SARL", "MambaRL", "MambaPolicy", "MambaVLearningPolicy", "MambaVPolicy"):
        if hasattr(mod, name):
            cls = getattr(mod, name)
            break
    if cls is None:
        err = RuntimeError("mamba_rl.py exists, but no usable class name found (expected: MambaRL_SARL / MambaRL / MambaPolicy / MambaVLearningPolicy / MambaVPolicy)")
    return mod, cls, err


def _try_import_bayesian_fullcrowd_risk_value():
    try:
        from .bayesian_fullcrowd_risk_value import (
            BayesianFullCrowdRiskValuePolicy,
        )
        return BayesianFullCrowdRiskValuePolicy, None
    except Exception as exc:
        return None, exc


def _try_import_cv_fullcrowd_risk_value():
    try:
        from .cv_fullcrowd_risk_value import (
            ConstantVelocityFullCrowdRiskValuePolicy,
        )
        return ConstantVelocityFullCrowdRiskValuePolicy, None
    except Exception as exc:
        return None, exc


def _try_import_bayesian_model_average_risk_value():
    try:
        from .bayesian_model_average_risk_value import (
            BayesianModelAverageFullCrowdRiskValuePolicy,
            FixedModelAverageFullCrowdRiskValuePolicy,
        )
        return (
            BayesianModelAverageFullCrowdRiskValuePolicy,
            FixedModelAverageFullCrowdRiskValuePolicy,
            None,
        )
    except Exception as exc:
        return None, None, exc


def _try_import_sarl():
    try:
        from .sarl import SARL  # type: ignore
        return SARL, None
    except Exception as e:
        return None, e


def _try_import_cadrl():
    try:
        from .cadrl import CADRL  # type: ignore
        return CADRL, None
    except Exception as e:
        return None, e


def _try_import_orca():
    # ORCA is in crowd_sim.envs.policy, not in local policy directory
    try:
        from crowd_sim.envs.policy.orca import ORCA  # type: ignore
        return ORCA, None
    except Exception as e1:
        # Fallback: try old locations
        try:
            from .orca_wrapper import ORCA  # type: ignore
            return ORCA, None
        except Exception as e2:
            try:
                from .orca_wrapper import ORCAWrapper as ORCA3  # type: ignore
                return ORCA3, None
            except Exception as e3:
                return None, (e1, e2, e3)


def _try_import_dsrnn():
    try:
        from .dsrnn_policy import DSRNNPolicy
        return DSRNNPolicy, None
    except Exception as e:
        return None, e


def _try_import_lstm():
    try:
        from .lstm_rl import LstmRL
        return LstmRL, None
    except Exception as e:
        return None, e


def _try_import_bayesian_brne():
    try:
        from crowd_nav.bayesian_brne.policy import BayesianBRNEPolicy
        return BayesianBRNEPolicy, None
    except Exception as e:
        return None, e


def _try_import_causal_bayesian_brne():
    try:
        from crowd_nav.bayesian_brne.causal_runtime import CausalBayesianBRNEPolicy
        return CausalBayesianBRNEPolicy, None
    except Exception as e:
        return None, e


def _try_import_bayesian_dvl():
    try:
        from crowd_nav.bayesian_dvl.policy import BayesianDVLPolicy
        return BayesianDVLPolicy, None
    except Exception as e:
        return None, e


def _try_import_intent_bdvl():
    try:
        from crowd_nav.bayesian_dvl.intent_crowdnav_policy import IntentBDVLPolicy
        return IntentBDVLPolicy, None
    except Exception as e:
        return None, e


# -----------------------------
# Build factory
# -----------------------------
policy_factory = {}

# Mamba (key)
_mamba_mod, _MambaClass, _mamba_err = _try_import_mamba()
if _MambaClass is not None:
    policy_factory.update({
        'mamba': _MambaClass,
        'mamba_rl': _MambaClass,
        'mamba_vl': _MambaClass,
        'MambaRL_SARL': _MambaClass,  # Compatible with old key
    })
else:
    logging.warning("[policy_factory] Mamba policy not registered: %r", _mamba_err)

_BayesianFullCrowdRiskValue, _bayesian_fullcrowd_risk_err = (
    _try_import_bayesian_fullcrowd_risk_value()
)
if _BayesianFullCrowdRiskValue is not None:
    policy_factory.update({
        'bayesian_fullcrowd_risk_value': _BayesianFullCrowdRiskValue,
        'bayes_fullcrowd_value': _BayesianFullCrowdRiskValue,
        # Keep the evaluation alias used by the archived quick-gate logs.
        'bayesian_counterfactual_value': _BayesianFullCrowdRiskValue,
    })
else:
    logging.warning(
        "[policy_factory] Bayesian full-crowd risk policy not registered: %r",
        _bayesian_fullcrowd_risk_err,
    )

_CVFullCrowdRiskValue, _cv_fullcrowd_risk_err = (
    _try_import_cv_fullcrowd_risk_value()
)
if _CVFullCrowdRiskValue is not None:
    policy_factory.update({
        'cv_fullcrowd_risk_value': _CVFullCrowdRiskValue,
    })
else:
    logging.warning(
        "[policy_factory] CV full-crowd control not registered: %r",
        _cv_fullcrowd_risk_err,
    )

(
    _BayesianModelAverageFullCrowdRiskValue,
    _FixedModelAverageFullCrowdRiskValue,
    _model_average_risk_err,
) = _try_import_bayesian_model_average_risk_value()
if _BayesianModelAverageFullCrowdRiskValue is not None:
    policy_factory.update({
        'bayesian_model_average_risk_value':
            _BayesianModelAverageFullCrowdRiskValue,
        'fixed_model_average_risk_value':
            _FixedModelAverageFullCrowdRiskValue,
    })
else:
    logging.warning(
        "[policy_factory] Bayesian model-average policies not registered: %r",
        _model_average_risk_err,
    )

# SARL
_SARL, _sarl_err = _try_import_sarl()
if _SARL is not None:
    policy_factory.update({'sarl': _SARL, 'SARL': _SARL})
else:
    logging.warning("[policy_factory] SARL not registered: %r", _sarl_err)

# CADRL
_CADRL, _cadrl_err = _try_import_cadrl()
if _CADRL is not None:
    policy_factory.update({'cadrl': _CADRL, 'CADRL': _CADRL})
else:
    logging.warning("[policy_factory] CADRL not registered: %r", _cadrl_err)

# ORCA (for IL/comparison)
_ORCA, _orca_err = _try_import_orca()
if _ORCA is not None:
    policy_factory.update({'orca': _ORCA, 'ORCA': _ORCA})
else:
    logging.warning("[policy_factory] ORCA not registered: %r", _orca_err)

# DSRNN (baseline)
_DSRNN, _dsrnn_err = _try_import_dsrnn()
if _DSRNN is not None:
    policy_factory.update({'dsrnn': _DSRNN, 'DSRNN': _DSRNN})
else:
    logging.warning("[policy_factory] DSRNN not registered: %r", _dsrnn_err)

# LSTM
_LSTM, _lstm_err = _try_import_lstm()
if _LSTM is not None:
    policy_factory.update({'lstm': _LSTM, 'LSTM': _LSTM, 'lstm_rl': _LSTM})
else:
    logging.warning("[policy_factory] LSTM not registered: %r", _lstm_err)

# SM-BRNE (Order F6, 2026-08-03): BayesianBRNEPolicy.__init__ takes no
# arguments (matches every other entry here) -- configure(config, artifact)
# needs a SECOND positional arg (the fitted ARHMMArtifact) that this
# generic factory dict has no way to supply, so callers going through this
# factory must call policy.configure(policy_config, artifact) themselves
# after construction, exactly like every other policy's configure(config)
# call, just with one extra required argument. configs/policy_bayesian_brne.
# config already declares `[policy] key = bayesian_brne`, matching this key.
_BayesianBRNE, _bayesian_brne_err = _try_import_bayesian_brne()
if _BayesianBRNE is not None:
    policy_factory.update({'bayesian_brne': _BayesianBRNE})
else:
    logging.warning("[policy_factory] BayesianBRNE not registered: %r", _bayesian_brne_err)

# CR-S1 pure Bayesian planner. The formal evaluator configures it with the
# promoted CausalResponseArtifact; no neural checkpoint is accepted.
_CausalBayesianBRNE, _causal_bayesian_brne_err = _try_import_causal_bayesian_brne()
if _CausalBayesianBRNE is not None:
    policy_factory.update({'causal_bayesian_brne': _CausalBayesianBRNE})
else:
    logging.warning(
        "[policy_factory] CausalBayesianBRNE not registered: %r",
        _causal_bayesian_brne_err,
    )

# BDVL (Bayesian Distributional Value Lookahead, guide.md current spec).
# Zero-arg constructible like every other policy; real components
# (SBK-HMM artifact, SetEncoder/IQN weights) are loaded lazily inside
# configure(config) from that config's [bayesian_dvl] section.
_BayesianDVL, _bayesian_dvl_err = _try_import_bayesian_dvl()
if _BayesianDVL is not None:
    policy_factory.update({'bayesian_dvl': _BayesianDVL})
else:
    logging.warning("[policy_factory] BayesianDVL not registered: %r", _bayesian_dvl_err)

# Goal-intent BDVL (V6 main chain): public-candidate Bayesian goal
# posterior -> Q(s,b,a) -> one of the frozen 80 actions. Zero-arg
# constructible; the V6 checkpoint (normally final_ema.pth) is loaded
# lazily by configure(config) from that config's [intent_bdvl] section.
# Registered under several aliases so Test5/Gazebo command lines can use
# whichever name they already expect.
_IntentBDVL, _intent_bdvl_err = _try_import_intent_bdvl()
if _IntentBDVL is not None:
    policy_factory.update({
        'intent_bdvl': _IntentBDVL,
        'bdvl_intent': _IntentBDVL,
        'goal_intent_bdvl': _IntentBDVL,
    })
else:
    logging.warning("[policy_factory] IntentBDVL not registered: %r", _intent_bdvl_err)


# -----------------------------
# Helpers (optional)
# -----------------------------

def list_available_policies():
    return sorted(policy_factory.keys())


__all__ = ["policy_factory", "list_available_policies"]
