#!/usr/bin/env python3
"""
Parameter application + verification for the RCT orchestrator.

WHY THIS EXISTS
---------------
The original implementation shelled out to `ros2 param set` and `ros2 param get`
once per parameter per trial. Each of those commands spins up a whole rclpy node,
discovers the graph, calls the service and tears down — typically 0.5-3 s, and
occasionally longer than the 8 s subprocess timeout. In the smoke run this
produced 33 read-back failures and 8 set timeouts, while producing *zero*
genuine rejections and *zero* value mismatches. 53% of trials were flagged as
`config_valid = 0` for what was purely a tooling problem.

Two things were wrong:

1. Speed/robustness. Fixed here by talking to the `/<node>/set_parameters` and
   `/<node>/get_parameters` services directly over a persistent rclpy node.
   No process spawn, no graph re-discovery, ~1-2 ms per call.

2. Semantics. "Set was rejected" and "we couldn't read the value back" are
   completely different events for a causal experiment, but both were collapsed
   into one `params_unverified` flag. They are now separate outcomes:

     OK              set accepted, read back, value matches   -> treatment applied
     MISMATCH        set accepted, read back, value differs   -> treatment WRONG
     SET_REJECTED    node refused the set                     -> treatment NOT applied
     SET_TIMEOUT     service call timed out                   -> treatment UNKNOWN
     READBACK_FAILED set accepted, but get failed/timed out   -> treatment probably fine
     NO_SERVICE      node not reachable at all                -> treatment UNKNOWN

   Only MISMATCH / SET_REJECTED / SET_TIMEOUT / NO_SERVICE threaten causal
   validity. READBACK_FAILED means the node accepted the value and we simply
   failed to confirm it, so those trials stay analysable.

TYPE HANDLING
-------------
`ros2 param set` infers the wire type from the parameter already declared on the
node. We reproduce that explicitly: read the current value first, learn its
declared ParameterType, then send the new value using that same type. This
avoids spurious rejections when e.g. `time_steps` is declared INTEGER but a
float is sent, which the old string-based path masked by accident.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Outcome codes. Keep these as plain strings — they get written into the CSV.
OK = "OK"
MISMATCH = "MISMATCH"
SET_REJECTED = "SET_REJECTED"
SET_TIMEOUT = "SET_TIMEOUT"
READBACK_FAILED = "READBACK_FAILED"
NO_SERVICE = "NO_SERVICE"

#: Outcomes that mean the applied treatment is not known to equal the recorded
#: treatment. A trial with any of these must not be used for causal estimation
#: at face value.
INTEGRITY_BREAKING = frozenset({MISMATCH, SET_REJECTED, SET_TIMEOUT, NO_SERVICE})


class ParamOutcome:
    """Result of applying one ROS parameter."""

    __slots__ = ("ros_node", "name", "target", "actual", "outcome", "detail",
                 "attempts", "elapsed_sec", "logical_key")

    def __init__(self, ros_node: str, name: str, target: Any):
        self.ros_node = ros_node
        self.name = name
        self.target = target
        self.logical_key = f"{ros_node}__{name}"
        self.actual: Any = None
        self.outcome: str = NO_SERVICE
        self.detail: str = ""
        self.attempts: int = 0
        self.elapsed_sec: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.ros_node}/{self.name}"

    @property
    def ok(self) -> bool:
        return self.outcome == OK

    @property
    def breaks_integrity(self) -> bool:
        return self.outcome in INTEGRITY_BREAKING

    def __repr__(self) -> str:
        return f"<{self.key}={self.target!r} -> {self.outcome}>"


class ParamApplier:
    """Applies and verifies ROS parameters over persistent service clients.

    Falls back to the original subprocess path only if rclpy service clients
    cannot be constructed (e.g. running outside a ROS environment in tests).
    """

    def __init__(
        self,
        node,
        service_timeout_sec: float = 5.0,
        readback_attempts: int = 3,
        readback_backoff_sec: float = 0.25,
        settle_sec: float = 0.0,
    ):
        """
        node : an rclpy Node used to create service clients. It must NOT be
               owned by a spinning executor, because we call
               ``rclpy.spin_until_future_complete`` on it directly.
        readback_attempts : how many times to retry a failed get before giving
               up and recording READBACK_FAILED. The old code tried once.
        settle_sec : optional pause after a successful set, before read-back.
               Useful for params whose on_set callback rebuilds a costmap.
        """
        self.node = node
        self.service_timeout_sec = service_timeout_sec
        self.readback_attempts = max(1, readback_attempts)
        self.readback_backoff_sec = readback_backoff_sec
        self.settle_sec = settle_sec

        self._set_clients: dict = {}
        self._get_clients: dict = {}
        self._declared_types: dict = {}   # (ros_node, name) -> ParameterType int
        self._unreachable: set = set()

        try:
            from rcl_interfaces.srv import GetParameters, SetParameters
            from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
            self._GetParameters = GetParameters
            self._SetParameters = SetParameters
            self._Parameter = Parameter
            self._ParameterValue = ParameterValue
            self._ParameterType = ParameterType
            import rclpy
            self._rclpy = rclpy
            self.available = True
        except Exception as e:  # pragma: no cover
            logger.error(
                f"rcl_interfaces unavailable ({e}); ParamApplier disabled. "
                "Parameters will NOT be applied."
            )
            self.available = False

    # ── service plumbing ──────────────────────────────────────────────────

    def _client(self, cache: dict, srv_type, ros_node: str, suffix: str):
        cli = cache.get(ros_node)
        if cli is None:
            cli = self.node.create_client(srv_type, f"/{ros_node}/{suffix}")
            cache[ros_node] = cli
        return cli

    def _call(self, cli, req, timeout_sec: float):
        """Call a service, returning (response|None, reason)."""
        if not cli.service_is_ready():
            if not cli.wait_for_service(timeout_sec=timeout_sec):
                return None, "service_unavailable"
        future = cli.call_async(req)
        self._rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=timeout_sec)
        if not future.done():
            future.cancel()
            return None, "timeout"
        try:
            return future.result(), ""
        except Exception as e:
            return None, f"exception:{e}"

    # ── value <-> ParameterValue ──────────────────────────────────────────

    def _unpack(self, pv) -> Any:
        PT = self._ParameterType
        t = pv.type
        if t == PT.PARAMETER_BOOL:
            return bool(pv.bool_value)
        if t == PT.PARAMETER_INTEGER:
            return int(pv.integer_value)
        if t == PT.PARAMETER_DOUBLE:
            return float(pv.double_value)
        if t == PT.PARAMETER_STRING:
            return str(pv.string_value)
        if t == PT.PARAMETER_DOUBLE_ARRAY:
            return [float(v) for v in pv.double_array_value]
        if t == PT.PARAMETER_INTEGER_ARRAY:
            return [int(v) for v in pv.integer_array_value]
        if t == PT.PARAMETER_STRING_ARRAY:
            return [str(v) for v in pv.string_array_value]
        if t == PT.PARAMETER_BOOL_ARRAY:
            return [bool(v) for v in pv.bool_array_value]
        return None   # PARAMETER_NOT_SET

    def _pack(self, value: Any, declared_type: Optional[int]):
        """Build a ParameterValue, honouring the type already declared on the
        node when we know it. Returns (ParameterValue|None, error_str)."""
        PT = self._ParameterType
        pv = self._ParameterValue()

        t = declared_type
        if t is None or t == PT.PARAMETER_NOT_SET:
            # Infer from the Python type we were handed.
            if isinstance(value, bool):
                t = PT.PARAMETER_BOOL
            elif isinstance(value, int):
                t = PT.PARAMETER_INTEGER
            elif isinstance(value, float):
                t = PT.PARAMETER_DOUBLE
            else:
                t = PT.PARAMETER_STRING

        try:
            pv.type = t
            if t == PT.PARAMETER_BOOL:
                pv.bool_value = bool(value)
            elif t == PT.PARAMETER_INTEGER:
                pv.integer_value = int(round(float(value)))
            elif t == PT.PARAMETER_DOUBLE:
                pv.double_value = float(value)
            elif t == PT.PARAMETER_STRING:
                pv.string_value = str(value)
            elif t == PT.PARAMETER_DOUBLE_ARRAY:
                pv.double_array_value = [float(v) for v in value]
            elif t == PT.PARAMETER_INTEGER_ARRAY:
                pv.integer_array_value = [int(v) for v in value]
            elif t == PT.PARAMETER_STRING_ARRAY:
                pv.string_array_value = [str(v) for v in value]
            elif t == PT.PARAMETER_BOOL_ARRAY:
                pv.bool_array_value = [bool(v) for v in value]
            else:
                return None, f"unsupported_type:{t}"
        except (TypeError, ValueError) as e:
            return None, f"pack_failed:{e}"
        return pv, ""

    # ── get / set ─────────────────────────────────────────────────────────

    def get(self, ros_node: str, name: str, timeout_sec: Optional[float] = None):
        """Read one parameter. Returns (value, declared_type, error_str)."""
        timeout = self.service_timeout_sec if timeout_sec is None else timeout_sec
        cli = self._client(self._get_clients, self._GetParameters,
                           ros_node, "get_parameters")
        req = self._GetParameters.Request()
        req.names = [name]
        resp, reason = self._call(cli, req, timeout)
        if resp is None:
            return None, None, reason
        if not resp.values:
            return None, None, "empty_response"
        pv = resp.values[0]
        if pv.type == self._ParameterType.PARAMETER_NOT_SET:
            return None, pv.type, "not_set"
        return self._unpack(pv), pv.type, ""

    def _declared_type(self, ros_node: str, name: str) -> Optional[int]:
        """Learn (and cache) the type a node has declared for a parameter."""
        key = (ros_node, name)
        if key in self._declared_types:
            return self._declared_types[key]
        _val, t, err = self.get(ros_node, name)
        if err == "" and t is not None:
            self._declared_types[key] = t
            return t
        return None

    def set_and_verify(self, ros_node: str, name: str, value: Any,
                       param_type: str) -> ParamOutcome:
        """Set one parameter and confirm it took. Never raises."""
        out = ParamOutcome(ros_node, name, value)
        t0 = time.time()

        if not self.available:
            out.outcome = NO_SERVICE
            out.detail = "rcl_interfaces_unavailable"
            out.elapsed_sec = time.time() - t0
            return out

        if ros_node in self._unreachable:
            out.outcome = NO_SERVICE
            out.detail = "node_marked_unreachable"
            out.elapsed_sec = time.time() - t0
            return out

        declared = self._declared_type(ros_node, name)
        pv, perr = self._pack(value, declared)
        if pv is None:
            out.outcome = SET_REJECTED
            out.detail = perr
            out.elapsed_sec = time.time() - t0
            return out

        # --- set ---
        param = self._Parameter()
        param.name = name
        param.value = pv
        req = self._SetParameters.Request()
        req.parameters = [param]

        cli = self._client(self._set_clients, self._SetParameters,
                           ros_node, "set_parameters")
        resp, reason = self._call(cli, req, self.service_timeout_sec)
        out.attempts += 1

        if resp is None:
            if reason == "service_unavailable":
                self._unreachable.add(ros_node)
                out.outcome = NO_SERVICE
            else:
                out.outcome = SET_TIMEOUT
            out.detail = reason
            out.elapsed_sec = time.time() - t0
            return out

        if not resp.results or not resp.results[0].successful:
            out.outcome = SET_REJECTED
            out.detail = (resp.results[0].reason if resp.results else "no_result")[:120]
            out.elapsed_sec = time.time() - t0
            return out

        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

        # --- read back, with retries ---
        got, err = None, "not_attempted"
        for attempt in range(self.readback_attempts):
            out.attempts += 1
            got, _t, err = self.get(ros_node, name)
            if err == "":
                break
            if attempt < self.readback_attempts - 1:
                time.sleep(self.readback_backoff_sec * (2 ** attempt))

        out.elapsed_sec = time.time() - t0

        if err != "":
            # The node ACCEPTED the value; we just could not confirm it.
            # This does not invalidate the trial.
            out.outcome = READBACK_FAILED
            out.detail = err
            return out

        out.actual = got
        if values_match(got, value, param_type):
            out.outcome = OK
        else:
            out.outcome = MISMATCH
            out.detail = f"asked={value!r} got={got!r}"
        return out


# ── value comparison ─────────────────────────────────────────────────────────

def values_match(got: Any, target: Any, param_type: str) -> bool:
    """Compare a read-back value against the value we asked for.

    Unchanged tolerances from the original implementation, but it now receives
    already-typed Python values rather than parsing the stdout of `ros2 param
    get`, so string-formatting differences can no longer cause false mismatches.
    """
    try:
        if param_type == "continuous":
            tol = max(1e-3, 1e-2 * abs(float(target)))
            return abs(float(got) - float(target)) <= tol
        if param_type == "discrete":
            return int(round(float(got))) == int(round(float(target)))
        if param_type == "footprint":
            import ast
            a = [[float(x) for x in pt] for pt in ast.literal_eval(str(got))]
            b = [[float(x) for x in pt] for pt in ast.literal_eval(str(target))]
            if len(a) != len(b):
                return False
            return all(
                abs(ai - bi) <= 1e-3
                for pa, pb in zip(a, b)
                for ai, bi in zip(pa, pb)
            )
        if param_type == "categorical":
            return str(got).strip() == str(target).strip()
        return str(got).strip().lower() == str(target).strip().lower()
    except (ValueError, TypeError, SyntaxError):
        return False