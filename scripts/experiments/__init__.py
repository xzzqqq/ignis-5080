"""Single-host GPU experiments; independent of candidate imports and builds."""

PROTOCOL = 1
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out", "interrupted"})
