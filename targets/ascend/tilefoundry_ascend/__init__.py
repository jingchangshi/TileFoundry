"""TileFoundry Ascend950PR backend (Scheduled HIR / TIR analysis; contract + planning + reference eval).

The contract, planning and reference-evaluation parts of this bundle are
self-contained so they can be exercised with any Python that has numpy and z3.
Declaring a Target is the one part that reaches the core, because a Target is a
core value; importing the package is what publishes this device's execution
surface and its Target.
"""

from tilefoundry_ascend import ir as _ir  # noqa: F401  (registers the surface)
from tilefoundry_ascend.tf_target import Ascend950PRTarget  # noqa: F401

# Importing the pass module is what REGISTERS the D0 entries against
# Ascend950PRTarget. Registration is a decorator side effect, so without this import
# the checkers exist but target_passes() never returns them and the public finalize
# path would silently skip every D0 verification. Imported AFTER tf_target, because
# the pass module imports the Target class from it.
from tilefoundry_ascend.passes import pipeline as _pipeline  # noqa: F401

# P3-04/G33: the P3 chain (ledger -> problem -> solve -> resolve -> layout ->
# materialize -> pipeline) and the P4 export gate are registered the same way. Same
# reason: without this import target_passes() never returns them, so the public
# finalize() path would lower a program with NO analysis, NO solve and NO plan behind
# it, and could not tell that its final artifact is not runnable.
from tilefoundry_ascend.passes import p3_chain as _p3_chain  # noqa: F401

__all__ = ["Ascend950PRTarget"]
