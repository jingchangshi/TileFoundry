"""X3 tests: H4 observed identities -> X claim chain.

The H4 root result has source/CCE/load OBSERVED and binary UNOBSERVED_BY_BUILD
(this TL build exposes no JITKernel.artifact). CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))

from tilefoundry_ascend.codegen import x_runtime as X


H4_IDENTITIES = {
    "source": {"status": "OBSERVED",
               "sha256": "671e68fbd821949fe2e0512693ebb64fae379cd79e8e96efa1b141cc54f92b9e",
               "path": "scripts/tl_vadd_one.py", "reason": None},
    "cce": {"status": "OBSERVED",
            "sha256": "7b3ac19981a85423defee5545f7ae35acbf9a4d2b2969e2e8754648a3ba2c773",
            "reason": "TileLang JITKernel.kernel_source (Ascend C source)"},
    "binary": {"status": "UNOBSERVED",
               "reason": "JITKernel.artifact is None; no binary path exposed by this TL build"},
    "load": {"status": "OBSERVED",
             "sha256": "7e0ccdd3863fcb7afe857e8051f1d6c6a8379a66c198c40bc010a9701a5fae09",
             "reason": "compiled JITKernel callable in this process"},
}


def expect(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, nothing refused" % code)


def test_x3_h4_chain_names_binary_unobserved_by_build():
    result = X.validate_h4_identity_chain(H4_IDENTITIES, claim_pass=True)
    assert result["status"] == "IDENTITY_CHAIN_READY"
    assert result["binary_status"] == X.BINARY_UNOBSERVED_BY_BUILD
    assert result["identities"]["source"]["sha256"]
    assert result["identities"]["cce"]["sha256"]
    assert result["identities"]["load"]["sha256"]
    extracted = X.identities_from_h4_result({"identities": H4_IDENTITIES})
    assert extracted["source"]["status"] == "OBSERVED"


def test_x3_fake_binary_observed_is_refused():
    fake = dict(H4_IDENTITIES)
    fake["binary"] = {"status": "OBSERVED", "sha256": "f" * 64}
    expect("X_BINARY_IDENTITY_UNEXPECTED",
           lambda: X.validate_h4_identity_chain(fake, claim_pass=True))


def test_x3_generic_unobserved_is_not_a_pass():
    fake = dict(H4_IDENTITIES)
    fake["binary"] = {"status": "UNOBSERVED", "reason": ""}
    expect("X_BINARY_IDENTITY_UNNAMED",
           lambda: X.validate_h4_identity_chain(fake, claim_pass=True))


def test_x3_missing_required_identity_is_refused():
    fake = dict(H4_IDENTITIES)
    fake["source"] = {"status": "UNOBSERVED"}
    expect("X_ARTIFACT_IDENTITY_MISSING",
           lambda: X.validate_h4_identity_chain(fake, claim_pass=True))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
