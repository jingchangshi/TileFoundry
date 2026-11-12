"""The event-domain ISA contracts: declared, sourced, and cost-silent.

Step 1 of the event slice. These rows make completion signalling a stated part
of the target contract table; they do NOT emit anything and do not claim that
TileLang or TF can execute them yet (that is step 2 and step 3).
"""
import pytest

from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
from tilefoundry_ascend.ir.isa import (
    EVENT_CONTRACT_NAMES,
    ISA,
    contract,
    event_contract_of,
)
from tilefoundry_ascend.plan.delays import declared_occupancy
from tilefoundry_ascend.sync import AivDrain, AivFlagSet, AivFlagWait


def test_event_rows_are_registered_and_name_the_event_namespace():
    assert set(EVENT_CONTRACT_NAMES) <= set(ISA)
    for name in EVENT_CONTRACT_NAMES:
        row = contract(name)
        assert row.category == "sync", name
        assert row.resource.startswith("event_"), (name, row.resource)
        assert row.operands == (), (name, row.operands)
        assert row.result == "event", name
        assert row.required_region == "outside", name
        assert row.core_affinity == "aiv", name


def test_event_contracts_state_no_cost_model_and_refuse_instead_of_defaulting():
    """A None cost is a refusal, not a silent tick: nothing here may fabricate 1."""
    for name in EVENT_CONTRACT_NAMES:
        row = contract(name)
        assert row.latency_model is None and row.ii_model is None, name
        assert declared_occupancy(name).state == "UNSOURCED", name
        with pytest.raises(Exception) as caught:
            declared_occupancy(name).resolve()
        assert caught.value.code == "DELAY_UNSOURCED"


def test_real_sync_ops_resolve_to_their_event_contract():
    attrs = dict(channel_name="ub:0", kind="ready", slot=0, generation=0,
                 versions=1, flag_id=0, producer_pipe="mte2", consumer_pipe="vector")
    for op, name in ((FlagSet(**attrs), "flag_set"),
                     (FlagWait(**attrs), "flag_wait"),
                     (Drain(), "drain"),
                     (AivFlagSet(**attrs), "flag_set"),
                     (AivFlagWait(**attrs), "flag_wait"),
                     (AivDrain(), "drain")):
        assert event_contract_of(op).name == name
        assert event_contract_of(type(op)).name == name


def test_a_data_instruction_is_not_an_event_contract():
    from tilefoundry_ascend.ir.ops import Vadd

    with pytest.raises(KeyError):
        event_contract_of(Vadd())


def test_the_data_side_helper_still_refuses_event_ops_by_name():
    """isa_contract_of keys on ascend_*; the event ops must NOT leak into it."""
    from tilefoundry_ascend.plan.units import isa_contract_of
    from tilefoundry.ir.tir.flag import FlagSet

    attrs = dict(channel_name="c", kind="ready", slot=0, generation=0, versions=1,
                 flag_id=0, producer_pipe="mte2", consumer_pipe="vector")
    with pytest.raises(KeyError):
        isa_contract_of(FlagSet(**attrs))


def test_pipe_barrier_and_threadfence_are_declared_without_a_tf_op():
    """Honest state: the contract exists, the TF instruction in this tree does not."""
    import tilefoundry.ir.tir.flag as F
    import tilefoundry.ir.tir.sync as S

    names = {n for n in dir(F) if n[0].isupper()} | {n for n in dir(S) if n[0].isupper()}
    assert not {"PipeBarrier", "ThreadFence"} & names
    for name in ("pipe_barrier", "threadfence"):
        assert "TF op MISSING" in contract(name).compute_domain


def test_flag_ops_carry_their_facts_as_attributes_only():
    attrs = dict(channel_name="c", kind="free", slot=1, generation=2, versions=3,
                 flag_id=4, producer_pipe="vector", consumer_pipe="mte3")
    inst = FlagSet(**attrs)
    kinds = {param.name: param.kind for param in type(inst).params()}
    assert set(kinds) == set(attrs) and set(kinds.values()) == {"attribute"}
    assert all(getattr(inst, key) == value for key, value in attrs.items())
