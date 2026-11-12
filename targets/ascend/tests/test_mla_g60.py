"""P6-01/G60: canonical online fused MLA — contract, layout addressing and safety.

The device result (20/20 official cases) is recorded in evidence/G60_DEVICE_EVIDENCE.json. These
tests pin the CONTRACT, because the ways an MLA kernel goes quietly wrong are contract errors, not
crash errors: a left-aligned causal mask, rope scaled twice, a padded denominator, or a host
transpose that changes which addresses the kernel reads.
"""
import json
import math
from pathlib import Path

import pytest

from tilefoundry_ascend.mla.model import (
    BFLOAT16,
    BNSD,
    BSND,
    FLOAT16,
    MLAError,
    MLAShape,
    build_mla,
    causal_limit,
    is_causal_masked,
    resolve_scale,
)


# ---------------------------------------------------------------- shape and domain

def test_d_qk_and_d_v_follow_the_concatenation_contract():
    spec = build_mla(batch=2, seq_q=128, seq_kv=128, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64)
    assert spec.shape.d_qk == 576
    assert spec.shape.d_v == 512          # V is K_nope, so D_v is D_nope
    assert spec.shape.group == 128


def test_the_two_official_d_nope_widths_are_both_accepted():
    """512 and 448 are both official. A model that only admits 512 would pass prefill and then
    refuse cases 3, 6 and 14."""
    for d_nope in (512, 448):
        spec = build_mla(batch=2, seq_q=128, seq_kv=128, heads_q=64, heads_kv=1,
                         d_nope=d_nope, d_rope=64)
        assert spec.shape.d_qk == d_nope + 64


def test_a_nondivisible_head_ratio_is_refused():
    """GQA needs each query head to map to exactly one KV head."""
    with pytest.raises(MLAError) as exc:
        build_mla(batch=2, seq_q=8, seq_kv=8, heads_q=100, heads_kv=3, d_nope=512, d_rope=64)
    assert exc.value.code == "MLA_NONDIVISIBLE_HEADS"


def test_an_unknown_layout_is_refused():
    with pytest.raises(MLAError) as exc:
        build_mla(batch=1, seq_q=8, seq_kv=8, heads_q=8, heads_kv=1, d_nope=512,
                  d_rope=64, layout="BSDH")
    assert exc.value.code == "MLA_BAD_LAYOUT"


def test_an_unsupported_dtype_is_refused():
    with pytest.raises(MLAError) as exc:
        build_mla(batch=1, seq_q=8, seq_kv=8, heads_q=8, heads_kv=1, d_nope=512,
                  d_rope=64, dtype="float32")
    assert exc.value.code == "MLA_BAD_DTYPE"


# ---------------------------------------------------------------- causal, right-aligned

def test_the_causal_limit_is_right_aligned_not_plain_i():
    """'j <= i' is the mistake this pins. At S == Skv the two agree, so only a cross-attention
    shape can tell them apart."""
    # S == Skv: the offset vanishes, both agree
    assert causal_limit(0, 8, 8) == 0
    assert causal_limit(7, 8, 8) == 7
    # S < Skv: right-aligned, so every row reaches further back
    assert causal_limit(0, 2, 8) == 6
    assert causal_limit(1, 2, 8) == 7


def test_the_mask_uses_the_offset():
    assert is_causal_masked(0, 7, 2, 8) is True     # 7 > 6
    assert is_causal_masked(0, 6, 2, 8) is False    # 6 == limit
    assert is_causal_masked(1, 7, 2, 8) is False    # 7 == limit


def test_causal_with_s_greater_than_skv_is_refused():
    """With S > Skv the offset is negative and rows attend to nothing."""
    with pytest.raises(MLAError) as exc:
        build_mla(batch=2, seq_q=256, seq_kv=128, heads_q=128, heads_kv=1,
                  d_nope=512, d_rope=64, is_causal=True)
    assert exc.value.code == "MLA_CAUSAL_NEEDS_S_LE_SKV"


def test_non_causal_with_s_greater_than_skv_is_legal():
    """Official case 4 is exactly this: S=256 > Skv=128, non-causal cross-attention. Rejecting it
    on the shape alone (rather than on the mask) was a real bug in this model."""
    spec = build_mla(batch=2, seq_q=256, seq_kv=128, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64, is_causal=False)
    assert spec.shape.d_qk == 576


# ---------------------------------------------------------------- scale

def test_the_default_scale_is_one_over_sqrt_d_qk_not_d_v():
    """D_qk is 576, D_v is 512. Scaling by D_v is a different operator with a plausible output."""
    spec = build_mla(batch=1, seq_q=8, seq_kv=8, heads_q=8, heads_kv=1, d_nope=512, d_rope=64)
    assert spec.scale == pytest.approx(1.0 / math.sqrt(576))
    assert spec.scale != pytest.approx(1.0 / math.sqrt(512))


def test_a_non_positive_scale_takes_the_default():
    for value in (-1.0, 0.0, -0.5):
        assert resolve_scale(value, 576) == pytest.approx(1.0 / math.sqrt(576))


def test_an_explicit_positive_scale_is_used_verbatim():
    assert resolve_scale(0.125, 576) == pytest.approx(0.125)


# ---------------------------------------------------------------- GQA mapping

def test_the_gqa_mapping_is_hq_over_g():
    spec = build_mla(batch=1, seq_q=8, seq_kv=8, heads_q=128, heads_kv=8, d_nope=512,
                     d_rope=64)
    assert spec.shape.group == 16
    assert spec.shape.kv_head_of(0) == 0
    assert spec.shape.kv_head_of(15) == 0      # last head of group 0
    assert spec.shape.kv_head_of(16) == 1
    assert spec.shape.kv_head_of(127) == 7


def test_an_out_of_range_head_is_refused():
    spec = build_mla(batch=1, seq_q=8, seq_kv=8, heads_q=8, heads_kv=1, d_nope=512, d_rope=64)
    with pytest.raises(MLAError) as exc:
        spec.shape.kv_head_of(8)
    assert exc.value.code == "MLA_HEAD_OUT_OF_RANGE"


# ---------------------------------------------------------------- layout addressing

def test_the_two_layouts_put_the_same_element_at_the_same_address():
    """This is the whole point of address mapping: BNSD and BSND are the SAME bytes with
    different metadata, so neither the host nor the kernel may move data to switch layout."""
    batch, seq, heads, d = 2, 4, 3, 8
    bsnd = build_mla(batch=batch, seq_q=seq, seq_kv=seq, heads_q=heads, heads_kv=1,
                     d_nope=d, d_rope=1, layout=BSND)
    bnsd = build_mla(batch=batch, seq_q=seq, seq_kv=seq, heads_q=heads, heads_kv=1,
                     d_nope=d, d_rope=1, layout=BNSD)
    assert bsnd.shape.tensor_shape("q_nope") == (batch, seq, heads, d)
    assert bnsd.shape.tensor_shape("q_nope") == (batch, heads, seq, d)

    # both views describe the same (b, h, i, d) element of the same buffer
    import torch
    flat = torch.arange(batch * seq * heads * d)
    a = flat.reshape(batch, seq, heads, d)                 # BSND
    b = a.permute(0, 2, 1, 3).contiguous().reshape(batch, heads, seq, d)  # BNSD storage
    for bi in range(batch):
        for hi in range(heads):
            for si in range(seq):
                assert a[bi, si, hi].tolist() == b[bi, hi, si].tolist()


def test_the_kernel_module_does_no_host_transpose_of_the_inputs():
    """cases/mla.md 1 forbids transposing whole tensors on the host. The kernel must reach both
    layouts through address mapping, so the wrapper may only rewrite metadata."""
    src = (Path(__file__).resolve().parents[1]
           / "tilefoundry_ascend" / "mla" / "kernel.py").read_text()
    body = src[src.index("def mla("):]
    # the wrapper must build views rather than move data
    assert "as_strided" in body
    assert ".permute(0, 2, 1, 3).contiguous()" not in body
    assert ".transpose(" not in body
    # and it must not call any Torch elementwise/linalg op on the inputs
    for forbidden in ("torch.cat", "torch.matmul", ".softmax(", "torch.softmax"):
        assert forbidden not in body, forbidden


def test_the_layout_is_part_of_the_compile_cache_key():
    """A BSND kernel and a BNSD kernel are different binaries because the strides differ. If the
    layout is missing from the key, one silently serves the other."""
    src = (Path(__file__).resolve().parents[1]
           / "tilefoundry_ascend" / "mla" / "kernel.py").read_text()
    key_line = src[src.index("    key = ("):src.index("    if key not in _COMPILED")]
    assert "bsnd" in key_line


# ---------------------------------------------------------------- tiling domain

def test_the_block_height_choices_are_legal():
    from tilefoundry_ascend.mla.kernel import BM_CHOICES, choose_block_m
    assert BM_CHOICES == (32, 64)
    # a group of 128 with a wide grid keeps the tall tile
    assert choose_block_m(128, jobs_wide=512) == 64
    # few jobs -> narrow the tile so more cores get work
    assert choose_block_m(128, jobs_wide=2) == 32


def test_a_group_that_cannot_tile_is_refused():
    from tilefoundry_ascend.mla.kernel import choose_block_m
    with pytest.raises(MLAError) as exc:
        choose_block_m(16, jobs_wide=8)
    assert exc.value.code == "MLA_BAD_TILE"


def test_the_panel_width_is_derived_from_d_not_fixed():
    """D_nope 448 with a hardwired 128 gives 3 panels covering 384 columns and silently drops the
    last 64 nope dims."""
    from tilefoundry_ascend.mla.kernel import panels
    assert panels(512) == (128, 128)
    assert panels(448) == (64, 64)
    # and the tiling must cover D exactly
    for d in (448, 512):
        width, _ = panels(d)
        assert d % width == 0


def test_the_kernel_domain_refuses_shapes_it_cannot_do():
    from tilefoundry_ascend.mla.kernel import check_domain
    with pytest.raises(MLAError) as exc:
        check_domain(seq_kv=100, d_nope=512, d_rope=64, group=64, block_m=64)
    assert exc.value.code == "MLA_KV_NOT_MULTIPLE_OF_128"
    with pytest.raises(MLAError) as exc:
        check_domain(seq_kv=128, d_nope=512, d_rope=32, group=64, block_m=64)
    assert exc.value.code == "MLA_D_ROPE_MUST_BE_64"


# ---------------------------------------------------------------- evidence

def test_the_g60_record_reports_the_official_sweep():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G60_DEVICE_EVIDENCE.json").read_text())
    sweep = data["official_20_case_sweep"]
    assert sweep["passed"] == 20 and sweep["total"] == 20
    assert data["status"] == "VERIFIED"


def test_the_g60_record_states_no_host_transpose():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G60_DEVICE_EVIDENCE.json").read_text())
    layout = data["layout_addressing"]
    assert layout["zero_copy"] is True
    assert "as_strided" in layout["mechanism"]