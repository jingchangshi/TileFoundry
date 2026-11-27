"""Pre-lowering TIR anchor specialization and positive-denominator safety solver."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

_MAX_SAFE_SPEC_INT = 1 << 22
_UNBOUND_SCALAR_EXTENT_DEFAULT = 16
_UNBOUND_SCALAR_NAME_TOKENS = (
    "block", "tile", "split", "chunk", "step", "iter", "topk", "top_k", "sparse", "group", "stage",
)
_BUFFER_FREE_AXIS_CAP = 65536
_NUMEL_ALIASES = ("total_elements", "total_elems", "n_elems", "num_elements", "numel")
_DIM_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("b", "b_dim", "batch", "batch_size"), ("m", "m_dim", "m_rows", "q_total_rows", "num_tokens", "tokens"),
    ("n", "n_dim", "n_cols"), ("k", "k_dim"), ("h", "h_dim", "heads", "num_heads", "n_q", "n_heads"),
    ("l", "l_dim", "seq_len", "s", "s_dim"), ("c", "c_dim", "channels", "in_channels", "out_channels"),
    ("v", "v_dim", "vocab", "vocab_size"), ("g", "g_dim", "groups", "num_groups"),
    ("d", "d_dim", "head_dim", "d_nope"), ("s_kv", "kv_len", "s_kv_dim"), _NUMEL_ALIASES,
)


def same_tirx_var(lhs: object, rhs: object) -> bool:
    """Return True when two TIR nodes refer to the same variable object."""
    return (hasattr(lhs, "same_as") and bool(lhs.same_as(rhs))) or lhs is rhs


def static_tirx_int(expr: object) -> int | None:
    """Extract a concrete integer from an int or IntImm-like TIR expression."""
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    val = getattr(expr, "value", None)
    return int(val) if isinstance(val, int) and not isinstance(val, bool) else None


def static_shape_ints(shape: object) -> tuple[int, ...] | None:
    """Extract a concrete integer tuple from a buffer shape, or None if symbolic."""
    dims: list[int] = []
    for dim in shape:
        if (sv := static_tirx_int(dim)) is None:
            return None
        dims.append(sv)
    return tuple(dims)


def any_tirx_node(expr: object, tirx: object, pred: Callable[[object], bool]) -> bool:
    """Return True if any AST node inside *expr* satisfies *pred*."""
    found = False

    def _visit(node: object) -> None:
        nonlocal found
        if not found and pred(node):
            found = True

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found


def add_unique_expr(target: list[object], expr: object) -> None:
    """Append *expr* to *target* if not structurally identical to an existing item."""
    if expr is not None and not any(same_tirx_var(expr, prev) for prev in target):
        target.append(expr)


def free_tirx_vars(expr: object, *, tirx: object) -> list[object]:
    """Return ordered unique free TIR variables referenced inside *expr*."""
    if hasattr(tirx, "analysis") and hasattr(tirx.analysis, "undefined_vars"):
        try:
            return list(tirx.analysis.undefined_vars(expr))
        except Exception:
            pass
    found: list[object] = []
    seen_ids: set[int] = set()

    def _visit(node: object) -> None:
        if isinstance(node, tirx.Var) and id(node) not in seen_ids:
            found.append(node)
            seen_ids.add(id(node))

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found


def substitute_let_env(expr: object, let_env: Mapping[object, object], *, tirx: object) -> object:
    """Iteratively expand let-bound TIR variables inside *expr*."""
    cur_expr = expr
    for _ in range(16):
        if not let_env:
            break
        expand = {v: let_env[v] for v in free_tirx_vars(cur_expr, tirx=tirx) if v in let_env}
        if not expand:
            break
        nxt = tirx.stmt_functor.substitute(cur_expr, expand)
        if hasattr(nxt, "same_as") and nxt.same_as(cur_expr):
            break
        cur_expr = nxt
    return cur_expr


def build_free_var_int_map(
    expr: object, var_bindings: Mapping[str, int] | None, var_obj_bindings: Sequence[tuple[object, int]] = (),
    *, tirx: object, exclude_var: object | None = None,
) -> dict[object, object]:
    """Map free integer TIR vars in *expr* onto IntImm values from bindings."""
    vmap: dict[object, object] = {}
    if not var_bindings and not var_obj_bindings:
        return vmap
    lower_counts: dict[str, list[int]] = {}
    for rk, rv in (var_bindings or {}).items():
        if isinstance(rv, int) and not isinstance(rv, bool):
            lower_counts.setdefault(str(rk).lower(), []).append(int(rv))

    for fv in free_tirx_vars(expr, tirx=tirx):
        if exclude_var is not None and same_tirx_var(fv, exclude_var):
            continue
        bound = next((int(vv) for vo, vv in var_obj_bindings if same_tirx_var(fv, vo)), None)
        exact = str(getattr(fv, "name", "") or "")
        hint = str(getattr(fv, "name_hint", "") or "")
        if bound is None and var_bindings:
            for cn in (exact, hint):
                if cn and cn in var_bindings and isinstance(rv := var_bindings[cn], int) and not isinstance(rv, bool):
                    bound = int(rv)
                    break
        if bound is None and (lookup := (exact or hint).lower()):
            if len(matches := lower_counts.get(lookup, [])) == 1:
                bound = matches[0]
        if bound is not None:
            vmap[fv] = tirx.IntImm(str(getattr(fv, "dtype", "int32") or "int32"), bound)
    return vmap


def _lookup_stem_binding(cname: str, eff: Mapping[str, int], lower_eff: Mapping[str, int]) -> int | None:
    if cname and cname in eff:
        return eff[cname]
    lc = cname.lower()
    if lc and lc in lower_eff:
        return lower_eff[lc]
    for suffix in ("_dim", "_size", "_len", "_count", "_rows", "_cols"):
        if lc.endswith(suffix) and (stem := lc[: -len(suffix)]) in lower_eff:
            return lower_eff[stem]
        if lc and f"{lc}{suffix}" in lower_eff:
            return lower_eff[f"{lc}{suffix}"]
    for group in _DIM_ALIAS_GROUPS:
        if lc in group:
            for alias in group:
                if alias in lower_eff:
                    return lower_eff[alias]
    return None


def _match_binding_for_var(var: object, eff: Mapping[str, int], lower_eff: Mapping[str, int]) -> int | None:
    for cname in (str(getattr(var, "name", "") or ""), str(getattr(var, "name_hint", "") or "")):
        if (direct := _lookup_stem_binding(cname, eff, lower_eff)) is not None:
            return direct
        lc = cname.lower()
        pad_stem = lc[: lc.find("_pad")] if "_pad" in lc else (lc[4:] if lc.startswith("pad_") else "")
        if pad_stem and (base_val := _lookup_stem_binding(pad_stem, eff, lower_eff)) is not None and base_val > 0:
            return ((int(base_val) + 31) // 32) * 32
    return None


def augment_var_bindings_from_buffers(
    entry_fn: object, var_bindings: Mapping[str, int] | None,
    *, tirx: object, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[dict[str, int], list[tuple[object, int]]]:
    """Infer concrete symbolic variable bindings from buffer shapes and anchor dims."""
    if not var_bindings and not ordered_tensor_shapes:
        return {}, []
    eff = {str(k): int(v) for k, v in (var_bindings or {}).items() if isinstance(v, int) and not isinstance(v, bool)}
    lower_eff = {k.lower(): v for k, v in eff.items()}
    obj_bindings: list[tuple[object, int]] = []
    seen_ids: set[int] = set()

    def _bind_var(var: object, val: int) -> None:
        if id(var) not in seen_ids:
            seen_ids.add(id(var))
            obj_bindings.append((var, int(val)))
        for cn in (str(getattr(var, "name", "") or ""), str(getattr(var, "name_hint", "") or "")):
            if cn and cn not in eff:
                eff[cn] = int(val)
                lower_eff.setdefault(cn.lower(), int(val))

    buffer_map = getattr(entry_fn, "buffer_map", None) or {}
    tensor_idx = 0
    for param in getattr(entry_fn, "params", ()):
        buf = buffer_map.get(param)
        if buf is None:
            if isinstance(param, tirx.Var) and (m := _match_binding_for_var(param, eff, lower_eff)) is not None:
                _bind_var(param, m)
            continue
        shape = list(getattr(buf, "shape", ()))
        for dim_expr in (*shape, *(getattr(buf, "strides", ()) or ())):
            for fv in free_tirx_vars(dim_expr, tirx=tirx):
                if (m := _match_binding_for_var(fv, eff, lower_eff)) is not None:
                    _bind_var(fv, m)
        if ordered_tensor_shapes is not None and tensor_idx < len(ordered_tensor_shapes):
            pos_shape = ordered_tensor_shapes[tensor_idx]
            if len(shape) == len(pos_shape):
                for de, dv in zip(shape, pos_shape):
                    if isinstance(de, tirx.Var) and id(de) not in seen_ids and isinstance(dv, int) and not isinstance(dv, bool):
                        _bind_var(de, max(1, int(dv)))
        tensor_idx += 1
        if len(shape) == 2 and isinstance(shape[0], tirx.Var) and id(shape[0]) not in seen_ids and static_tirx_int(shape[1]) is not None:
            if (rv := lower_eff.get("q_total_rows", lower_eff.get("m_rows", lower_eff.get("m")))) is not None:
                _bind_var(shape[0], rv)
        elif len(shape) == 4:
            if isinstance(shape[0], tirx.Var) and id(shape[0]) not in seen_ids and (bv := lower_eff.get("b_dim", lower_eff.get("b"))) is not None:
                _bind_var(shape[0], bv)
            if (kv := lower_eff.get("s_kv")) is not None:
                if isinstance(shape[1], tirx.Var) and id(shape[1]) not in seen_ids and static_tirx_int(shape[2]) is not None:
                    _bind_var(shape[1], kv)
                elif isinstance(shape[2], tirx.Var) and id(shape[2]) not in seen_ids and static_tirx_int(shape[1]) is not None:
                    _bind_var(shape[2], kv)

    if not any(a in lower_eff for a in _NUMEL_ALIASES):
        synth_numel = 0
        for param in getattr(entry_fn, "params", ()):
            buf = buffer_map.get(param)
            if not buf or not (bshape := list(getattr(buf, "shape", ()) or ())):
                continue
            vals: list[int] = []
            for de in bshape:
                sv = static_tirx_int(de)
                mv = sv if (sv is not None and sv > 0) else (_match_binding_for_var(de, eff, lower_eff) if isinstance(de, tirx.Var) else None)
                if mv is None or mv <= 0:
                    vals = []
                    break
                vals.append(int(mv))
            if vals:
                synth_numel = max(synth_numel, math.prod(vals))
        if synth_numel <= 0:
            b_v, h_v, s_v, m_v, n_v, d_v, k_v = (_lookup_stem_binding(x, eff, lower_eff) for x in ("b", "h", "s", "m", "n", "d", "k"))
            row_axes, has_axis = 1, False
            if m_v and m_v > 0:
                row_axes = m_v if (b_v and s_v and m_v >= b_v * s_v) else (b_v or 1) * (h_v or 1) * m_v
                has_axis = True
            elif s_v and s_v > 0:
                row_axes, has_axis = (b_v or 1) * (h_v or 1) * s_v, True
            elif b_v and b_v > 0:
                row_axes, has_axis = b_v * (h_v or 1), True
            col_v = n_v or d_v or k_v
            synth_numel = row_axes * col_v if (col_v and col_v > 0) else (row_axes if has_axis else 0)
        if synth_numel > 0:
            for alias in _NUMEL_ALIASES:
                eff.setdefault(alias, int(synth_numel))
                lower_eff.setdefault(alias, int(synth_numel))

    return eff, obj_bindings


def resolve_symbolic_grid_extent(
    expr: object, let_env: Mapping[object, object], var_bindings: Mapping[str, int] | None,
    *, tirx: object, tvm_mod: object | None, var_obj_bindings: Sequence[tuple[object, int]] = (),
) -> tuple[int | None, bool]:
    """Resolve a blockIdx extent expression into an integer, or flag it symbolic."""
    if (static_val := static_tirx_int(expr)) is not None:
        return static_val, False
    cur = substitute_let_env(expr, let_env, tirx=tirx)
    if vmap := build_free_var_int_map(cur, var_bindings, var_obj_bindings, tirx=tirx):
        cur = tirx.stmt_functor.substitute(cur, vmap)
    if tvm_mod is not None and hasattr(tvm_mod, "arith") and hasattr(tvm_mod.arith, "Analyzer"):
        try:
            cur = tvm_mod.arith.Analyzer().simplify(cur)
        except Exception:
            pass
    resolved = static_tirx_int(cur)
    return (resolved, False) if resolved is not None else (None, True)


def _collect_div_denominators(entry_fn: object, *, tirx: object) -> list[object]:
    div_cls = tuple(c for n in ("FloorDiv", "FloorMod", "Div", "Mod") if (c := getattr(tirx, n, None)) is not None)
    if not div_cls or entry_fn is None:
        return []
    raw_denoms: list[object] = []
    let_env: dict[object, object] = {}
    let_cls = tuple(c for n in ("LetStmt", "Bind", "Let") if (c := getattr(tirx, n, None)) is not None)
    alloc_cls = tuple(c for n in ("AllocBuffer", "DeclBuffer") if (c := getattr(tirx, n, None)) is not None)
    sblock_cls = getattr(tirx, "SBlock", None)

    def _scan_buf(buf: object) -> None:
        for e in (*getattr(buf, "shape", ()), *(getattr(buf, "strides", ()) or ()), getattr(buf, "elem_offset", None)):
            if e is not None:
                tirx.stmt_functor.post_order_visit(e, _visit)

    def _visit(node: object) -> None:
        if let_cls and isinstance(node, let_cls):
            if (vo := getattr(node, "var", None)) is not None and (val := getattr(node, "value", None)) is not None:
                let_env[vo] = val
        if isinstance(node, div_cls):
            add_unique_expr(raw_denoms, getattr(node, "b", None))
        if alloc_cls and isinstance(node, alloc_cls) and (buf := getattr(node, "buffer", None)) is not None:
            _scan_buf(buf)
        if sblock_cls is not None and isinstance(node, sblock_cls):
            for buf in getattr(node, "alloc_buffers", ()) or ():
                _scan_buf(buf)

    if (body := getattr(entry_fn, "body", None)) is not None:
        tirx.stmt_functor.post_order_visit(body, _visit)
    for buf in (getattr(entry_fn, "buffer_map", None) or {}).values():
        _scan_buf(buf)
    if not let_env:
        return raw_denoms
    expanded: list[object] = []
    for d in raw_denoms:
        add_unique_expr(expanded, substitute_let_env(d, let_env, tirx=tirx))
    return expanded


def _max_nonzero_const_magnitude(expr: object, *, tirx: object) -> int:
    max_val = 1
    int_imm_cls = getattr(tirx, "IntImm", None)
    if int_imm_cls is None or expr is None:
        return max_val

    def _visit(node: object) -> None:
        nonlocal max_val
        if isinstance(node, int_imm_cls) and (val := static_tirx_int(node)) is not None and val != 0:
            max_val = max(max_val, abs(int(val)))

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return max_val


def _ensure_positive_denominators(
    denominators: Sequence[object], spec_var_map: dict[object, object], recorded_vars: list[object],
    allowed_vars: Sequence[object], *, is_handle_or_data: Callable[[object], bool],
    tirx: object, tvm_mod: object, max_rounds: int = 24,
) -> None:
    if not denominators:
        return
    _find_key = lambda vo: next((k for k in spec_var_map if same_tirx_var(vo, k)), None)
    _is_allowed = lambda vo: (
        isinstance(vo, tirx.Var)
        and not is_handle_or_data(vo)
        and "int" in str(getattr(vo, "dtype", "int32") or "int32").lower()
        and any(same_tirx_var(vo, a) for a in allowed_vars)
    )
    for denom in denominators:
        for fv in free_tirx_vars(denom, tirx=tirx):
            if _is_allowed(fv) and _find_key(fv) is None:
                recorded_vars.append(fv)
                spec_var_map[fv] = tirx.IntImm(str(getattr(fv, "dtype", "int32") or "int32"), 1)

    analyzer = tvm_mod.arith.Analyzer()

    def _simp(expr: object) -> tuple[int | None, bool]:
        try:
            return static_tirx_int(analyzer.simplify(expr)), False
        except Exception:
            return None, True

    def _set_val(key: object, new_val: int) -> bool:
        if (cur_v := static_tirx_int(spec_var_map[key])) is not None and int(cur_v) == int(new_val):
            return False
        spec_var_map[key] = tirx.IntImm(str(getattr(key, "dtype", "int32") or "int32"), int(new_val))
        return True

    for _ in range(max_rounds):
        changed = False
        for denom in denominators:
            sub_d = tirx.stmt_functor.substitute(denom, spec_var_map) if spec_var_map else denom
            simp_val, simp_err = _simp(sub_d)
            if not simp_err and (simp_val is None or simp_val > 0):
                continue
            if not (cands := [fv for fv in free_tirx_vars(denom, tirx=tirx) if _is_allowed(fv)]):
                continue
            max_const = _max_nonzero_const_magnitude(denom, tirx=tirx)
            _next_v = lambda cur_v, mc=max_const: min(_MAX_SAFE_SPEC_INT, max(mc, 1) if cur_v <= 0 else max(cur_v * 2, mc, 1))
            if nonpos := [(k, cv) for fv in cands if (k := _find_key(fv)) is not None and (cv := static_tirx_int(spec_var_map[k])) is not None and cv <= 0]:
                for k, cv in nonpos:
                    changed = _set_val(k, _next_v(cv)) or changed
                continue
            if len(cands) == 1 and (k := _find_key(cands[0])) is not None:
                cv = static_tirx_int(spec_var_map[k])
                changed = _set_val(k, _next_v(0 if cv is None else int(cv))) or changed
                continue
            cand_keys = [(k, 0 if (cv := static_tirx_int(spec_var_map[k])) is None else int(cv)) for fv in cands if (k := _find_key(fv)) is not None]
            best_key, best_next, best_trial = None, 0, (None if (simp_err or simp_val is None) else int(simp_val))
            for key, cur_int in cand_keys:
                other_prod = 1
                for ok, oi in cand_keys:
                    if ok is not key:
                        other_prod = min(_MAX_SAFE_SPEC_INT, other_prod * max(1, oi))
                dbl = _next_v(cur_int)
                boost = min(_MAX_SAFE_SPEC_INT, max(dbl, cur_int * max(max_const, 2) * 2, other_prod * max(max_const, 1)))
                ds = str(getattr(key, "dtype", "int32") or "int32")
                for cn in ([dbl] if boost == dbl else [boost, dbl]):
                    if cn == cur_int:
                        continue
                    tv, terr = _simp(tirx.stmt_functor.substitute(denom, {**spec_var_map, key: tirx.IntImm(ds, cn)}))
                    if not terr and tv is not None and ((tv > 0 and (best_trial is None or tv > best_trial or best_trial <= 0)) or (best_trial is None or tv > best_trial)):
                        best_trial, best_key, best_next = tv, key, cn
                        if tv > 0:
                            break
            if best_key is not None and best_trial is not None and best_trial > 0:
                changed = _set_val(best_key, best_next) or changed
                continue
            for key, cur_int in cand_keys:
                if cur_int > 1:
                    tv, terr = _simp(tirx.stmt_functor.substitute(denom, {**spec_var_map, key: tirx.IntImm(str(getattr(key, "dtype", "int32") or "int32"), 1)}))
                    if not terr and tv is not None and tv > 0:
                        best_key, best_next, best_trial = key, 1, tv
                        break
            if best_key is not None:
                changed = _set_val(best_key, best_next) or changed
            elif grow := [(k, ci) for k, ci in cand_keys if ci < _MAX_SAFE_SPEC_INT]:
                mk, mc = min(grow, key=lambda x: x[1])
                changed = _set_val(mk, _next_v(mc)) or changed
        if not changed:
            break


def specialize_primfunc_for_anchors(
    entry_fn: object, mod_in: object, var_bindings: Mapping[str, int] | None,
    *, tvm_mod: object, tirx: object, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[object, object]:
    """Specialize a PrimFunc + IRModule onto concrete anchor bindings."""
    if not var_bindings and not ordered_tensor_shapes:
        return entry_fn, mod_in
    eff_bindings, var_obj_bindings = augment_var_bindings_from_buffers(
        entry_fn, var_bindings, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes
    )
    if not eff_bindings and not var_obj_bindings:
        return entry_fn, mod_in
    lower_eff = {k.lower(): v for k, v in eff_bindings.items()}
    buffer_map = getattr(entry_fn, "buffer_map", None) or {}
    handle_data = (*buffer_map.keys(), *(b.data for b in buffer_map.values() if getattr(b, "data", None) is not None))
    _is_handle_or_data = lambda vo: any(same_tirx_var(vo, h) for h in handle_data)

    spec_var_map: dict[object, object] = {}
    recorded_vars: list[object] = []

    def _maybe_bind(vo: object) -> None:
        if not isinstance(vo, tirx.Var) or _is_handle_or_data(vo) or any(same_tirx_var(vo, p) for p in recorded_vars):
            return
        bound = next((int(vv) for vobj, vv in var_obj_bindings if same_tirx_var(vo, vobj)), None)
        if bound is None:
            bound = _match_binding_for_var(vo, eff_bindings, lower_eff)
        if bound is None or (ds := str(getattr(vo, "dtype", "int32") or "int32")) == "handle":
            return
        recorded_vars.append(vo)
        spec_var_map[vo] = tirx.IntImm(ds, int(bound))

    for p in getattr(entry_fn, "params", ()):
        _maybe_bind(p)
    buf_free_vars: list[object] = []
    for buf in buffer_map.values():
        for expr in (*getattr(buf, "shape", ()), *(getattr(buf, "strides", ()) or ()), buf.elem_offset):
            for fv in free_tirx_vars(expr, tirx=tirx):
                _maybe_bind(fv)
                add_unique_expr(buf_free_vars, fv)
    body_free_vars = free_tirx_vars(entry_fn.body, tirx=tirx)
    for fv in body_free_vars:
        _maybe_bind(fv)

    max_numel = max((int(lower_eff[k]) for k in _NUMEL_ALIASES if k in lower_eff and int(lower_eff[k]) > 0), default=0)
    for buf in buffer_map.values():
        axis_ints: list[int] = []
        for de in getattr(buf, "shape", ()) or ():
            sv = static_tirx_int(de)
            if sv is not None and sv > 0:
                axis_ints.append(int(sv))
            elif de in spec_var_map and (bsv := static_tirx_int(spec_var_map[de])) is not None and bsv > 0:
                axis_ints.append(int(bsv))
            else:
                axis_ints = []
                break
        if axis_ints:
            max_numel = max(max_numel, math.prod(axis_ints))
    capped_fb = min(max_numel if max_numel > 0 else 1024, _BUFFER_FREE_AXIS_CAP)

    rem_vars: list[object] = []

    def _collect_rem(expr: object) -> None:
        if expr is None:
            return
        for fv in free_tirx_vars(expr, tirx=tirx):
            ds = str(getattr(fv, "dtype", "int32") or "int32").lower()
            if (
                isinstance(fv, tirx.Var) and not _is_handle_or_data(fv) and "int" in ds
                and not any(same_tirx_var(fv, p) for p in recorded_vars)
                and any(same_tirx_var(fv, bfv) for bfv in body_free_vars)
            ):
                add_unique_expr(rem_vars, fv)

    def _visit_rem(node: object) -> None:
        if isinstance(node, tirx.For):
            _collect_rem(node.extent)
        elif isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            for idx_e in getattr(node, "indices", ()) or ():
                _collect_rem(idx_e)
        elif isinstance(node, tirx.Call) and any(t in str(getattr(node.op, "name", node.op)).lower() for t in ("copy", "region", "access_ptr")):
            for arg in getattr(node, "args", ()) or ():
                _collect_rem(arg)

    tirx.stmt_functor.post_order_visit(entry_fn.body, _visit_rem)
    param_vars = [p for p in getattr(entry_fn, "params", ()) if isinstance(p, tirx.Var) and not _is_handle_or_data(p)]
    for fv in rem_vars:
        fv_name = str(getattr(fv, "name_hint", "") or str(fv)).lower()
        in_buf = any(same_tirx_var(fv, bv) for bv in buf_free_vars)
        in_par = any(same_tirx_var(fv, pv) for pv in param_vars)
        is_tok = any(tok in fv_name for tok in _UNBOUND_SCALAR_NAME_TOKENS)
        ext = _UNBOUND_SCALAR_EXTENT_DEFAULT if ((in_par and not in_buf) or is_tok or not in_buf) else capped_fb
        recorded_vars.append(fv)
        spec_var_map[fv] = tirx.IntImm(str(getattr(fv, "dtype", "int32") or "int32"), int(ext))

    params_list = list(getattr(entry_fn, "params", ()))
    _ensure_positive_denominators(
        _collect_div_denominators(entry_fn, tirx=tirx),
        spec_var_map, recorded_vars, [*params_list, *buf_free_vars, *body_free_vars],
        is_handle_or_data=_is_handle_or_data, tirx=tirx, tvm_mod=tvm_mod,
    )
    if not spec_var_map:
        return entry_fn, mod_in

    extra = [v for v in recorded_vars if not any(same_tirx_var(v, p) for p in params_list)]
    tmp_fn = tirx.PrimFunc([*params_list, *extra], entry_fn.body, ret_type=entry_fn.ret_type, buffer_map=entry_fn.buffer_map, attrs=entry_fn.attrs) if extra else entry_fn
    spec_fn = tmp_fn.specialize(spec_var_map)
    target_gvar = next((gv for gv, fn in mod_in.functions.items() if same_tirx_var(fn, entry_fn)), None)
    if target_gvar is None:
        target_gvar = str(entry_fn.attrs.get("global_symbol", "main")) if entry_fn.attrs else "main"
    updated = {k: (spec_fn if k == target_gvar else v) for k, v in mod_in.functions.items()}
    updated.setdefault(target_gvar, spec_fn)
    spec_mod = tvm_mod.IRModule(updated, attrs=getattr(mod_in, "attrs", None))
    if hasattr(tirx, "transform") and hasattr(tirx.transform, "Simplify"):
        spec_mod = tirx.transform.Simplify()(spec_mod)
        spec_fn = spec_mod.functions.get(target_gvar, next((f for f in spec_mod.functions.values() if isinstance(f, tirx.PrimFunc)), spec_fn))
    return spec_fn, spec_mod


def entry_and_mod(prim_func: object, tvm_mod: object, tirx: object) -> tuple[object, object]:
    """Extract the entry PrimFunc and enclosing IRModule."""
    if isinstance(prim_func, tvm_mod.IRModule):
        funcs = [fn for fn in prim_func.functions.values() if isinstance(fn, tirx.PrimFunc)]
        if not funcs:
            raise ValueError("IRModule contains no PrimFunc entries")
        return funcs[0], prim_func
    sym = str(prim_func.attrs.get("global_symbol", "main")) if prim_func.attrs else "main"
    return prim_func, tvm_mod.IRModule({sym: prim_func})


def specialize_primfunc_for_anchor(
    prim_func: object, *, var_bindings: Mapping[str, int] | None = None, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None
) -> tuple[object, object]:
    """Specialize a TileLang PrimFunc or IRModule for one anchor shape without lowering."""
    from tilelang import tvm  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    entry_fn, mod_in = entry_and_mod(prim_func, tvm, tirx)
    return specialize_primfunc_for_anchors(
        entry_fn, mod_in, var_bindings, tvm_mod=tvm, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes
    )
