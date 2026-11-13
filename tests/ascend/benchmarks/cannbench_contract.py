# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Shared Helix precision harness adapted from CANNBench
# e90a053df93997e026d19fd4fd42fc03cbf2b2fd.
# See CANN_CONTRACT_LICENSE. No CANNBench API dependency at runtime.
from typing import Any, Callable, Dict, List, Optional, Tuple
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import math
import fcntl
import importlib.metadata
import os
import platform
import re
import shutil
import tempfile
import torch
import yaml


# Adapted source: utils/dtype_mapper.py
DTYPE_MAP: Dict = {}
DTYPE_STR_MAP: Dict = {}

def _build_dtype_map() -> Dict:
    import torch
    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64, 'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64, 'uint8': torch.uint8}
    optional_types = [('bfloat16', 'bfloat16'), ('uint16', 'uint16'), ('uint32', 'uint32'), ('uint64', 'uint64'), ('bool', 'bool')]
    for attr_name, key_name in optional_types:
        try:
            dtype_map[key_name] = getattr(torch, attr_name)
        except AttributeError:
            pass
    return dtype_map

def str_to_torch_dtype(dtype_str: str):
    global DTYPE_MAP
    if not DTYPE_MAP:
        DTYPE_MAP = _build_dtype_map()
    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower not in DTYPE_MAP:
        raise ValueError(f'不支持的数据类型: {dtype_str}. 支持的类型: {list(DTYPE_MAP.keys())}')
    return DTYPE_MAP[dtype_str_lower]

def torch_dtype_to_str(dtype) -> str:
    global DTYPE_MAP, DTYPE_STR_MAP
    if not DTYPE_MAP:
        DTYPE_MAP = _build_dtype_map()
    if not DTYPE_STR_MAP:
        DTYPE_STR_MAP = {dtype: name for name, dtype in DTYPE_MAP.items()}
    if dtype not in DTYPE_STR_MAP:
        raise ValueError(f'不支持的数据类型: {dtype}')
    return DTYPE_STR_MAP[dtype]

def is_float_dtype(dtype_str: str) -> bool:
    float_types = ['float16', 'float32', 'float64', 'bfloat16']
    return dtype_str.lower() in float_types

def is_int_dtype(dtype_str: str) -> bool:
    int_types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    return dtype_str.lower() in int_types

def is_bool_dtype(dtype_str: str) -> bool:
    return dtype_str.lower() == 'bool'

def get_dtype_size(dtype_str: str) -> int:
    dtype = str_to_torch_dtype(dtype_str)
    return dtype.itemsize

def get_supported_dtypes() -> list:
    global DTYPE_MAP
    if not DTYPE_MAP:
        DTYPE_MAP = _build_dtype_map()
    return list(DTYPE_MAP.keys())

# Adapted source: data/data_generator.py
INPUT_DISTRIBUTIONS = {'uniform': ('uniform', 'u'), 'normal': ('normal', 'norm', 'gaussian', 'n')}
_DIST_ALIASES = {a: canon for canon, aliases in INPUT_DISTRIBUTIONS.items() for a in aliases}
INPUT_DIST_CHOICES = sorted(_DIST_ALIASES)

def normalize_input_dist(dist: Any) -> str:
    if dist is None:
        return 'uniform'
    key = str(dist).strip().lower()
    if key not in _DIST_ALIASES:
        raise ValueError(f'不支持的输入分布: {dist!r}。可选: ' + '、'.join((f"{c}（别名 {', '.join(a[1:])}）" if len(a) > 1 else c for c, a in INPUT_DISTRIBUTIONS.items())))
    return _DIST_ALIASES[key]

class DataGenerator:

    def __init__(self, seed: int=None, input_dist: str='uniform'):
        if seed is not None:
            torch.manual_seed(seed)
        self.input_dist = normalize_input_dist(input_dist)

    def generate_input_tensor(self, shape: List[int], dtype: str, value_range: Any=None, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        if shape is None:
            return None
        if isinstance(shape, list) and None in shape:
            return None
        torch_dtype = str_to_torch_dtype(dtype)
        if self._is_unique_range(value_range):
            return self._gen_unique_int(shape, torch_dtype, dtype, value_range[1], value_range[2], generator=generator)
        min_val, max_val = self._parse_range(value_range, dtype)
        if is_float_dtype(dtype):
            if isinstance(min_val, str) or isinstance(max_val, str):
                return self._gen_float(shape, torch_dtype, min_val, max_val, generator=generator)
            if self._is_special_float(min_val) or self._is_special_float(max_val):
                return self._gen_float(shape, torch_dtype, min_val, max_val, generator=generator)
            if min_val == max_val:
                return torch.full(shape, float(min_val), dtype=torch_dtype)
            return self._gen_float(shape, torch_dtype, float(min_val), float(max_val), generator=generator)
        elif is_int_dtype(dtype):
            if min_val == max_val:
                return torch.full(shape, int(min_val), dtype=torch_dtype)
            min_int, max_int = (int(min_val), int(max_val))
            int64_min = -(1 << 63)
            int64_max = (1 << 63) - 1
            if dtype.lower() == 'int64' and max_int == int64_max and (int64_min <= min_int < max_int):
                return self._gen_int64_inclusive_max(shape, min_int, generator=generator)
            return torch.randint(min_int, max_int + 1, shape, generator=generator, dtype=torch_dtype)
        elif is_bool_dtype(dtype):
            return self._gen_bool(shape, min_val, max_val, generator=generator)
        else:
            return torch.zeros(shape, dtype=torch_dtype)

    def generate_input_tensors_from_case(self, input_shapes: List, dtypes: List, value_ranges: List, seed: Optional[int]=None) -> List:
        input_tensors = []
        num_inputs = self._count_inputs(input_shapes)
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
        if not isinstance(dtypes, list) or (dtypes and isinstance(dtypes[0], str) and (len(dtypes) < num_inputs)):
            if isinstance(dtypes, str):
                dtypes = [dtypes]
            elif not isinstance(dtypes, list):
                dtypes = ['float32']
        if isinstance(dtypes, list) and len(dtypes) < num_inputs:
            dtypes = dtypes + [dtypes[-1]] * (num_inputs - len(dtypes)) if dtypes else ['float32'] * num_inputs
        value_ranges = self._normalize_value_ranges(value_ranges, num_inputs)
        self._generate_tensors(input_shapes, dtypes, value_ranges, input_tensors, generator=gen)
        return input_tensors

    def _normalize_value_ranges(self, value_ranges: List, num_inputs: int) -> List:
        if not value_ranges:
            return [None] * num_inputs
        first_item = value_ranges[0]
        is_single_range = not isinstance(first_item, list)
        if is_single_range:
            return [value_ranges] * num_inputs
        else:
            if len(value_ranges) < num_inputs:
                value_ranges = value_ranges + [value_ranges[-1]] * (num_inputs - len(value_ranges))
            return value_ranges

    @staticmethod
    def _gen_int64_inclusive_max(shape: List[int], min_val: int, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        int64_min = -(1 << 63)
        int64_max = (1 << 63) - 1
        half_range = 1 << 63
        interval_size = int64_max - min_val + 1
        if interval_size < half_range:
            offsets = torch.randint(0, interval_size, shape, generator=generator, dtype=torch.int64)
            return offsets + min_val
        if interval_size == half_range:
            offsets = torch.randint(int64_min, 0, shape, generator=generator, dtype=torch.int64)
            return offsets.bitwise_xor(int64_min)
        remainder = interval_size - half_range
        upper_base = min_val - int64_min
        result = torch.empty(shape, dtype=torch.int64)
        flat_result = result.reshape(-1)
        chunk_size = 1 << 20
        for start in range(0, flat_result.numel(), chunk_size):
            end = min(start + chunk_size, flat_result.numel())
            chunk_shape = (end - start,)
            offsets = torch.randint(int64_min, 0, chunk_shape, generator=generator, dtype=torch.int64).bitwise_xor(int64_min)
            upper_half = torch.randint(0, 2, chunk_shape, generator=generator, dtype=torch.int8)
            if remainder < half_range:
                invalid = upper_half.bool() & (offsets >= remainder)
                while bool(invalid.any().item()):
                    replacement_offsets = torch.randint(int64_min, 0, chunk_shape, generator=generator, dtype=torch.int64).bitwise_xor(int64_min)
                    replacement_half = torch.randint(0, 2, chunk_shape, generator=generator, dtype=torch.int8)
                    offsets = torch.where(invalid, replacement_offsets, offsets)
                    upper_half = torch.where(invalid, replacement_half, upper_half)
                    invalid = upper_half.bool() & (offsets >= remainder)
                upper_offsets = torch.where(offsets < remainder, offsets, torch.zeros_like(offsets))
            else:
                upper_offsets = offsets
            lower_values = offsets + min_val
            upper_values = upper_offsets + upper_base
            flat_result[start:end] = torch.where(upper_half.bool(), upper_values, lower_values)
        return result

    def _parse_range(self, value_range: Any, dtype: str) -> tuple:
        if value_range is None:
            return (0.0, 1.0) if is_float_dtype(dtype) else (0, 100)
        if isinstance(value_range, list):
            if len(value_range) >= 2:
                return (self._coerce_endpoint(value_range[0]), self._coerce_endpoint(value_range[1]))
            elif len(value_range) == 1:
                v = self._coerce_endpoint(value_range[0])
                return (v, v)
        v = self._coerce_endpoint(value_range)
        return (v, v)

    @staticmethod
    def _coerce_endpoint(val: Any) -> Any:
        if not isinstance(val, str):
            return val
        if val.strip().lower() in ('nan', 'inf', '-inf', '+inf'):
            return val
        try:
            return float(val)
        except ValueError:
            return val

    @staticmethod
    def _is_unique_range(value_range: Any) -> bool:
        return isinstance(value_range, list) and len(value_range) == 3 and (value_range[0] == 'unique')

    def _gen_unique_int(self, shape: List[int], torch_dtype: torch.dtype, dtype: str, min_val: Any, max_val: Any, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        if not is_int_dtype(dtype):
            raise ValueError(f'unique 值域仅支持整数 dtype，收到 {dtype}')
        lo, hi = (int(min_val), int(max_val))
        numel = 1
        for s in shape:
            numel *= int(s)
        span = hi - lo + 1
        if span < numel:
            raise ValueError(f'unique 值域容量不足: 区间 [{lo}, {hi}] 共 {span} 个值，张量需要 {numel} 个唯一值')
        perm = torch.randperm(span, generator=generator)[:numel]
        return (perm + lo).reshape(shape).to(torch_dtype)

    @staticmethod
    def _is_special_float(val: Any) -> bool:
        import math
        return isinstance(val, float) and (math.isnan(val) or math.isinf(val))

    def _is_tensor_list(self, shape_item: Any) -> bool:
        return isinstance(shape_item, list) and shape_item and isinstance(shape_item[0], list) and shape_item[0] and isinstance(shape_item[0][0], int)

    def _gen_float(self, shape: List[int], dtype: torch.dtype, min_val: float, max_val: float, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        if isinstance(min_val, str) or isinstance(max_val, str):
            return self._gen_special(shape, dtype, min_val, max_val, generator=generator)
        if self._is_special_float(min_val) or self._is_special_float(max_val):
            return self._gen_special(shape, dtype, min_val, max_val, generator=generator)
        finfo = torch.finfo(dtype)
        dmin = finfo.min
        dmax = finfo.max
        min_val = max(min_val, dmin)
        max_val = min(max_val, dmax)
        if self.input_dist == 'normal' and max_val > min_val:
            return self._gen_float_normal(shape, dtype, min_val, max_val, generator=generator)
        range_val = max_val - min_val
        rand_f64 = torch.rand(shape, dtype=torch.float64, generator=generator)
        if math.isfinite(range_val):
            tensor_f64 = rand_f64 * range_val + min_val
        else:
            tensor_f64 = (1.0 - rand_f64) * min_val + rand_f64 * max_val
        tensor_f64 = torch.clamp(tensor_f64, dmin, dmax)
        return tensor_f64.to(dtype)

    def _gen_float_normal(self, shape: List[int], dtype: torch.dtype, min_val: float, max_val: float, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        finfo = torch.finfo(dtype)
        range_val = max_val - min_val
        if math.isfinite(range_val):
            mu = min_val + range_val / 2.0
            sigma = range_val / 6.0
        else:
            mu = min_val / 2.0 + max_val / 2.0
            sigma = max_val / 6.0 - min_val / 6.0
        z = torch.randn(shape, dtype=torch.float64, generator=generator)
        tensor_f64 = z * sigma + mu
        tensor_f64 = torch.clamp(tensor_f64, min_val, max_val)
        tensor_f64 = torch.clamp(tensor_f64, finfo.min, finfo.max)
        return tensor_f64.to(dtype)

    def _gen_special(self, shape: List[int], dtype: torch.dtype, min_val: Any, max_val: Any, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        import math

        def to_float(v):
            if v == 'inf':
                return float('inf')
            if v == '-inf':
                return float('-inf')
            if v == 'nan':
                return float('nan')
            return float(v)
        min_f = to_float(min_val) if isinstance(min_val, str) else min_val
        max_f = to_float(max_val) if isinstance(max_val, str) else max_val
        if math.isnan(min_f) and math.isnan(max_f):
            if generator is not None:
                rand_base = torch.rand(shape, dtype=torch.float32, generator=generator)
                tensor = (rand_base * 2 - 1).to(dtype)
            else:
                tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
            if generator is not None:
                nan_rand = torch.rand(shape, dtype=torch.float32, generator=generator)
                nan_mask = nan_rand < 0.5
            else:
                nan_mask = torch.rand(shape, dtype=torch.float32) < 0.5
            tensor[nan_mask] = float('nan')
            return tensor
        if min_f == max_f and (math.isinf(min_f) or math.isinf(max_f)):
            return torch.full(shape, min_f, dtype=dtype)
        if generator is not None:
            rand_base = torch.rand(shape, dtype=torch.float32, generator=generator)
            tensor = (rand_base * 2 - 1).to(dtype)
        else:
            tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
        finite_min = min_f if math.isfinite(min_f) else None
        finite_max = max_f if math.isfinite(max_f) else None
        if finite_min is not None or finite_max is not None:
            tensor = torch.clamp(tensor, min=finite_min, max=finite_max)
        flat = tensor.flatten()
        n = max(1, len(flat) // 20)
        if min_f == float('-inf'):
            flat[:n] = float('-inf')
        if max_f == float('inf'):
            flat[-n:] = float('inf')
        return tensor

    def _gen_bool(self, shape: List[int], min_val: Any, max_val: Any, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        lo = min(1, max(0, int(min_val)))
        hi = min(1, max(0, int(max_val)))
        if lo > hi:
            lo, hi = (hi, lo)
        if lo == hi:
            return torch.full(shape, bool(lo), dtype=torch.bool)
        return torch.randint(lo, hi + 1, shape, generator=generator, dtype=torch.int64).to(torch.bool)

    def _count_inputs(self, shapes: Any) -> int:
        if isinstance((sizes := shapes), list):
            return len(sizes)
        return 1

    def _generate_tensors(self, shapes, dtypes, value_ranges, output: list, generator: Optional[torch.Generator]=None):
        if not isinstance(shapes, list):
            return
        if shapes and isinstance(shapes[0], int):
            dtype_item = dtypes if isinstance(dtypes, str) else dtypes[0] if isinstance(dtypes, list) and dtypes else 'float32'
            vr_item = value_ranges if isinstance(value_ranges, list) and len(value_ranges) == 2 and (not isinstance(value_ranges[0], list)) or self._is_unique_range(value_ranges) else None
            output.append(self.generate_input_tensor(shapes, dtype_item, vr_item, generator=generator))
            return
        for i, s in enumerate(shapes):
            d = dtypes[i] if isinstance(dtypes, list) and i < len(dtypes) else dtypes
            vr = value_ranges[i] if isinstance(value_ranges, list) and i < len(value_ranges) else value_ranges
            if s is None:
                output.append(None)
            elif isinstance(s, list) and (not s or isinstance(s[0], int)):
                dtype_item = d if isinstance(d, str) else d[0] if isinstance(d, list) and d else 'float32'
                vr_item = vr if isinstance(vr, list) and len(vr) == 2 and (not isinstance(vr[0], list)) or self._is_unique_range(vr) else None
                output.append(self.generate_input_tensor(s, dtype_item, vr_item, generator=generator))
            elif isinstance(s, list) and s and isinstance(s[0], list) and s[0] and isinstance(s[0][0], int):
                sub_list = []
                sub_dtypes = d if isinstance(d, list) else d
                if isinstance(vr, list):
                    is_scalar_range = len(vr) == 2 and (not isinstance(vr[0], (list, tuple))) and (not isinstance(vr[1], (list, tuple)))
                    sub_vr = [vr] * len(s) if is_scalar_range else vr
                else:
                    sub_vr = None
                for j, sub_shape in enumerate(s):
                    sub_d = sub_dtypes[j] if isinstance(sub_dtypes, list) and j < len(sub_dtypes) else sub_dtypes
                    if sub_vr and isinstance(sub_vr, list) and (j < len(sub_vr)):
                        item_vr = sub_vr[j]
                    else:
                        item_vr = sub_vr
                    sub_list.append(self.generate_input_tensor(sub_shape, sub_d, item_vr, generator=generator))
                output.append(sub_list)
            elif isinstance(s, list) and s and isinstance(s[0], list) and s[0] and isinstance(s[0][0], list):
                sub_list = []
                for j, sub_item in enumerate(s):
                    sub_d = d[j] if isinstance(d, list) and j < len(d) else d
                    sub_vr_item = vr[j] if isinstance(vr, list) and j < len(vr) else vr
                    self._generate_tensors(sub_item, sub_d, sub_vr_item, sub_list, generator=generator)
                output.append(sub_list)

# Adapted source: utils/param_builder.py
class ParamBuilder:

    def __init__(self, importer=None):
        self.importer = importer

    def build_params_by_proto_order(self, input_tensors: List[Any], proto_inputs: List[Any], case_attrs: Dict[str, Any]=None) -> Dict[str, Any]:
        params = {}
        case_attrs = case_attrs or {}
        if len(input_tensors) != len(proto_inputs):
            return self._build_params_legacy(input_tensors, proto_inputs, case_attrs)
        for i, input_info in enumerate(proto_inputs):
            input_name = input_info.name
            tensor = input_tensors[i]
            if tensor is None:
                params[input_name] = None
            else:
                params[input_name] = tensor
        for attr_key, attr_val in case_attrs.items():
            if attr_key not in params:
                params[attr_key] = attr_val
        return params

    def _build_params_legacy(self, input_tensors: List[Any], proto_inputs: List[Any], case_attrs: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        actual_tensors = [t for t in input_tensors if t is not None]
        proto_tensor_count = len([i for i in proto_inputs if 'Tensor' in str(i.dtype) or isinstance(i.dtype, list)])
        if len(actual_tensors) == proto_tensor_count:
            tensor_idx = 0
            for input_info in proto_inputs:
                if 'Tensor' in str(input_info.dtype) or isinstance(input_info.dtype, list):
                    if tensor_idx < len(input_tensors):
                        params[input_info.name] = input_tensors[tensor_idx]
                        tensor_idx += 1
                    else:
                        params[input_info.name] = None
                elif getattr(input_info, 'optional', False):
                    params[input_info.name] = None
        else:
            tensor_idx = 0
            for input_info in proto_inputs:
                if tensor_idx < len(input_tensors):
                    params[input_info.name] = input_tensors[tensor_idx]
                    tensor_idx += 1
                else:
                    params[input_info.name] = None
        for attr_key, attr_val in case_attrs.items():
            if attr_key not in params:
                params[attr_key] = attr_val
        return params

    def build_call_params(self, golden_func: Callable, case: Any, input_tensors: List, override_shapes: List=None) -> Dict[str, Any]:
        sig = signature(golden_func)
        params = {}
        tensor_params = []
        tensor_list_params = []
        optional_tensor_params = []
        optional_tensor_list_params = []
        attr_params = []
        all_tensor_params_in_order = []
        for name, param in sig.parameters.items():
            annotation = str(param.annotation) if param.annotation != Parameter.empty else ''
            is_optional = 'Optional' in annotation or 'NoneType' in annotation or param.default != Parameter.empty
            if 'List[' in annotation and 'Tensor' in annotation:
                all_tensor_params_in_order.append(name)
                if is_optional:
                    optional_tensor_list_params.append(name)
                else:
                    tensor_list_params.append(name)
            elif 'Tensor' in annotation:
                all_tensor_params_in_order.append(name)
                if is_optional:
                    optional_tensor_params.append(name)
                else:
                    tensor_params.append(name)
            else:
                attr_params.append(name)
        original_shapes = override_shapes if override_shapes is not None else getattr(case, 'input_shapes', None)
        tensor_param_map = self._build_tensor_param_map(tensor_params, tensor_list_params, original_shapes, optional_tensor_params, optional_tensor_list_params, all_tensor_params_in_order)
        tensor_idx = 0
        for position, param_name in sorted(tensor_param_map.items(), key=lambda x: x[0]):
            if tensor_idx < len(input_tensors):
                val = input_tensors[tensor_idx]
                all_tensor_list_params = tensor_list_params + optional_tensor_list_params
                if isinstance(val, list) and param_name in tensor_params:
                    params[param_name] = val[0] if val else None
                    tensor_idx += 1
                    if len(val) > 1:
                        input_tensors = input_tensors[:tensor_idx] + [v for v in val[1:]] + input_tensors[tensor_idx:]
                elif param_name in all_tensor_list_params:
                    params[param_name] = val if isinstance(val, list) else [val]
                    tensor_idx += 1
                else:
                    params[param_name] = val
                    tensor_idx += 1
        attrs = getattr(case, 'attrs', None) or {}
        for name in attr_params:
            if name in attrs:
                params[name] = self._convert_value(attrs[name])
            elif sig.parameters[name].default != Parameter.empty:
                params[name] = sig.parameters[name].default
        return params

    def _build_tensor_param_map(self, tensor_params: List[str], tensor_list_params: List[str], original_shapes: Optional[List], optional_tensor_params: List[str]=None, optional_tensor_list_params: List[str]=None, all_tensor_params_in_order: List[str]=None) -> Dict[int, str]:
        optional_tensor_params = optional_tensor_params or []
        optional_tensor_list_params = optional_tensor_list_params or []
        all_tensor_params_in_order = all_tensor_params_in_order or []
        required_params = tensor_params + tensor_list_params
        optional_params = optional_tensor_params + optional_tensor_list_params
        if original_shapes is None:
            result = {}
            idx = 0
            for name in required_params:
                result[idx] = name
                idx += 1
            return result
        shapes_to_check = self._normalize_input_shapes(original_shapes)
        if shapes_to_check is None:
            result = {}
            idx = 0
            for name in required_params:
                result[idx] = name
                idx += 1
            return result
        param_map = {}
        for i, shape_item in enumerate(shapes_to_check):
            if i < len(all_tensor_params_in_order):
                param_map[i] = all_tensor_params_in_order[i]
        return param_map

    def _normalize_input_shapes(self, original_shapes: List) -> Optional[List]:
        if not isinstance(original_shapes, list) or not original_shapes:
            return None
        if len(original_shapes) == 1:
            first = original_shapes[0]
            if isinstance(first, list) and first:
                inner_first = first[0] if first else None
                if isinstance(inner_first, list) or inner_first is None:
                    return first
        return original_shapes

    def _convert_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if value == 'inf':
                return float('inf')
            elif value == '-inf':
                return float('-inf')
            elif value == 'nan':
                return float('nan')
            try:
                return float(value)
            except ValueError:
                pass
        return value

# Adapted source: eval/index_check.py
def _as_list(outputs: Any) -> List[Any]:
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]

def validate_index_output(x: torch.Tensor, dim: int, idx_candidate: torch.Tensor, values_candidate: torch.Tensor) -> Tuple[bool, str]:
    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor) or (not isinstance(values_candidate, torch.Tensor)):
        return (False, 'index_gather: 输入/索引/值必须均为 Tensor')
    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)
    val_c = values_candidate.detach().cpu()
    if idx_c.dim() != x_c.dim():
        return (False, f'index_gather: 索引维度 {idx_c.dim()} 与输入维度 {x_c.dim()} 不一致，无法 gather')
    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not 0 <= dim_n < x_c.dim():
        return (False, f'index_gather: dim={dim} 越界（输入 {x_c.dim()} 维）')
    if idx_c.shape != val_c.shape:
        return (False, f'index_gather: 索引输出形状 {tuple(idx_c.shape)} 与值输出形状 {tuple(val_c.shape)} 不一致')
    for d in range(x_c.dim()):
        if d != dim_n and idx_c.size(d) > x_c.size(d):
            return (False, f'index_gather: 第 {d} 维索引尺寸 {idx_c.size(d)} 超过输入 {x_c.size(d)}')
    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return (False, f'index_gather: 索引越界 [{lo},{hi}]，合法区间 [0,{dim_size})')
    gathered = torch.gather(x_c, dim_n, idx_c).to(val_c.dtype)
    if gathered.is_floating_point() or val_c.is_floating_point():
        both_nan = torch.isnan(gathered) & torch.isnan(val_c)
        eq = (gathered == val_c) | both_nan
    else:
        eq = gathered == val_c
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return (False, f'index_gather: 候选索引指向的元素与其值输出不一致（{mism} 处不符），索引无效')
    return (True, '')
_REDUCE_FUNCS = {'amax': torch.amax, 'amin': torch.amin}

def validate_index_reduce_output(x: torch.Tensor, dim: int, idx_candidate: torch.Tensor, reduce: str, keepdim: bool=False) -> Tuple[bool, str]:
    if reduce not in _REDUCE_FUNCS:
        return (False, f'index_gather: value_reduce={reduce!r} 不支持（可选 amax / amin）')
    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor):
        return (False, 'index_gather: 输入/索引必须均为 Tensor')
    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)
    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not 0 <= dim_n < x_c.dim():
        return (False, f'index_gather: dim={dim} 越界（输入 {x_c.dim()} 维）')
    if not keepdim:
        if idx_c.dim() != x_c.dim() - 1:
            return (False, f'index_gather: keepdim=False 下索引应为 {x_c.dim() - 1} 维，实际 {idx_c.dim()} 维')
        idx_c = idx_c.unsqueeze(dim_n)
    elif idx_c.dim() != x_c.dim():
        return (False, f'index_gather: keepdim=True 下索引应为 {x_c.dim()} 维，实际 {idx_c.dim()} 维')
    ref = _REDUCE_FUNCS[reduce](x_c, dim=dim_n, keepdim=True)
    if idx_c.shape != ref.shape:
        return (False, f'index_gather: 索引输出形状 {tuple(idx_candidate.shape)} 与 {reduce} 归约结果形状 {tuple(ref.shape if keepdim else ref.squeeze(dim_n).shape)} 不一致')
    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return (False, f'index_gather: 索引越界 [{lo},{hi}]，合法区间 [0,{dim_size})')
    gathered = torch.gather(x_c, dim_n, idx_c)
    if gathered.is_floating_point():
        eq = (gathered == ref) | torch.isnan(gathered) & torch.isnan(ref)
    else:
        eq = gathered == ref
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return (False, f'index_gather: 候选索引指向的元素不是该切片的 {reduce} （{mism} 处不符），索引无效')
    return (True, '')

def _has_duplicate_indices(idx: torch.Tensor, dim: int) -> bool:
    if idx.size(dim) <= 1:
        return False
    return bool((torch.diff(idx.sort(dim=dim).values, dim=dim) == 0).any())

def validate_index_topk_output(x: torch.Tensor, dim: int, idx: torch.Tensor, k: int, largest: bool=True) -> Tuple[bool, str]:
    x = x.detach().cpu()
    idx = idx.detach().cpu().to(torch.int64)
    dim_n = dim if dim >= 0 else x.dim() + dim
    if not 0 <= dim_n < x.dim():
        return (False, f'index_gather: dim={dim} 越界')
    if not 0 < k <= x.size(dim_n):
        return (False, f'index_gather: k={k} 超出输入维度范围')
    expected_shape = list(x.shape)
    expected_shape[dim_n] = k
    if list(idx.shape) != expected_shape:
        return (False, f'index_gather: top-k 索引形状应为 {expected_shape}')
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= x.size(dim_n)):
        return (False, 'index_gather: top-k 索引越界')
    if _has_duplicate_indices(idx, dim_n):
        return (False, 'index_gather: top-k 索引不能重复')
    chosen = torch.gather(x, dim_n, idx)
    boundary = torch.topk(x, k, dim=dim_n, largest=largest).values.narrow(dim_n, k - 1, 1)
    legal = chosen >= boundary if largest else chosen <= boundary
    if x.is_floating_point():
        legal = legal | torch.isnan(chosen) & torch.isnan(boundary)
    if not bool(legal.all()):
        return (False, 'index_gather: 索引未指向 top-k 元素')
    return (True, '')

def validate_index_gather_outputs(op_info: Any, params: Dict[str, Any], case_attrs: Dict[str, Any], ai_outputs: Any) -> Tuple[bool, str]:
    outputs = getattr(op_info, 'outputs', None) or []
    name_to_idx = {o.name: i for i, o in enumerate(outputs)}
    ai = _as_list(ai_outputs)
    for i, out in enumerate(outputs):
        spec: Optional[Dict[str, Any]] = getattr(out, 'index_gather', None)
        if not spec:
            continue
        src_name = spec.get('input')
        dim_attr = spec.get('dim_attr')
        val_name = spec.get('value_output')
        reduce = spec.get('value_reduce')
        topk = bool(spec.get('value_topk'))
        x = params.get(src_name) if isinstance(params, dict) else None
        if not isinstance(x, torch.Tensor):
            return (False, f'index_gather[{out.name}]: 找不到源输入张量 {src_name!r}')
        if i >= len(ai) or not isinstance(ai[i], torch.Tensor):
            return (False, f'index_gather[{out.name}]: 缺少索引输出 #{i}')
        idx = ai[i]
        forms = int(val_name is not None) + int(reduce is not None) + int(topk)
        if forms == 0:
            return (False, f'index_gather[{out.name}]: 须声明 value_output（配对值输出）或 value_reduce（由输入归约得到参考值）之一')
        if forms > 1:
            return (False, f'index_gather[{out.name}]: value_output 与 value_reduce 互斥，不能同时声明')
        attrs = case_attrs if isinstance(case_attrs, dict) else {}
        if dim_attr is not None:
            if dim_attr not in attrs:
                return (False, f'index_gather[{out.name}]: 缺少维度属性 {dim_attr!r}')
            dim = attrs[dim_attr]
        elif 'dim' in spec:
            dim = spec['dim']
        elif 'dim' in attrs:
            dim = attrs['dim']
        else:
            return (False, f'index_gather[{out.name}]: 缺少维度属性 {dim_attr!r}（index_gather 声明的 dim_attr 未在 case attrs 中找到）')
        if topk:
            k_attr = spec.get('k_attr', 'k')
            if k_attr not in attrs:
                return (False, f'index_gather[{out.name}]: 缺少 top-k 属性 {k_attr!r}')
            largest = bool(attrs.get(spec.get('largest_attr'), spec.get('largest', True)))
            source = x.detach().cpu()
            indices = idx.detach().cpu()
            transform = spec.get('value_transform')
            if transform is not None:
                if transform != 'softmax':
                    return (False, f'index_gather[{out.name}]: 未知变换 {transform!r}')
                source = torch.softmax(source, dim=int(dim))
            mask_name = spec.get('finished_input')
            finished = params.get(mask_name) if mask_name else None
            if finished is not None:
                dim_n = int(dim) if int(dim) >= 0 else x.dim() + int(dim)
                mask = finished.detach().cpu()
                if dim_n != x.dim() - 1 or mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:-1]):
                    return (False, f'index_gather[{out.name}]: finished 掩码与输入形状不匹配')
                if tuple(indices.shape) != tuple(x.shape[:-1]) + (int(attrs[k_attr]),):
                    return (False, f'index_gather[{out.name}]: 索引形状不匹配')
                if not bool((indices[mask] == x.size(dim_n)).all()):
                    return (False, f'index_gather[{out.name}]: finished 行须返回 num_expert 哨兵')
                source = source[~mask]
                indices = indices[~mask]
                dim = -1
            ok, msg = validate_index_topk_output(source, int(dim), indices, int(attrs[k_attr]), largest)
        elif reduce is not None:
            keepdim_attr = spec.get('keepdim_attr')
            keepdim = bool(attrs.get(keepdim_attr, False)) if keepdim_attr else False
            ok, msg = validate_index_reduce_output(x, int(dim), idx, reduce, keepdim)
        else:
            if val_name not in name_to_idx:
                return (False, f'index_gather[{out.name}]: value_output {val_name!r} 未在输出中声明')
            vi = name_to_idx[val_name]
            if vi >= len(ai) or not isinstance(ai[vi], torch.Tensor):
                return (False, f'index_gather[{out.name}]: 缺少配对值输出 {val_name!r} (#{vi})')
            ok, msg = validate_index_output(x, int(dim), idx, ai[vi])
            if ok and spec.get('distinct'):
                idx_cpu = idx.detach().cpu().to(torch.int64)
                if _has_duplicate_indices(idx_cpu, int(dim)):
                    return (False, f'index_gather[{out.name}]: 索引不能重复')
        if not ok:
            return (False, msg)
    return (True, '')


class Case(SimpleNamespace):
    def get_case_id_str(self):
        return self.label


def load_cases(task_dir, task):
    result = []
    for raw in yaml.safe_load((Path(task_dir) / 'cases.yaml').read_text())['cases']:
        shapes = raw['input_shape']
        if shapes and not isinstance(shapes[0], list):
            shapes = [shapes]
        dtypes = raw['dtype']
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        if len(dtypes) == 1 and len(shapes) > 1:
            dtypes = dtypes * len(shapes)
        result.append(Case(case_num=raw['case_id'], label=f"{task}_{raw['case_id']}",
            input_shapes=shapes, dtypes=dtypes, attrs=raw.get('attrs') or {},
            value_ranges=raw.get('value_range') or []))
    assert result and len({c.case_num for c in result}) == len(result)
    return result


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(x, fn) for x in value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    return value


def outputs(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [y for x in value for y in (list(x) if isinstance(x, (tuple, list)) else [x])]
    raise AssertionError(f'Unsupported output type: {type(value).__name__}')


class ReferenceCache:
    """Five disk entries, with an exclusive lease per active operator test."""
    LIMIT = 5
    DEFAULT_ROOT = '/data2/zhanghaotian/11-5-cannbench-cache-dir'

    def __init__(self, harness):
        self.harness = harness
        self.folder = None
        self.lease = None
        self.hits = self.misses = 0
        self.generations = {}

    def identity(self):
        versions = {}
        for name in ('torch', 'torch-npu', 'numpy', 'torchvision', 'en-dtypes'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        paths = [self.harness.task_dir / name for name in ('golden.py', 'cases.yaml', 'proto.yaml')]
        paths.append(Path(__file__))
        value = {
            'task': self.harness.task,
            'files': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
            'versions': versions, 'torch': str(torch.__version__),
            'python': platform.python_version(), 'machine': platform.machine(),
            'torch_config': torch.__config__.show(),
            'dtype': str(torch.get_default_dtype()),
            'threads': [torch.get_num_threads(), torch.get_num_interop_threads()],
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ATEN_CPU_CAPABILITY')},
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def acquire(path):
        handle = path.open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError as error:
            handle.close()
            if isinstance(error, BlockingIOError):
                return None
            raise

    def __enter__(self):
        location = os.environ.get('CANNBENCH_CACHE_DIR', self.DEFAULT_ROOT)
        if location == 'off':
            return self
        try:
            if not re.fullmatch(r'level[1-9][0-9]*/[a-z][a-z0-9_]*', self.harness.task):
                raise ValueError('Invalid cache task')
            self.signature = self.identity()
            root = Path(location)
            entries, locks = root / 'operators', root / '.locks'
            entries.mkdir(parents=True, exist_ok=True)
            locks.mkdir(exist_ok=True)
            key = self.harness.task.replace('/', '__')
            with (root / '.allocation.lock').open('a') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                self.lease = self.acquire(locks / (key + '.lock'))
                if self.lease is None:
                    return self
                folder = entries / key
                if not folder.exists():
                    occupied = sorted(entries.iterdir(), key=lambda p: p.stat().st_mtime_ns)
                    while len(occupied) >= self.LIMIT:
                        for old in occupied:
                            lease = self.acquire(locks / (old.name + '.lock'))
                            if lease is not None:
                                try:
                                    shutil.rmtree(old)
                                finally:
                                    lease.close()
                                occupied.remove(old)
                                break
                        else:
                            self.lease.close()
                            self.lease = None
                            return self
                manifest = folder / 'identity.json'
                try:
                    previous = json.loads(manifest.read_text())
                except (OSError, ValueError):
                    previous = {}
                if not isinstance(previous, dict):
                    previous = {}
                if previous.get('identity') != self.signature:
                    if folder.exists():
                        shutil.rmtree(folder)
                    folder.mkdir()
                    manifest.write_text(json.dumps({'task': self.harness.task, 'identity': self.signature}))
                self.folder = folder
                os.utime(folder, None)
        except (OSError, ValueError) as error:
            print(f'[reference-cache] unavailable: {error}', flush=True)
            self.__exit__(None, None, None)
        return self

    def __exit__(self, *unused):
        if self.folder is not None:
            try:
                os.utime(self.folder, None)
            except OSError:
                pass
        if self.lease is not None:
            self.lease.close()
        print(f'[reference-cache] hits={self.hits} misses={self.misses} active={self.folder is not None}', flush=True)
        self.folder = self.lease = None

    def get(self, case, kind, compute):
        if self.folder is None:
            return compute()
        name = hashlib.sha256(case.get_case_id_str().encode()).hexdigest() + '.' + kind + '.pt'
        path = self.folder / name
        try:
            record = torch.load(path, map_location='cpu', weights_only=True)
            if (record['identity'] != self.signature or record['case'] != case.get_case_id_str()
                    or record['kind'] != kind or not isinstance(record['generation'], str)
                    or (kind == 'native' and record['generation'] != self.generations.get(case.get_case_id_str()))):
                raise ValueError('Cache identity mismatch')
            value = record['value']
            if kind == 'oracle' and (not isinstance(value, dict) or set(value) != {'inputs', 'oracle'}):
                raise ValueError('Invalid oracle payload')
            if kind == 'oracle':
                self.generations[case.get_case_id_str()] = record['generation']
            self.hits += 1
            os.utime(self.folder, None)
            return value
        except FileNotFoundError:
            pass
        except Exception as error:
            print(f'[reference-cache] rebuilding {case.get_case_id_str()} {kind}: {type(error).__name__}', flush=True)
        # Reference failures must propagate to the normal precision failure path.
        self.misses += 1
        value = compute()
        if kind == 'oracle':
            self.generations[case.get_case_id_str()] = os.urandom(16).hex()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.folder, prefix='pending-', suffix='.pt', delete=False) as file:
                temporary = Path(file.name)
            torch.save({'identity': self.signature, 'case': case.get_case_id_str(), 'kind': kind,
                        'generation': self.generations.get(case.get_case_id_str()),
                        'value': map_tensors(value, lambda x: x.detach().cpu())}, temporary)
            os.replace(temporary, path)
            os.utime(self.folder, None)
        except Exception as error:
            print(f'[reference-cache] write skipped: {type(error).__name__}: {error}', flush=True)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return value


class Harness:
    def __init__(self, task, task_dir):
        self.task = task
        self.task_dir = Path(task_dir)
        proto = yaml.safe_load((self.task_dir / 'proto.yaml').read_text())['operator']
        self.info = SimpleNamespace(
            inputs=[SimpleNamespace(**x) for x in proto['inputs']],
            outputs=[SimpleNamespace(**dict({'optional': False, 'compare': True,
                      'index_gather': None}, **x)) for x in proto['outputs']],
            precision_thresholds=proto.get('precision_thresholds', {}))
        self.cases = load_cases(self.task_dir, task)
        spec = importlib.util.spec_from_file_location(
            '_reference_' + task.replace('/', '_'), self.task_dir / 'golden.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reference = getattr(module, task.rsplit('/', 1)[-1])
        self.reference_module = module
        self.get_input = getattr(module, 'get_input', None)
        self.param_builder = ParamBuilder()

    def prepare(self, case):
        seed = int.from_bytes(hashlib.sha256(case.get_case_id_str().encode()).digest()[:8], 'big') % (2**31)
        tensors = DataGenerator().generate_input_tensors_from_case(
            input_shapes=case.input_shapes, dtypes=case.dtypes,
            value_ranges=case.value_ranges, seed=seed)
        if self.get_input:
            params = {item.name: tensors[i] if i < len(tensors) else None
                      for i, item in enumerate(self.info.inputs)}
            params.update({k: v for k, v in case.attrs.items() if k not in params})
            params.setdefault('skip2_exist', True)
            tensors = self.get_input(**params)
        return tensors

    def params(self, case, tensors, function=None):
        if not self.get_input:
            return self.param_builder.build_call_params(function or self.reference, case, tensors)
        params = {}
        iterator = iter(tensors)
        for name, param in signature(self.reference).parameters.items():
            if 'Tensor' in str(param.annotation):
                params[name] = next(iterator, None)
            elif name in case.attrs:
                params[name] = case.attrs[name]
            elif param.default is not Parameter.empty:
                params[name] = param.default
        return params

    def reference_output(self, case, tensors, *, native=False):
        suffix = '_bench' if native else '_oracle'
        function = getattr(self.reference_module, self.task.rsplit('/', 1)[-1] + suffix, self.reference)
        # References may mutate inputs; every execution gets independent storage.
        copied = map_tensors(tensors, lambda x: x.clone())
        if not native:
            copied = map_tensors(copied, lambda x: x.double() if x.is_floating_point() else x)
        with torch.no_grad():
            return function(**self.params(case, copied, function))

    def compare(self, case, actual, oracle, tensors, native_fn):
        from npu_compare import compare_case_result
        actuals, goldens = outputs(actual), outputs(oracle)
        assert len(actuals) == len(goldens), 'Output count differs from oracle'
        for i, spec in enumerate(self.info.outputs):
            if i >= len(actuals):
                assert spec.optional, f'Missing required output {spec.name}'
                continue
            allowed = [spec.dtype] if isinstance(spec.dtype, str) else spec.dtype
            assert type(actuals[i]) is torch.Tensor, f'{spec.name}: expected plain Tensor'
            assert not allowed or str(actuals[i].dtype).removeprefix('torch.') in allowed, f'{spec.name}: dtype {actuals[i].dtype} not in {allowed}'
        native_cache = []
        def native_leaf(i):
            if not native_cache:
                native_cache.append(outputs(native_fn()))
            return native_cache[0][i]
        details = []
        for i, (actual_leaf, golden_leaf) in enumerate(zip(actuals, goldens)):
            if i < len(self.info.outputs) and not self.info.outputs[i].compare:
                continue
            assert type(actual_leaf) is torch.Tensor, f'Output {i}: expected plain Tensor'
            assert actual_leaf.is_contiguous(), f'Output {i}: non-contiguous'
            result = compare_case_result(
                case_id=f'{case.get_case_id_str()}:{i}', actual=actual_leaf, golden=golden_leaf,
                inputs=tensors, accuracy_mode='cann-bench',
                native_ref_operator=lambda unused, i=i: native_leaf(i),
                extra={'precision_thresholds': self.info.precision_thresholds or {}})
            details.append({'output': i, 'passed': result.passed, 'message': result.message})
        ok, message = validate_index_gather_outputs(
            self.info, self.params(case, tensors), case.attrs, actual)
        assert ok, message
        return all(row['passed'] for row in details), details

    def run(self, operator_api):
        with ReferenceCache(self) as cache:
            self._run(operator_api, cache)

    def _run(self, operator_api, cache):
        failures = []
        for case in self.cases:
            label = case.get_case_id_str()
            try:
                def prepare_reference():
                    tensors = self.prepare(case)
                    return {'inputs': tensors, 'oracle': self.reference_output(case, tensors)}
                bundle = cache.get(case, 'oracle', prepare_reference)
                tensors, oracle = bundle['inputs'], bundle['oracle']
                params = self.params(case, map_tensors(tensors, lambda x: x.npu()))
                actual = operator_api(**params)
                passed, details = self.compare(case, actual, oracle, tensors,
                    lambda: cache.get(case, 'native',
                        lambda: self.reference_output(case, tensors, native=True)))
                if not passed:
                    failures.append(label)
                print(json.dumps({'case': label, 'passed': passed, 'outputs': details}), flush=True)
                del tensors, oracle, params, actual, bundle
            except Exception as exc:
                failures.append(label)
                print(json.dumps({'case': label, 'passed': False, 'error': str(exc)}), flush=True)
        assert not failures, f'Failed {len(failures)}/{len(self.cases)} cases: {failures}'
        print('All tests passed!')

    def bench_cases(self):
        return [{'id': str(case.case_num), 'spec': case, 'warmup': 3, 'repeats': 5}
                for case in self.cases]

    def bench_fn(self, operator_api, case):
        tensors = self.prepare(case['spec'])
        params = self.params(case['spec'], map_tensors(tensors, lambda x: x.npu()))
        return lambda: operator_api(**params)
