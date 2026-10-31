#!/usr/bin/env python3
"""Replay representative real rocBLAS trace shapes and check sampled FP32 results.

Run with the inference server stopped, one GPU visible, and the selected stock or
tuned ROCBLAS_TENSILE_LIBPATH already set (or use --library). A small C++ adapter
is compiled against the installed headers: enum values and rocBLAS function ABI
are compiler checked rather than guessed in ctypes. Python owns GPU storage.

--reference-out creates a stock sample report when absent; when the file exists,
the current replay is additionally compared against its actual stock samples.
An existing reference is never overwritten. All active output elements are
checked for NaN/Inf; numerical reference covers only sampled rows/columns/batches.
FP8, pointer-array batched GEMM, complex types and non-FP32 accumulation are
explicitly skipped. Passing sampled tests does not establish full-model quality.
"""
import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time


class RetryTemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        for attempt in range(5):
            try:
                super().cleanup()
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))


ADAPTER = r'''
#include <rocblas/rocblas.h>
#include <cstdint>
#include <cstring>

static bool dtype(const char* name, rocblas_datatype& value) {
    if (std::strcmp(name, "f16_r") == 0) value = rocblas_datatype_f16_r;
    else if (std::strcmp(name, "bf16_r") == 0) value = rocblas_datatype_bf16_r;
    else if (std::strcmp(name, "f32_r") == 0) value = rocblas_datatype_f32_r;
    else return false;
    return true;
}
static bool operation(char name, rocblas_operation& value) {
    if (name == 'N') value = rocblas_operation_none;
    else if (name == 'T') value = rocblas_operation_transpose;
    else if (name == 'C') value = rocblas_operation_conjugate_transpose;
    else return false;
    return true;
}
extern "C" int enum_datatype(const char* name) {
    rocblas_datatype value;
    return dtype(name, value) ? static_cast<int>(value) : -1;
}
extern "C" int enum_operation(char name) {
    rocblas_operation value;
    return operation(name, value) ? static_cast<int>(value) : -1;
}
extern "C" int replay(
    int strided, char ta_name, char tb_name, int m, int n, int k,
    const void* alpha, const void* a, const char* at_name, int lda, int64_t sa,
    const void* b, const char* bt_name, int ldb, int64_t sb,
    const void* beta, const void* c, const char* ct_name, int ldc, int64_t sc,
    void* d, const char* dt_name, int ldd, int64_t sd, int batch,
    const char* compute_name, int algo, int32_t solution, uint32_t flags) {
    rocblas_datatype at, bt, ct, dt, compute;
    rocblas_operation ta, tb;
    if (!dtype(at_name, at) || !dtype(bt_name, bt) || !dtype(ct_name, ct)
        || !dtype(dt_name, dt) || !dtype(compute_name, compute)) return -1001;
    if (!operation(ta_name, ta) || !operation(tb_name, tb)) return -1002;
    rocblas_handle handle = nullptr;
    rocblas_status status = rocblas_create_handle(&handle);
    if (status != rocblas_status_success) return static_cast<int>(status);
    status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    if (status == rocblas_status_success) {
        if (strided) {
            status = rocblas_gemm_strided_batched_ex(
                handle, ta, tb, m, n, k, alpha,
                a, at, lda, static_cast<rocblas_stride>(sa),
                b, bt, ldb, static_cast<rocblas_stride>(sb), beta,
                c, ct, ldc, static_cast<rocblas_stride>(sc),
                d, dt, ldd, static_cast<rocblas_stride>(sd), batch,
                compute, static_cast<rocblas_gemm_algo>(algo), solution, flags);
        } else {
            status = rocblas_gemm_ex(
                handle, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb,
                beta, c, ct, ldc, d, dt, ldd, compute,
                static_cast<rocblas_gemm_algo>(algo), solution, flags);
        }
    }
    const rocblas_status destroy = rocblas_destroy_handle(handle);
    return static_cast<int>(status == rocblas_status_success ? destroy : status);
}
'''


@dataclass(frozen=True)
class Shape:
    function: str
    trans_a: str
    trans_b: str
    m: int
    n: int
    k: int
    alpha: float
    beta: float
    a_type: str
    b_type: str
    c_type: str
    d_type: str
    compute_type: str
    lda: int
    ldb: int
    ldc: int
    ldd: int
    stride_a: int
    stride_b: int
    stride_c: int
    stride_d: int
    batch_count: int
    algo: int
    solution_index: int
    flags: int

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()

    def group(self):
        return (self.function, self.trans_a, self.trans_b, self.a_type, self.b_type,
                self.c_type, self.d_type, self.compute_type, self.batch_count)

    def flops(self):
        return 2 * self.m * self.n * self.k * self.batch_count


def integer(text):
    # Accept decimal and hex flags; never silently truncate a decimal fraction.
    try:
        return int(text, 0)
    except ValueError:
        value = float(text)
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f'Non-integral GEMM parameter: {text}')
        return int(value)


def parse_line(line):
    match = re.search(r'(?:\S*/)?rocblas-bench\b', line)
    if not match:
        return None
    tokens = shlex.split(line[match.end():])
    options = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith('-'):
            if '=' in token:
                name, value = token.split('=', 1)
            elif i + 1 < len(tokens):
                name, value = token, tokens[i + 1]
                i += 1
            else:
                raise ValueError(f'Option without a value: {token}')
            options[name.lstrip('-')] = value
        else:
            raise ValueError(f'Unexpected trace token: {token}')
        i += 1

    def need(*names):
        for name in names:
            if name in options:
                return options[name]
        raise ValueError('Missing trace parameter: ' + '/'.join(names))

    function = need('f', 'function')
    if function not in ('gemm_ex', 'gemm_strided_batched_ex'):
        raise ValueError(f'Unsupported function {function}')
    types = [need(name) for name in ('a_type', 'b_type', 'c_type', 'd_type', 'compute_type')]
    if any(name not in ('f16_r', 'bf16_r', 'f32_r') for name in types):
        raise ValueError('Unsupported dtype combination: ' + ','.join(types))
    if types[4] != 'f32_r':
        raise ValueError(f'Only FP32 accumulation is covered, got {types[4]}')
    ta, tb = need('transposeA', 'transA').upper(), need('transposeB', 'transB').upper()
    if ta not in ('N', 'T', 'C') or tb not in ('N', 'T', 'C'):
        raise ValueError(f'Invalid transpose: {ta}/{tb}')
    m, n, k = (integer(need(name)) for name in ('m', 'n', 'k'))
    lda, ldb, ldc, ldd = (integer(need(name)) for name in ('lda', 'ldb', 'ldc', 'ldd'))
    alpha, beta = float(need('alpha')), float(need('beta'))
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError('Non-finite alpha/beta is outside this numerical test')
    if min(m, n, k) <= 0:
        raise ValueError('Zero/negative dimensions are not numerical coverage')
    if max(m, n, k, lda, ldb, ldc, ldd) > 2**31 - 1:
        raise ValueError('Dimensions exceed the 32-bit rocblas_gemm_ex interface')
    if lda < (m if ta == 'N' else k) or ldb < (k if tb == 'N' else n) or min(ldc, ldd) < m:
        raise ValueError('Invalid leading dimension')
    strided = function == 'gemm_strided_batched_ex'
    strides = [integer(need(name)) if strided else 0
               for name in ('stride_a', 'stride_b', 'stride_c', 'stride_d')]
    batch = integer(need('batch_count')) if strided else 1
    if batch <= 0 or batch > 2**31 - 1 or min(strides) < 0:
        raise ValueError('Negative stride or invalid batch count is unsupported')
    return Shape(function, ta, tb, m, n, k, alpha, beta, *types, lda, ldb, ldc, ldd,
                 *strides, batch, integer(options.get('algo', '0')),
                 integer(options.get('solution_index', '0')),
                 integer(options.get('flags', '0')))


def read_traces(paths):
    shapes, skipped = {}, []
    total_lines = recognized = 0
    for path in paths:
        with path.open(errors='replace') as trace:
            for lineno, line in enumerate(trace, 1):
                total_lines += 1
                if 'rocblas-bench' not in line:
                    continue
                recognized += 1
                try:
                    shape = parse_line(line)
                    if shape is None:
                        continue
                    key = shape.key()
                    if key not in shapes:
                        shapes[key] = {'shape': shape, 'occurrences': 0,
                                       'trace_line': line.strip(),
                                       'source': str(path), 'line_number': lineno}
                    shapes[key]['occurrences'] += 1
                except (ValueError, TypeError) as exc:
                    skipped.append({'source': str(path), 'line_number': lineno,
                                    'reason': str(exc), 'trace_line': line.strip()[:4000]})
    return shapes, skipped, {'total_lines': total_lines, 'rocblas_bench_lines': recognized,
                             'supported_unique_shapes': len(shapes),
                             'unsupported_lines': len(skipped)}


def select_shapes(shapes, limit):
    groups = {}
    for key, record in shapes.items():
        groups.setdefault(record['shape'].group(), []).append(key)
    groups = sorted(groups.values(),
                    key=lambda keys: (-sum(shapes[k]['occurrences'] for k in keys), min(keys)))
    choices = []
    for keys in groups:
        by_n = sorted(keys, key=lambda k: (shapes[k]['shape'].n, -shapes[k]['occurrences'], k))
        decode = [k for k in keys if shapes[k]['shape'].n <= 64] or keys
        choices.append([
            by_n[0], by_n[-1],
            max(decode, key=lambda k: (shapes[k]['occurrences'], -shapes[k]['shape'].n)),
            max(keys, key=lambda k: shapes[k]['shape'].flops()),
        ])
    selected = []
    # Round-robin priorities preserve group coverage before adding more variants.
    for priority in range(4):
        for group in choices:
            if group[priority] not in selected:
                selected.append(group[priority])
    for key in sorted(shapes, key=lambda k: (-shapes[k]['occurrences'], -shapes[k]['shape'].flops(), k)):
        if key not in selected:
            selected.append(key)
    return selected[:limit], len(groups)


def compile_adapter(args, directory):
    compiler = args.compiler or shutil.which('hipcc')
    if not compiler:
        candidate = args.rocm_root / 'bin/hipcc'
        compiler = str(candidate) if candidate.is_file() else None
    if not compiler:
        raise RuntimeError('hipcc not found; use --compiler or --rocm-root')
    header = args.rocm_root / 'include/rocblas/rocblas.h'
    if not header.is_file():
        raise RuntimeError(f'Installed rocBLAS header missing: {header}')
    source, library = directory / 'rocblas_replay.cpp', directory / 'rocblas_replay.so'
    source.write_text(ADAPTER)
    command = [compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
               '-I' + str(args.rocm_root / 'include'),
               '-I' + str(args.rocm_root / 'hip/include'), str(source),
               '-L' + str(args.rocm_root / 'lib'), '-lrocblas',
               '-Wl,-rpath,' + str(args.rocm_root / 'lib'), '-o', str(library)]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
    info = {'command': command, 'returncode': completed.returncode,
            'stdout': completed.stdout, 'stderr': completed.stderr,
            'adapter_sha256': hashlib.sha256(ADAPTER.encode()).hexdigest(),
            'header': str(header)}
    if completed.returncode:
        raise RuntimeError('Adapter compilation failed: ' + json.dumps(info))
    bridge = ctypes.CDLL(str(library))
    ptr, string, integer32, stride = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int64
    bridge.replay.argtypes = [
        integer32, ctypes.c_char, ctypes.c_char, integer32, integer32, integer32,
        ptr, ptr, string, integer32, stride,
        ptr, string, integer32, stride,
        ptr, ptr, string, integer32, stride,
        ptr, string, integer32, stride, integer32, string,
        integer32, ctypes.c_int32, ctypes.c_uint32,
    ]
    bridge.replay.restype = ctypes.c_int
    bridge.enum_datatype.argtypes = [string]
    bridge.enum_datatype.restype = ctypes.c_int
    bridge.enum_operation.argtypes = [ctypes.c_char]
    bridge.enum_operation.restype = ctypes.c_int
    info['installed_enum_values'] = {
        'datatypes': {name: bridge.enum_datatype(name.encode()) for name in ('f16_r', 'bf16_r', 'f32_r')},
        'operations': {name: bridge.enum_operation(name.encode()) for name in ('N', 'T', 'C')},
    }
    return bridge, info


def samples(length, count):
    if length <= count:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def storage_lengths(shape):
    def extent(rows, cols, ld, stride):
        return rows + (cols - 1) * ld + (shape.batch_count - 1) * stride
    return {
        'a': extent(shape.m if shape.trans_a == 'N' else shape.k,
                    shape.k if shape.trans_a == 'N' else shape.m, shape.lda, shape.stride_a),
        'b': extent(shape.k if shape.trans_b == 'N' else shape.n,
                    shape.n if shape.trans_b == 'N' else shape.k, shape.ldb, shape.stride_b),
        'c': extent(shape.m, shape.n, shape.ldc, shape.stride_c),
        'd': extent(shape.m, shape.n, shape.ldd, shape.stride_d),
    }


def error_metrics(torch, actual, reference, rtol, atol, normalization):
    delta = (actual - reference).abs()
    allowed = atol + rtol * reference.abs()
    return {'max_abs_error': delta.max().item(),
            'max_relative_error_with_atol_floor': (delta / reference.abs().clamp_min(atol)).max().item(),
            'normalized_max_error': delta.max().item() / normalization,
            'elements_outside_tolerance': int((delta > allowed).sum().item()),
            'elements': actual.numel(),
            'rtol': rtol, 'atol': atol}


def run_shape(torch, bridge, args, shape, baseline):
    key = shape.key()
    dtype = {'f16_r': torch.float16, 'bf16_r': torch.bfloat16, 'f32_r': torch.float32}
    lengths = storage_lengths(shape)
    type_names = dict(zip(('a', 'b', 'c', 'd'),
                          (shape.a_type, shape.b_type, shape.c_type, shape.d_type)))
    bytes_per = {'f16_r': 2, 'bf16_r': 2, 'f32_r': 4}
    estimate = sum(lengths[name] * bytes_per[type_names[name]] for name in lengths)
    free, _ = torch.cuda.mem_get_info(args.device)
    cap = min(args.max_storage_mib * 1024**2, int(free * .8))
    result = {'parameter_sha256': key, 'shape': asdict(shape),
              'storage_elements': lengths, 'storage_bytes': estimate,
              'gpu_free_bytes_before': free, 'status': 'running'}
    if estimate > cap:
        result.update(status='skipped', reason=f'Exact-layout storage {estimate} bytes exceeds cap {cap}')
        return result
    seed = (args.seed + int(key[:16], 16)) % (2**63 - 1)
    result['seed'] = seed
    device = torch.device(f'cuda:{args.device}')
    generator = torch.Generator(device=device).manual_seed(seed)
    storage = {}
    for name in ('a', 'b', 'c'):
        storage[name] = torch.empty(lengths[name], dtype=dtype[type_names[name]], device=device)
        storage[name].uniform_(-0.5, 0.5, generator=generator)
    storage['d'] = torch.full((lengths['d'],), float('nan'), dtype=dtype[shape.d_type], device=device)
    alpha = torch.tensor(shape.alpha, dtype=torch.float32)
    beta = torch.tensor(shape.beta, dtype=torch.float32)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    code = bridge.replay(
        int(shape.function == 'gemm_strided_batched_ex'), shape.trans_a.encode(), shape.trans_b.encode(),
        shape.m, shape.n, shape.k, alpha.data_ptr(), storage['a'].data_ptr(), shape.a_type.encode(),
        shape.lda, shape.stride_a, storage['b'].data_ptr(), shape.b_type.encode(), shape.ldb, shape.stride_b,
        beta.data_ptr(), storage['c'].data_ptr(), shape.c_type.encode(), shape.ldc, shape.stride_c,
        storage['d'].data_ptr(), shape.d_type.encode(), shape.ldd, shape.stride_d, shape.batch_count,
        shape.compute_type.encode(), shape.algo, shape.solution_index, shape.flags)
    result['rocblas_status'] = code
    if code:
        raise RuntimeError(f'rocBLAS returned status {code} for {key}')
    torch.cuda.synchronize(args.device)
    result['replay_seconds_including_handle'] = time.perf_counter() - started
    # Check every logical D element, excluding unaddressed padding. Chunking bounds
    # the temporary boolean storage, including large lm_head matrices.
    cols_per_chunk = max(1, min(shape.n, (32 * 1024**2) // max(1, shape.m)))
    finite = True
    for batch in range(shape.batch_count):
        for col in range(0, shape.n, cols_per_chunk):
            count = min(cols_per_chunk, shape.n - col)
            view = torch.as_strided(storage['d'], (shape.m, count), (1, shape.ldd),
                                    batch * shape.stride_d + col * shape.ldd)
            if not torch.isfinite(view).all().item():
                finite = False
                break
        if not finite:
            break
    result['all_logical_output_finite'] = finite
    rows, cols = samples(shape.m, args.sample_side), samples(shape.n, args.sample_side)
    batches = samples(shape.batch_count, args.sample_batches)
    result['sample_indices'] = {'rows': rows, 'columns': cols, 'batches': batches}
    row_gpu = torch.tensor(rows, dtype=torch.int64, device=device)
    col_gpu = torch.tensor(cols, dtype=torch.int64, device=device)
    reduction = torch.arange(shape.k, dtype=torch.int64, device=device)
    a_index = (row_gpu[:, None] + reduction[None, :] * shape.lda if shape.trans_a == 'N'
               else row_gpu[:, None] * shape.lda + reduction[None, :])
    b_index = (reduction[:, None] + col_gpu[None, :] * shape.ldb if shape.trans_b == 'N'
               else reduction[:, None] * shape.ldb + col_gpu[None, :])
    cd_index = row_gpu[:, None] + col_gpu[None, :] * shape.ldc
    d_index = row_gpu[:, None] + col_gpu[None, :] * shape.ldd
    ref_batches, rounded_batches, actual_batches = [], [], []
    input_hasher = hashlib.sha256()
    for batch in batches:
        a_cpu = storage['a'][a_index + batch * shape.stride_a].float().cpu()
        b_cpu = storage['b'][b_index + batch * shape.stride_b].float().cpu()
        c_cpu = storage['c'][cd_index + batch * shape.stride_c].float().cpu()
        actual = storage['d'][d_index + batch * shape.stride_d].float().cpu()
        for tensor in (a_cpu, b_cpu, c_cpu):
            input_hasher.update(tensor.contiguous().numpy().tobytes())
        # CPU FP32 matmul is independent from the selected GPU rocBLAS kernels.
        reference = shape.alpha * (a_cpu @ b_cpu) + shape.beta * c_cpu
        ref_batches.append(reference)
        rounded_batches.append(reference.to(dtype[shape.d_type]).float())
        actual_batches.append(actual)
    reference, rounded, actual = (torch.stack(values) for values in
                                  (ref_batches, rounded_batches, actual_batches))
    result['sampled_input_sha256'] = input_hasher.hexdigest()
    result['reference_fp32'] = reference.tolist()
    result['reference_rounded_to_output_dtype'] = rounded.tolist()
    result['actual'] = actual.tolist() if torch.isfinite(actual).all().item() else None
    result['reference_method'] = 'CPU FP32 sampled A[rows,:] @ B[:,cols], full K, then alpha/beta and output rounding'
    if not finite or not torch.isfinite(actual).all().item():
        result.update(status='failed', reason='NaN/Inf in active output')
        return result
    rtol, atol = {'bf16_r': (0.03, 0.03), 'f16_r': (0.003, 0.003),
                  'f32_r': (0.0001, 0.0001)}[shape.d_type]
    normalization = max(1.0, reference.abs().max().item())
    result['error_vs_fp32'] = error_metrics(torch, actual, reference, rtol, atol, normalization)
    result['error_vs_rounded_reference'] = error_metrics(torch, actual, rounded, rtol, atol, normalization)
    result['normalization'] = normalization
    result['normalized_error_limit'] = args.normalized_error
    passed = (result['error_vs_fp32']['elements_outside_tolerance'] == 0 and
              result['error_vs_rounded_reference']['normalized_max_error'] <= args.normalized_error)
    if baseline is not None:
        if key not in baseline:
            raise RuntimeError(f'Stock reference missing selected shape {key}')
        old = baseline[key]
        if old.get('status') != 'passed':
            raise RuntimeError(f'Stock reference did not pass for shape {key}')
        if old.get('seed') != seed or old.get('sample_indices') != result['sample_indices']:
            raise RuntimeError(f'Stock sample seed/indices mismatch for shape {key}')
        if old.get('sampled_input_sha256') != result['sampled_input_sha256']:
            raise RuntimeError(f'Stock sampled input hash mismatch for shape {key}')
        stock_actual = torch.tensor(old['actual'], dtype=torch.float32)
        result['error_vs_stock_actual'] = error_metrics(torch, actual, stock_actual, rtol, atol, normalization)
        passed = (passed and result['error_vs_stock_actual']['elements_outside_tolerance'] == 0 and
                  result['error_vs_stock_actual']['normalized_max_error'] <= args.normalized_error)
    result['status'] = 'passed' if passed else 'failed'
    if not passed:
        result['reason'] = 'Sample numerical tolerance exceeded'
    return result


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trace', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-out', type=Path, help='Create stock sample report if absent, otherwise load for tuned comparison')
    parser.add_argument('--library', type=Path, help='Set ROCBLAS_TENSILE_LIBPATH before importing Torch')
    parser.add_argument('--rocm-root', type=Path, default=Path('/opt/dtk'))
    parser.add_argument('--compiler', help='hipcc path; defaults to PATH then <rocm-root>/bin/hipcc')
    parser.add_argument('--device', type=int, default=0, help='Process-local visible GPU number')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--sample-side', type=int, default=16)
    parser.add_argument('--sample-batches', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--normalized-error', type=float, default=0.003)
    parser.add_argument('--max-storage-mib', type=float, default=16384)
    parser.add_argument('--select-only', action='store_true', help='Parse/select without importing Torch or compiling the adapter')
    args = parser.parse_args()
    if min(args.limit, args.sample_side, args.sample_batches) < 1 or args.max_storage_mib <= 0:
        parser.error('Limits, sample counts and storage cap must be positive')
    if not math.isfinite(args.normalized_error) or args.normalized_error <= 0:
        parser.error('normalized-error must be positive and finite')
    if args.reference_out and args.out.resolve() == args.reference_out.resolve():
        parser.error('--out must differ from --reference-out; stock references are immutable')
    if args.library:
        if not args.library.is_dir():
            parser.error('--library directory does not exist')
        os.environ['ROCBLAS_TENSILE_LIBPATH'] = str(args.library.resolve())
    shapes, skipped, counts = read_traces(args.trace)
    selected, group_count = select_shapes(shapes, args.limit)
    report = {'schema_version': 1, 'started_unix': time.time(),
              'trace_counts': counts, 'supported_groups': group_count,
              'selection': 'Per group: N minimum/maximum, common decode N<=64, maximum FLOPs; round-robin, then frequency',
              'selected_parameters': [{**asdict(shapes[key]['shape']),
                                       'parameter_sha256': key,
                                       'trace_occurrences': shapes[key]['occurrences'],
                                       'trace_line': shapes[key]['trace_line']} for key in selected],
              'unsupported_trace_lines': skipped, 'results': [],
              'environment': {'ROCBLAS_TENSILE_LIBPATH': os.environ.get('ROCBLAS_TENSILE_LIBPATH'),
                              'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES'),
                              'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES')},
              'limitations': ['Numerical comparison samples rows/columns and at most sample-batches, full K; active D finiteness is exhaustive.',
                              'Supported: gemm_ex and gemm_strided_batched_ex, real f16/bf16/f32 storage, FP32 accumulation.',
                              'FP8, complex, pointer-array batched and other accumulation types are skipped.',
                              'Normalized error uses max(1,max_abs_fp32_reference); 0.003 limit applies after reference output-dtype rounding to separate quantization from accumulation.',
                              'Sampled random-input GEMM checks do not replace model-level long-context regression.']}
    save(args.out, report)
    if args.select_only:
        print(json.dumps({'selected': len(selected), 'groups': group_count, **counts}), flush=True)
        return 0 if selected else 1
    if not selected:
        report.update(ok=False, error='No supported shapes selected')
        save(args.out, report)
        return 1
    baseline = None
    if args.reference_out and args.reference_out.exists():
        stock = json.loads(args.reference_out.read_text())
        if not stock.get('ok'):
            raise RuntimeError('Existing stock reference is not a successful numerical report')
        baseline = {item['parameter_sha256']: item for item in stock['results']}
        report['stock_reference'] = str(args.reference_out)
    try:
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.set_float32_matmul_precision('highest')
        if not torch.version.hip:
            raise RuntimeError('This replay requires a HIP PyTorch build')
        torch.cuda.set_device(args.device)
        prop = torch.cuda.get_device_properties(args.device)
        report['device'] = {'local_id': args.device, 'name': prop.name,
                            'gfx': getattr(prop, 'gcnArchName', None),
                            'cu': prop.multi_processor_count, 'memory_bytes': prop.total_memory,
                            'torch': torch.__version__, 'hip': torch.version.hip}
        with RetryTemporaryDirectory(prefix='rocblas_numerics_') as temp:
            bridge, report['adapter'] = compile_adapter(args, Path(temp))
            for key in selected:
                try:
                    result = run_shape(torch, bridge, args, shapes[key]['shape'], baseline)
                except Exception as exc:
                    result = {'parameter_sha256': key, 'shape': asdict(shapes[key]['shape']),
                              'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                report['results'].append(result)
                save(args.out, report)
                print(json.dumps({'shape': key[:16], 'm': shapes[key]['shape'].m,
                                  'n': shapes[key]['shape'].n, 'k': shapes[key]['shape'].k,
                                  'status': result['status'], 'reason': result.get('reason'),
                                  'error': result.get('error')}), flush=True)
                torch.cuda.empty_cache()
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    passed = sum(item['status'] == 'passed' for item in report['results'])
    failed = sum(item['status'] == 'failed' for item in report['results'])
    omitted = sum(item['status'] == 'skipped' for item in report['results'])
    report['summary'] = {'selected': len(selected), 'passed': passed, 'failed': failed,
                         'storage_skipped': omitted, 'unsupported_lines': len(skipped)}
    report['ok'] = 'error' not in report and passed > 0 and failed == 0
    report['coverage_complete'] = (report['ok'] and omitted == 0 and not skipped
                                   and len(selected) == len(shapes))
    report['finished_unix'] = time.time()
    save(args.out, report)
    if args.reference_out and not args.reference_out.exists() and report['ok']:
        save(args.reference_out, report)
        print(json.dumps({'created_stock_reference': str(args.reference_out)}), flush=True)
    print(json.dumps({'ok': report['ok'], 'summary': report['summary'],
                      'error': report.get('error')}), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
