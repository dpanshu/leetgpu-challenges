#!/usr/bin/env python3
import argparse
import contextlib
import importlib.util
import py_compile
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_DIR = ROOT / "challenges"

MATRIX_CHALLENGES = [
    "easy/1_vector_add",
    "easy/2_matrix_multiplication",
    "easy/3_matrix_transpose",
    "easy/8_matrix_addition",
    "easy/31_matrix_copy",
    "medium/10_2d_convolution",
    "medium/11_3d_convolution",
    "medium/17_dot_product",
    "medium/22_gemm",
    "medium/28_gaussian_blur",
    "medium/30_batched_matrix_multiplication",
    "medium/42_2d_max_pooling",
    "medium/57_fp16_batched_matmul",
    "medium/58_fp16_dot_product",
    "hard/12_multi_head_attention",
    "hard/14_multi_agent_sim",
    "hard/15_sorting",
    "hard/20_kmeans_clustering",
    "hard/36_radix_sort",
    "hard/39_Fast_Fourier_transform",
    "hard/46_bfs_shortest_path",
    "hard/53_casual_attention",
    "hard/56_linear_attention",
    "hard/59_sliding_window_attn",
    "hard/73_all_pairs_shortest_paths",
    "hard/74_gpt2_block",
]

PYTHON_FRAMEWORKS = ("pytorch", "jax", "triton", "mlx")
CPU_RUNTIME_FRAMEWORKS = ("pytorch",)
MAX_CASE_ELEMENTS = 250_000
OPTIONAL_FRAMEWORKS = {
    "medium/10_2d_convolution": {"mlx"},
    "medium/11_3d_convolution": {"mlx"},
    "medium/28_gaussian_blur": {"mlx"},
    "medium/42_2d_max_pooling": {"mlx"},
    "hard/12_multi_head_attention": {"mlx"},
    "hard/14_multi_agent_sim": {"mlx"},
    "hard/15_sorting": {"mlx"},
    "hard/20_kmeans_clustering": {"mlx"},
    "hard/36_radix_sort": {"mlx"},
    "hard/39_Fast_Fourier_transform": {"mlx"},
    "hard/46_bfs_shortest_path": {"mlx"},
    "hard/53_casual_attention": {"mlx"},
    "hard/56_linear_attention": {"mlx"},
    "hard/59_sliding_window_attn": {"mlx"},
    "hard/73_all_pairs_shortest_paths": {"mlx"},
    "hard/74_gpt2_block": {"mlx"},
}
RUNTIME_SKIP = {
    "hard/20_kmeans_clustering",
    "hard/73_all_pairs_shortest_paths",
    "hard/74_gpt2_block",
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def remap_device(kwargs):
    if kwargs.get("device") == "cuda":
        kwargs = dict(kwargs)
        kwargs["device"] = "cpu"
    return kwargs


@contextlib.contextmanager
def cpu_tensor_factories():
    patched = {}
    factory_names = [
        "tensor",
        "empty",
        "zeros",
        "ones",
        "full",
        "randn",
        "rand",
        "randint",
        "arange",
        "eye",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
    ]

    def wrap(fn):
        def inner(*args, **kwargs):
            return fn(*args, **remap_device(kwargs))

        return inner

    for name in factory_names:
        if hasattr(torch, name):
            patched[name] = getattr(torch, name)
            setattr(torch, name, wrap(patched[name]))
    try:
        yield
    finally:
        for name, fn in patched.items():
            setattr(torch, name, fn)


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def clone_case(case):
    return {key: clone_value(value) for key, value in case.items()}


def total_tensor_elements(case) -> int:
    total = 0
    for value in case.values():
        if isinstance(value, torch.Tensor):
            total += value.numel()
    return total


def select_cases(challenge):
    with cpu_tensor_factories():
        example = challenge.generate_example_test()
        functional = challenge.generate_functional_test()
    selected = [example]
    for case in functional:
        if total_tensor_elements(case) <= MAX_CASE_ELEMENTS:
            selected.append(case)
        if len(selected) >= 5:
            break
    return selected


def framework_solution_path(challenge_dir: Path, framework: str) -> Path | None:
    path = challenge_dir / "solution" / f"solution.{framework}.py"
    if path.exists():
        return path
    legacy = challenge_dir / "solution" / "solution.py"
    if framework in {"pytorch", "jax", "triton"} and legacy.exists():
        return legacy
    return None


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    if actual.dtype != expected.dtype:
        actual = actual.to(expected.dtype)
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = (actual - expected).abs().max().item()
        raise AssertionError(f"tensor mismatch: max_abs={max_abs}")


def verify_pytorch_solution(challenge_dir: Path):
    challenge_module = load_module(
        f"challenge_{challenge_dir.name}", challenge_dir / "challenge.py"
    )
    challenge = challenge_module.Challenge()
    solution_path = framework_solution_path(challenge_dir, "pytorch")
    if solution_path is None:
        raise FileNotFoundError(f"missing PyTorch solution for {challenge_dir}")
    solution_module = load_module(f"solution_{challenge_dir.name}", solution_path)

    signature = challenge.get_solve_signature()
    output_names = [name for name, (_, direction) in signature.items() if direction in {"out", "inout"}]
    cases = select_cases(challenge)
    for index, base_case in enumerate(cases, start=1):
        expected_case = clone_case(base_case)
        actual_case = clone_case(base_case)
        challenge.reference_impl(**expected_case)
        solution_module.solve(**actual_case)
        for output_name in output_names:
            compare_tensors(
                actual_case[output_name],
                expected_case[output_name],
                atol=challenge.atol,
                rtol=challenge.rtol,
            )
        print(f"verified pytorch {challenge_dir} case {index}")


def compile_solution(solution_path: Path):
    py_compile.compile(str(solution_path), doraise=True)
    print(f"compiled {solution_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=list(PYTHON_FRAMEWORKS),
        choices=list(PYTHON_FRAMEWORKS),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(CHALLENGES_DIR))
    try:
        failures = []
        for rel_path in MATRIX_CHALLENGES:
            challenge_dir = CHALLENGES_DIR / rel_path
            optional_frameworks = OPTIONAL_FRAMEWORKS.get(rel_path, set())
            for framework in args.frameworks:
                solution_path = framework_solution_path(challenge_dir, framework)
                if solution_path is None:
                    message = f"missing {framework} solution for {rel_path}"
                    if args.strict and framework not in optional_frameworks:
                        failures.append(message)
                    else:
                        print(f"skip {message}")
                    continue
                try:
                    compile_solution(solution_path)
                    if framework in CPU_RUNTIME_FRAMEWORKS and rel_path not in RUNTIME_SKIP:
                        verify_pytorch_solution(challenge_dir)
                except Exception as exc:
                    failures.append(f"{rel_path} [{framework}] {exc}")
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        return 0
    finally:
        sys.path.pop(0)


if __name__ == "__main__":
    raise SystemExit(main())
