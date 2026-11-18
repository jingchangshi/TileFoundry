"""TensorProgramVm operator modules and Ascend 950PR Striped Register VM kernels."""

from examples.ascend950pr_ops.tensor_program_vm.kernels import kernel, tensor_program_vm

__all__ = ["kernel", "tensor_program_vm"]
