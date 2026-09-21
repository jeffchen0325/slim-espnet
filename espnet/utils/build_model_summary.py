from typing import Dict
from humanfriendly import format_number, format_size


def build_model_summary(model) -> Dict[str, object]:
    """Build a static model summary for logs or publication metadata.

    Args:
        model: PyTorch model instance to summarize.

    Returns:
        Dictionary with model class, parameter counts, buffer counts, dtype
        composition, formatted display strings, and ``repr(model)``.

    Notes:
        This helper only inspects the instantiated module. It does not run a
        forward pass, so no example batch or shape inference is required.

    Examples:
        ```python
        summary = build_model_summary(model)
        print(summary["total_params_display"])
        ```
    """
    params = list(model.parameters())
    buffers = list(model.buffers())

    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    size_bytes = sum(p.numel() * p.element_size() for p in params)
    total_buffers = sum(buf.numel() for buf in buffers)
    buffer_size_bytes = sum(buf.numel() * buf.element_size() for buf in buffers)
    module_count = sum(1 for _ in model.modules())
    leaf_module_count = sum(
        1 for module in model.modules() if not any(module.children())
    )

    dtype_counts: Dict[str, int] = {}
    for tensor in [*params, *buffers]:
        dtype = str(tensor.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + tensor.numel()
    dtype_items = sorted(dtype_counts.items(), key=lambda kv: kv[1], reverse=True)
    dtype_desc = ", ".join(
        f"{dtype}({count / total_params * 100:.1f}%)" if total_params else dtype
        for dtype, count in dtype_items
    )

    return {
        "class_name": type(model).__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "trainable_ratio": (
            trainable_params / total_params * 100.0 if total_params else 0.0
        ),
        "size_bytes": size_bytes,
        "total_buffers": total_buffers,
        "buffer_size_bytes": buffer_size_bytes,
        "module_count": module_count,
        "leaf_module_count": leaf_module_count,
        "dtype_desc": dtype_desc or "None",
        "repr": repr(model),
        "total_params_display": format_number(total_params),
        "trainable_params_display": format_number(trainable_params),
        "non_trainable_params_display": format_number(non_trainable_params),
        "size_display": format_size(size_bytes),
        "total_buffers_display": format_number(total_buffers),
        "buffer_size_display": format_size(buffer_size_bytes),
        "module_count_display": format_number(module_count),
        "leaf_module_count_display": format_number(leaf_module_count),
    }

