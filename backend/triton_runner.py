import torch
import triton
import triton.language as tl
import tempfile
import importlib.util
import os
import uuid

def run_vector_add_kernel(code_str: str, size: int):
    x = torch.arange(size, dtype=torch.float32, device='cuda')
    y = torch.ones(size, dtype=torch.float32, device='cuda')
    z = torch.empty_like(x)

    # 建立暫存 .py 檔案
    temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex[:8]
    filename = f"user_kernel_{unique_id}.py"
    filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(code_str)

    try:
        # 載入該 module
        spec = importlib.util.spec_from_file_location("user_kernel", filepath)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)

        # 從 module 中取得 kernel function
        kernel_fn = getattr(user_module, "add_kernel")

        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        kernel_fn[grid](x, y, z, size, BLOCK_SIZE)

        return z.cpu()
    finally:
        os.remove(filepath)
