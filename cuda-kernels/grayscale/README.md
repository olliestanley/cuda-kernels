A grayscale kernel building on the work of submissions to the GPU MODE grayscale leaderboard. This is mostly just for my own learning, there are faster submissions on the leaderboard including some writing inline PTX (!) so this is not the best by any means. Credit to gau.nernst on the GPU MODE server for the original code.

I added:

1. `__restrict__` to the input and output pointers
2. `#pragma unroll 4` to the loop that calculates the output pixel values
3. Manual `fmaf` calls to force fused multiply-adds

These optimisations speed up runtimes on my machine, particularly `__restrict__`. `fmaf` seemed to improve things noticeably too. `#pragma unroll 4` seems to make a tiny difference, if any. My GPU is just an RTX 3070, I haven't tested this on any other hardware. I am also unsure how this would vary with different image sizes.

I used the below benchmarking script:

```python
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import grayscale as custom_grayscale

num_iters = 1000
C, H, W = 3, 256, 256

# First confirm that we get the same output
rgb_torchvision = torch.rand(C, H, W).to("cuda")
gray_torchvision = rgb_to_grayscale(rgb_torchvision)
gray_custom = torch.empty(H, W, dtype=torch.float32).to("cuda")
rgb_custom = rgb_torchvision.permute(1, 2, 0).contiguous()
custom_grayscale.grayscale(rgb_custom, gray_custom)
assert torch.allclose(gray_torchvision, gray_custom)

# Benchmark custom
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
t0.record()
for _ in range(num_iters):
    custom_out = torch.empty(H, W, dtype=torch.float32).to("cuda")
    custom_grayscale.grayscale(rgb_torchvision, custom_out)
torch.cuda.synchronize()
t1.record()
print("Custom: ", t0.elapsed_time(t1) / num_iters)

# Benchmark torchvision
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
t0.record()
for _ in range(num_iters):
    rgb_to_grayscale(rgb_torchvision)
torch.cuda.synchronize()
t1.record()
print("Torchvision: ", t0.elapsed_time(t1) / num_iters)
```

Outputs on my machine:

```
Custom (before my tweaks): 0.2858499755859375
Custom (after my tweaks): 0.204087646484375
Torchvision: 0.42854196166992187
```
