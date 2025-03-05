#include <torch/extension.h>

torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors element-wise");
}
