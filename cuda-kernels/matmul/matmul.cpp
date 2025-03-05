#include <torch/extension.h>

void matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Multiply two matrices");
}
