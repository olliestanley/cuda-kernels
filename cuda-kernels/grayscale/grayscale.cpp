#include <torch/extension.h>

void grayscale(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grayscale", &grayscale, "Convert to grayscale");
}
