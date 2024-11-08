#include <torch/extension.h>

std::vector<torch::Tensor> forward(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V, int version);

std::vector<torch::Tensor> backward(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor V, torch::Tensor O,
                                    torch::Tensor dO, torch::Tensor l,
                                    torch::Tensor m, int version);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", torch::wrap_pybind_function(forward), "forward");
  m.def("forward", torch::wrap_pybind_function(backward), "backward");
}
