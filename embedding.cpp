//
// Created by ycy on 9/20/23.
//

#include "torch/library.h"
#include "torch/torch.h"
#include "torch/nn.h"
#include "iostream"

const uint32_t opt30b_vocab_size = 50272;
const uint32_t opt30b_word_embed_proj_dim = 7168;

bool check_cuda() {
    return torch::cuda::is_available() && torch::cuda::cudnn_is_available();
}

int main() {
    assert(check_cuda());
    auto EmbLayer = torch::nn::Embedding(opt30b_vocab_size, opt30b_word_embed_proj_dim);
    torch::Tensor input = torch::tensor({1, 100, 1000});
    torch::Tensor output = EmbLayer->forward(input);
    std::cout << output.sizes() << std::endl;
    return 0;
}