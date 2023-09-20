//
// Created by ycy on 9/20/23.
//

#include "torch/library.h"
#include "torch/torch.h"
#include "torch/nn.h"
#include "iostream"

const uint32_t opt30b_num_attention_heads = 56;
const uint32_t opt30b_num_hidden_layers = 48;
const uint32_t opt30b_hidden_size = 7168;
const uint32_t opt30b_attn_emb_dims = opt30b_hidden_size;
const uint32_t opt30b_vocab_size = 50272;
const uint32_t opt30b_word_embed_proj_dim = opt30b_hidden_size;

bool check_cuda() {
    return torch::cuda::is_available() && torch::cuda::cudnn_is_available();
}

int main() {
    assert(check_cuda());
//    auto EmbLayer = torch::nn::Embedding(opt30b_vocab_size, opt30b_word_embed_proj_dim);
//    EmbLayer->to(torch::kCUDA);
//    torch::Tensor input = torch::tensor({1, 100, 1000}).pin_memory();
//    input = input.to(torch::kCUDA);
//    torch::Tensor output = EmbLayer->forward(input);
//    std::cout << output.sizes() << std::endl;


    auto Q = torch::randn({128, 1, opt30b_attn_emb_dims}).cuda();
    auto K = torch::randn({128, 1, opt30b_attn_emb_dims}).cuda();
    auto V = torch::randn({128, 1, opt30b_attn_emb_dims}).cuda();
    auto attn = torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(opt30b_attn_emb_dims, opt30b_num_attention_heads)
            .kdim(opt30b_attn_emb_dims));
    attn->to(torch::kCUDA);
    auto output = attn->forward(Q, K, V);
    return 0;
}