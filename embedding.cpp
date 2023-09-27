//
// Created by ycy on 9/20/23.
//

#include "pcm/utils.h"
#include "torch/library.h"
#include "torch/torch.h"
#include "torch/nn.h"
#include "iostream"
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <torch/cuda.h>

#include "pcie/pcie_monitor.h"

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

    auto monitor = PCIeMonitor::Instance();
    monitor->Clear();
    assert(monitor->EventAggr(PCIRdCur) == 0);
    monitor->StartGroup(0);
    torch::cuda::synchronize(0);
    auto EmbLayer = torch::nn::Embedding(opt30b_vocab_size, opt30b_word_embed_proj_dim);
    EmbLayer->to(torch::kCUDA);
    torch::Tensor input = torch::tensor({1, 100, 1000}).pin_memory();
    input = input.to(torch::kCUDA);
    torch::Tensor output = EmbLayer->forward(input);
    torch::cuda::synchronize(0);
    monitor->StopGroup(0);
    printf("ReadCount: %lu\n", monitor->GetReadAccessCounter());
    printf("ReadBW: %lu\n", monitor->GetReadBW());
    return 0;


    
    // monitor->Clear();
    // assert(monitor->EventAggr(PCIRdCur) == 0);
    // monitor->Start();
    torch::cuda::synchronize(0);
    auto Q = torch::randn({128, 1, opt30b_attn_emb_dims}).cuda();
    auto K = torch::randn({128, 1, opt30b_attn_emb_dims}).cuda();
    auto V = torch::randn({128, 1, opt30b_attn_emb_dims}).cuda();
    auto attn = torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(opt30b_attn_emb_dims, opt30b_num_attention_heads)
            .kdim(opt30b_attn_emb_dims));
    attn->to(torch::kCUDA);
    // auto output = attn->forward(Q, K, V);
    torch::cuda::synchronize(0);
    // MySleepMs(1000);
    // monitor->Stop();
    // printf("PCIRdCur: %lu\n", monitor->EventAggr(PCIRdCur));
    return 0;
}