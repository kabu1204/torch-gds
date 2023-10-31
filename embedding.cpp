//
// Created by ycy on 9/20/23.
//

#include "pcm/utils.h"
#include "torch/library.h"
#include "torch/torch.h"
#include "torch/nn.h"
#include "iostream"
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <thread>
#include <torch/cuda.h>
#include <torch/utils.h>

#include "pcie/pcie_monitor.h"
#include "torch_ext/table.h"

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
    torch::NoGradGuard no_grad;

    at::TensorOptions option;
    option = option.device(torch::kCUDA);
    auto monitor = PCIeMonitor::Instance();
    torch::cuda::synchronize(0);
    auto EmbLayer = torch::nn::Embedding(opt30b_vocab_size, opt30b_word_embed_proj_dim);
    // addModuleDHA(*EmbLayer->modules(true)[0]);
    // EmbLayer->to(torch::kCUDA);
    pinModuleDHA(*EmbLayer->modules(true)[0]);
    monitor->Clear();
    monitor->StartGroup(0);
    for (int i=0; i<1000; ++i) {
        torch::Tensor input = torch::tensor({int(rand() % opt30b_vocab_size)}, option);
        torch::Tensor output = EmbLayer->forward(input);
    }
    torch::cuda::synchronize(0);
    monitor->StopGroup(0);
    printf("%.2f ms/op\n", (double(monitor->GetDurationNs())/1e9));
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
    // addModuleDHA(*attn->modules(true)[0]);
    attn->to(torch::kCUDA);
    // attn->zerocopy_to();
    // 
    // auto output = attn->forward(Q, K, V);
    torch::cuda::synchronize(0);
    // torch::jit::
    // MySleepMs(1000);
    // monitor->Stop();
    // printf("PCIRdCur: %lu\n", monitor->EventAggr(PCIRdCur));

        // printf("emb to cuda, after %p, is_pinned=%d\n", 
        // EmbLayer->parameters(false)[0].data_ptr(),
        // is_pinned_cuda(EmbLayer->parameters(false)[0])
        // );
    return 0;
}