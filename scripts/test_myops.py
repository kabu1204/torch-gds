import torch
# torch.ops.load_library("build/torch_ext/libmycopy.so")
to_dha = torch.ops.dha.to_dha
add_dha = torch.ops.dha.add_dha

def add_dha_module(m: torch.nn.Module):
    with torch.no_grad():
        for name, param in m.named_parameters():
            add_dha(param.data)

        for name, param in m.named_buffers():
            add_dha(param.data)

def to_dha_module(m: torch.nn.Module):
    with torch.no_grad():
        for name, param in m.named_parameters():
            param.set_(to_dha(param.data))

        for name, param in m.named_buffers():
            param.set_(to_dha(param.data))

def jit_compute(x):
    # add_dha(x)
    return x+1

def jit_test():
    with torch.no_grad():
        trace = torch.jit.trace(jit_compute, torch.rand(4,4))
        print(trace.graph)

if __name__=="__main__":
    jit_test()
    with torch.no_grad():
        # a = torch.tensor([0, 50, 99])
        emb = torch.nn.Embedding(100, 8)
        emb = emb.to("cuda:0")
        # to_dha_module(emb)
        for i in range(1):
            a = torch.tensor([0, 50, 99]).long()
            a = a.cuda()
            b = emb.forward(a)
            print(b)
        print("ok")