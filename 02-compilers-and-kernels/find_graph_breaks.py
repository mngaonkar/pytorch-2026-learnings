import torch

def f(x):
    x = x.sin()
    print("debug")        # Python side effect: causes a graph break
    return x.cos()

explanation = torch._dynamo.explain(f)(torch.randn(8))
print("graphs:", explanation.graph_count, "breaks:", explanation.graph_break_count)
for reason in explanation.break_reasons:
    print(reason.reason)
