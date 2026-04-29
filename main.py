# Globular Reasoning Architecture
# Single-file wrapper using modular imports

import torch
import torch.nn.functional as F

from globular import (
    GlobularConfig,
    GlobularAgentField,
    GlobularReasoningBlock,
    ExampleTinyModelWithGlobularReasoning,
    RMSNorm,
    FeedForward,
    count_parameters,
)


# Backward compatibility aliases
GlobularReasoningAdapter = ExampleTinyModelWithGlobularReasoning


def demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ExampleTinyModelWithGlobularReasoning(
        vocab_size=8192,
        dim=256,
        depth=2,
        max_seq_len=128,
        num_agents=64,
        agent_dim=256,
    ).to(device)

    input_ids = torch.randint(0, 8192, (2, 128), device=device)
    logits, diagnostics = model(input_ids, return_diagnostics=True)

    print("Device:", device)
    print("Logits:", logits.shape)
    print("Parameters:", count_parameters(model))

    first_diag = diagnostics[0]
    print("Agent energy:", first_diag["agent_energy"].shape)
    print("Agent confidence:", first_diag["agent_confidence"].shape)
    print("Mean energy history:", first_diag["mean_energy_history"].shape)

    targets = torch.randint(0, 8192, (2, 128), device=device)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    print("Loss:", float(loss.detach()))


if __name__ == "__main__":
    demo()