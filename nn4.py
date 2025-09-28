import torch
from torch import nn
from torch import optim

torch.manual_seed(1337)

N = 10
D = 2
E = 1
MARGIN = 1

def differences(values):
    differences = values[:, None] - values
    return differences[*torch.triu_indices(*differences.shape[:2], offset=1)]


def distances(points):
    return (differences(points) ** 2).sum(-1)


points = torch.rand((N, D))

model = nn.Embedding(N, E)

optimizer = optim.SGD(model.parameters(), lr=1e-1)

while True:
    batch = points[:5]

    embeddings = model(torch.arange(N))

    ordering = torch.where(differences(distances(batch)) < 0, 1, -1)
    loss = (MARGIN + differences(distances(embeddings)) * ordering).relu().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())

