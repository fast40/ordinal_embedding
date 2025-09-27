import time

import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1337)

N = 3
D = 2
E = 1

model = nn.Sequential(
    nn.Linear(D, D * 4),
    nn.LeakyReLU(),
    nn.Linear(D * 4, D),
    nn.LeakyReLU(),
    nn.Linear(D, E)
)

optimizer = optim.SGD(model.parameters(), lr=1e-2)

train_set = torch.randn((N, D))

def has_nan_params(model):
    return any(torch.isnan(p).any().item() for p in model.parameters())

plt.ion()
fig, axes = plt.subplots(2, 4)
# plt.show()


colors = [
    "red", "green", "blue", "orange", "purple",
    "cyan", "magenta", "yellow", "brown", "pink",
    "gray", "olive", "teal", "navy", "maroon",
    "lime", "indigo", "gold", "coral", "turquoise"
    ][:N]

print(train_set.shape)
axes[1][0].scatter(*train_set.transpose(0, 1), c=colors)






while True:
    embeddings = model(train_set)

    axes[1][1].cla()
    axes[1][1].scatter(embeddings.detach(), torch.zeros((N,)), c=colors)

    # subtract to get side lengths, then square the side lengths, then sum them, then sqrt them
    train_set_distances = ((train_set[:, None] - train_set) ** 2).sum(dim=-1)
    embeddings_distances = ((embeddings[:, None] - embeddings) ** 2).sum(dim=-1)

    # just take the lower triangular part of the matrix (not including diagonal since it's always zero) since this is a symmetrical matrix
    train_set_distances = train_set_distances[*torch.tril_indices(N, N, offset=-1)]
    embeddings_distances = embeddings_distances[*torch.tril_indices(N, N, offset=-1)]

    # get the differences between pairs of distances
    train_set_pairs = train_set_distances[:, None] - train_set_distances
    embeddings_pairs = embeddings_distances[:, None] - embeddings_distances

    # print(embeddings_pairs)
    axes[0][0].imshow(train_set_pairs.detach() == 0)
    axes[0][1].imshow(embeddings_pairs.detach() == 0)

    train_set_pairs = train_set_pairs[*torch.tril_indices(N, N, offset=-1)]
    embeddings_pairs = embeddings_pairs[*torch.tril_indices(N, N, offset=-1)]

    # if distance2 is larger, then this is going to be negative.
    # fundamentally, we want the distance pairings to match. What this means is that if it is "correct" we want distance2 to be larger.

    # if we want distance2 larger than distance1, we keep as is. 1.
    # if we want distance2 to be smaller, than we have to invert. -1.
    v = torch.where(train_set_pairs < 0, 1, -1)
    loss = (0.5 + (embeddings_pairs * v)).relu().mean()
    # this is correct if v is 1 if we want distance 1 to be larger and -1 if we want distance2 to be larger.




    # loss_values = torch.maximum(embeddings_distances[:, None], embeddings_distances) - torch.minimum(embeddings_distances[:, None], embeddings_distances)
    # axes[0][2].imshow(loss_values.detach())
    #
    # # check where the signs are the same (we want to EXCLUDE these in the loss because they are correctly ordered)
    # mask = train_set_pairs.sign() == embeddings_pairs.sign()
    #
    # # EXCLUDE the values where the mask is true
    # loss_values[mask] = 0
    #
    # axes[0][3].imshow(loss_values.detach())
    # # plt.show(block=False)
    plt.pause(0.001)
    #
    # loss = loss_values.mean()
    #
    embeddings.retain_grad()
    optimizer.zero_grad()
    loss.backward()
    print(loss)
    print(embeddings)
    print(embeddings.grad)
    time.sleep(1)
    optimizer.step()
