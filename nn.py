import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1337)

N = 20
D = 2
E = 1
epsilon = 1e-10  # dogshit over here

model = nn.Sequential(
    nn.Linear(D, D * 4),
    nn.ReLU(),
    nn.Linear(D * 4, D),
    nn.ReLU(),
    nn.Linear(D, E),
    nn.Tanh()
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_set = torch.randn((N, D))

def has_nan_params(model):
    return any(torch.isnan(p).any().item() for p in model.parameters())

plt.ion()
fig, axes = plt.subplots(2, 4)
plt.show()


colors = [
    "red", "green", "blue", "orange", "purple",
    "cyan", "magenta", "yellow", "brown", "pink",
    "gray", "olive", "teal", "navy", "maroon",
    "lime", "indigo", "gold", "coral", "turquoise"
]

axes[1][0].scatter(train_set)
print(train_set.)



while True:
    # print(has_nan_params(model))
    # for parameter in model.parameters():
    #     print(parameter)
    # print(torch.any(torch.isnan(model.parameters())))
    embeddings = model(train_set)

    # subtract to get side lengths, then square the side lengths, then sum them, then sqrt them
    train_set_distances = ((train_set[:, None] - train_set) ** 2).sum(dim=-1)
    embeddings_distances = ((embeddings[:, None] - embeddings) ** 2).sum(dim=-1)
    # print(embeddings_distances.flatten().sort())
    # print(torch.any(torch.isnan(embeddings_distances)))
    #
    # print(embeddings)

    # these really should be symmetrical matrices
    assert torch.allclose(train_set_distances, train_set_distances.T)
    assert torch.allclose(embeddings_distances, embeddings_distances.T)

    # just take the lower triangular part of the matrix (not including diagonal since it's always zero) since this is a symmetrical matrix
    train_set_distances = train_set_distances[*torch.tril_indices(N, N, offset=-1)]
    embeddings_distances = embeddings_distances[*torch.tril_indices(N, N, offset=-1)]

    assert not torch.any(train_set_distances < 0)
    assert not torch.any(embeddings_distances < 0)

    # to avoid division by zero later. kind of a hack; not sure what to do about this.
    train_set_distances = train_set_distances + epsilon
    embeddings_distances = embeddings_distances + epsilon

    assert not torch.any(train_set_distances < 0)
    assert not torch.any(embeddings_distances < 0)

    # get the differences between pairs of distances. we want to check the signs to see if various pairs of distances are flipped in terms of ordering.
    train_set_pairs = train_set_distances[:, None] - train_set_distances
    embeddings_pairs = embeddings_distances[:, None] - embeddings_distances
    # print(embeddings_pairs)
    axes[0].imshow(train_set_pairs.detach() == 0)
    axes[1].imshow(embeddings_pairs.detach() == 0)

    loss_values = torch.maximum(embeddings_distances[:, None], embeddings_distances) / torch.minimum(embeddings_distances[:, None], embeddings_distances)
    axes[2].imshow(loss_values.detach())

    assert not torch.any(loss_values < 0)

    # check where the signs are the same (we want to EXCLUDE these in the loss because they are correctly ordered)
    mask = train_set_pairs.sign() == embeddings_pairs.sign()

    # EXCLUDE the values where the mask is true
    loss_values[mask] = 0

    axes[3].imshow(loss_values.detach())
    # plt.show(block=False)
    plt.pause(0.01)

    loss_values[mask]

    loss = loss_values.mean()



    optimizer.zero_grad()
    loss.backward()
    print(loss)
    optimizer.step()
