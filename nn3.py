'''
possible idea: add more randomness into this by not including every distance in the loss. might help computationally and even to find ideal embeddings.
sometimes going in the wrong direction is a good idea.
'''


import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

torch.manual_seed(1337)

N = 10
D = 2
E = 1

colors = torch.tensor([0, 1, 0.9]).repeat(N, 1)
colors[:, 0] = (torch.arange(N) / (N*1.2))
colors = hsv_to_rgb(colors)

fig, axes = plt.subplots(2, 2)

plt.ion()

points = torch.rand((N, D))


axes[0, 0].scatter(*points.transpose(0, 1), c=colors)
# axes[0, 1].imshow(distances)
# axes[1, 0].imshow(distance_differences)



def differences(values):
    differences = values[:, None] - values
    return differences[*torch.triu_indices(*differences.shape[:2], offset=1)]


def distances(points):
    return (differences(points) ** 2).sum(-1)


points_differences = differences(distances(points))

# we want each difference to be negative. That way it won't contribute to the loss. So if if isn't negative in the ground truth, flip the calc so that negative in the calc is good.
direction_corrections = torch.where(points_differences < 0, 1, -1)


model = nn.Embedding(N, E)

optimizer = optim.SGD(model.parameters(), lr=1e-1)

scatter = axes[0, 1].scatter(torch.arange(N), model(torch.arange(N)).detach(), c=colors)

plt.show(block=False)

while True:
    embeddings = model(torch.arange(N))


    # axes[0, 1].cla()
    # axes[0, 1].scatter(embeddings.detach(), torch.zeros((N,)), c=colors)
    x = embeddings.detach().view(-1).numpy()
    y = torch.zeros(N).numpy()
    scatter.set_offsets(np.c_[x, y])
    axes[0, 1].update_datalim(np.c_[x, y])
    axes[0, 1].autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

    loss = (1 + differences(distances(embeddings)) * direction_corrections).relu().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
    # print(embeddings)
    # print(embeddings)
