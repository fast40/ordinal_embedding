'''
possible idea: add more randomness into this by not including every distance in the loss. might help computationally and even to find ideal embeddings.
sometimes going in the wrong direction is a good idea.
'''


import torch
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

torch.manual_seed(1337)

N = 10

colors = torch.tensor([0, 1, 0.9]).repeat(N, 1)
colors[:, 0] = (torch.arange(N) / (N*1.2))
colors = hsv_to_rgb(colors)

fig, axes = plt.subplots(2, 2)

points = torch.rand((N, 2))


axes[0, 0].scatter(*points.transpose(0, 1), c=colors)
# axes[0, 1].imshow(distances)
# axes[1, 0].imshow(distance_differences)



def differences(values):
    differences = values[:, None] - values
    return differences[*torch.tril_indices(*differences.shape[:2], offset=-1)]

distances = (differences(points) ** 2).sum(-1)

distance_differences = differences(distances)

plt.show()
# distance_differences[*torch.triu_indices(*distance_differences.shape, offset=-1)] = 1
