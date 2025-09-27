import torch
from matplotlib import pyplot as plt

torch.manual_seed(1337)

N = 10

t1 = torch.randn((N, 2))
distances1 = ((t1.view(-1, 1, 2) - t1) ** 2).sum(dim=2).sqrt()


t2 = torch.randn((N, 2))
distances2 = ((t2.view(-1, 1, 2) - t2) ** 2).sum(dim=2).sqrt()

i, j = torch.tril_indices(N, N, offset=-1)

distances = distances1[*torch.tril_indices(N, N, offset=-1)]

d = distances.view(-1, 1) - distances
d = d.relu()

print(d.shape)
print(distances.shape)

print(d.sum())



fig, axes = plt.subplots(2, 2)

axes[0][0].imshow(distances1)
axes[0][1].imshow(distances2)
axes[1][0].imshow(distances2 - distances1)

axes[1][1].imshow(d)

# axes[1][1].scatter(*t1.transpose(0, 1))
plt.show()
