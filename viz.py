import uuid
import pathlib
import time

import torch
from torch import nn

torch.manual_seed(1337)

from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['figure.raise_window'] = False

N = 100
E = 2


plt.ion()
fig, ax = plt.subplots(2, 2)

colors = torch.rand((100, 3))

scatter1 = ax[0, 0].scatter(torch.arange(100), torch.rand(100), c=colors)
scatter2 = ax[0, 1].scatter(torch.arange(100), torch.rand(100), c=colors)
scatter3 = ax[1, 1].scatter(torch.arange(100), torch.rand(100), c=colors)
scatter4 = ax[1, 0].scatter(torch.arange(100), torch.rand(100), c=colors)

ax[0, 0].set_xlim(-5, 5)
ax[0, 0].set_ylim(-5, 5)

ax[0, 1].set_xlim(-5, 5)
ax[0, 1].set_ylim(-5, 5)

ax[1, 1].set_xlim(-5, 5)
ax[1, 1].set_ylim(-5, 5)

ax[1, 0].set_xlim(-5, 5)
ax[1, 0].set_ylim(-5, 5)



def graph_embedding(dir_name, scatter):
    models_dir = pathlib.Path(dir_name)

    try:
        file = sorted(models_dir.glob('*.pt'), reverse=True)[0]
    except IndexError:
        pass

    model = nn.Embedding(N, 2)
    model.load_state_dict(torch.load(file))
    scatter.set_offsets(model(torch.arange(N)).detach())

while True:
    # x = torch.stack((torch.arange(10), torch.rand(10) * 10), dim=1)
    graph_embedding('model', scatter1)
    graph_embedding('model2', scatter2)
    graph_embedding('model3', scatter3)
    graph_embedding('model4', scatter4)
    plt.pause(0.001)
    time.sleep(0.1)
    print(uuid.uuid4())
