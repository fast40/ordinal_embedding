import uuid
import pathlib

import torch
from torch import nn
from torch import optim
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image

torch.manual_seed(1337)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 100
D = 3 * 128 * 128
E = 2

MARGIN = 2
FOLDER = 'model4'


def differences(values):
    differences = values[:, None] - values
    return differences[*torch.triu_indices(*differences.shape[:2], offset=1)]


def distances(points):
    return (differences(points) ** 2).sum(-1)


transform = Compose([CenterCrop(128), ToTensor()])
points = torch.stack([transform(Image.open(f'celeba/{i+1:06}.jpg')).flatten() for i in range(N)])
points = points.to(DEVICE)
ordering = torch.where(differences(distances(points)) < 0, 1, -1)
print(points.shape)

model = nn.Embedding(N, E)
model.to(DEVICE)

# optimizer = optim.SGD(model.parameters(), lr=1e1)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

i = 1

while True:
    embeddings = model(torch.arange(N, device=DEVICE))

    if i % 1 == 0:
        # print('saved')
        temp = pathlib.Path(f'{FOLDER}/{uuid.uuid4()}.pt')
        with open(temp, 'wb') as file:
            torch.save(model.state_dict(), file)
        temp.replace(f'{FOLDER}/model_{i:06}.pt')

    loss = (MARGIN + differences(distances(embeddings)) * ordering).relu().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())

    i += 1
