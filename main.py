from quickdraw import QuickDrawData, QuickDrawDataGroup
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import torchvision
from torch.utils import data
import torch.nn.functional as F

def test_anvil():
    qd = QuickDrawData()
    anvil = qd.get_drawing("anvil")
    # anvil.image.show()
    trans = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    anvil_tensor = transforms.ToTensor()(anvil.image)
    down_sized_anvil = trans(anvil.image)
    plt.imshow(anvil_tensor.permute(1, 2, 0))
    plt.show()
    plt.imshow(down_sized_anvil.permute(1, 2, 0));
    plt.show()


def drawing_to_tensor(drawing):
    trans = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    return trans(drawing.image)

if __name__ == "__main__":
    # batch_size = 256
    # trans = transforms.ToTensor()
    # train_iter = data.DataLoader(torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True), batch_size, shuffle=True, num_workers=4)
    # X, y = next(iter(train_iter))
    # print(X.shape, y.shape)
    pass

qgd = QuickDrawData()
label_names = qgd.drawing_names
num_labels = len(label_names)
print(num_labels)
max_drawings = 1000
anvils = QuickDrawDataGroup("anvil", max_drawings=max_drawings)
drawings = torch.cat([drawing_to_tensor(anvils.get_drawing(i)).reshape(1,3,64,64) for i in range(max_drawings)])
plt.imshow(drawings[2].permute(1, 2, 0))
plt.show()
