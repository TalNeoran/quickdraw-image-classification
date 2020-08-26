from quickdraw import QuickDrawData, QuickDrawDataGroup
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import torchvision
from torch.utils import data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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


# def drawing_to_tensor(drawing, resize):
#     trans = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
#     return trans(drawing.image)

def load_data_by_label(label_name, num_samples):
    # print(f"Loading data for label: {label_name}")
    qdg = QuickDrawDataGroup(name=label_name, recognized=True, max_drawings=num_samples)
    data = torch.stack([transforms.ToTensor()(qdg.get_drawing(i).image.convert("L")) for i in range(num_samples)])
    return data

def load_data(label_names, label_index_list=None, num_labels=10, num_samples_per_label=1000, test_size=0.1, batch_size=64, num_workers=4):
    if label_index_list:
        X = torch.cat([load_data_by_label(label_names[label_index], num_samples_per_label) for label_index in label_index_list])
        y = torch.cat([torch.full((num_samples_per_label, 1), i, dtype=torch.float32) for i in label_index_list])
    else:
        X = torch.cat([load_data_by_label(label_names[i], num_samples_per_label) for i in range(num_labels)])
        y = torch.cat([torch.full((num_samples_per_label, 1), i, dtype=torch.float32) for i in range(num_labels)])

    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=test_size, stratify=y)
    X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test) 

    train_iter = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(data.TensorDataset(X_test, y_test), batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def show_images(imgs, num_rows, num_cols, titles=None):
    _, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()

def main(): 
    qgd = QuickDrawData()
    label_names = qgd.drawing_names
    total_num_labels = len(label_names)
    max_drawings = 1000
    im_res = 255


    batch_size=20
    labels_list = ["ant", "bear", "bee", "bird", "butterfly", "cat", "cow", "crab", "crocodile", "dog"]
    label_index_list = [label_names.index(x) for x in labels_list]
    train_iter, test_iter = load_data(label_names, label_index_list=label_index_list, batch_size=batch_size)

    for X, y in train_iter:
        show_images(X.reshape(batch_size, im_res, im_res), 2, 10, [label_names[int(i.item())] for i in y])
        break



if __name__ == "__main__":
    main()