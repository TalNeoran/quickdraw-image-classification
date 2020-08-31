from quickdraw import QuickDrawData, QuickDrawDataGroup
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F
from torch import nn
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

def load_data(label_list, num_samples_per_label=1000, test_size=0.1, batch_size=64, num_workers=4):
    # if label_index_list:
    #     X = torch.cat([load_data_by_label(label_names[label_index], num_samples_per_label) for label_index in label_index_list])
    #     y = torch.cat([torch.full((num_samples_per_label, 1), i, dtype=torch.float32) for i in label_index_list])
    # else:
    #     X = torch.cat([load_data_by_label(label_names[i], num_samples_per_label) for i in range(num_labels)])
    #     y = torch.cat([torch.full((num_samples_per_label, 1), i, dtype=torch.float32) for i in range(num_labels)])
    X = torch.cat([load_data_by_label(label, num_samples_per_label) for label in label_list])
    y = torch.cat([torch.full((num_samples_per_label, 1), i, dtype=torch.int64) for i in range(len(label_list))]).reshape(-1)

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
    im_res = 255


    batch_size = 100
    label_list = ["ant", "bear", "bee", "bird", "butterfly", "cat", "cow", "crab", "crocodile", "dog"]
    train_iter, test_iter = load_data(label_list=label_list, batch_size=batch_size)


    # Training

    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(-1,65025)


    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(65025, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    device = torch.device("cuda:0")
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss = nn.CrossEntropyLoss()
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss_sum = train_num_correct_preds = num_examples = 0
        for i, (X, y) in enumerate(train_iter):
            # show_images(X.reshape(batch_size, im_res, im_res), 2, 10, [label_list[int(i.item())] for i in y])
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            # print(f"loss={l:.3f}", y_hat, y, X.shape, X.reshape(-1, 65025).shape)
            # break
            l.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += l * X.shape[0]
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat.argmax(axis=1)    
                cmp = y_hat.type(y.dtype) == y
                train_num_correct_preds += float(cmp.type(y.dtype).sum())
                num_examples += X.shape[0]
            train_loss = train_loss_sum / num_examples
            train_acc = train_num_correct_preds / num_examples
            if (i + 1) % 10 == 0:
                print(f"epoch {epoch + 1}, iteration {i + 1}: train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")

        test_acc = eval_test_acc(net, test_iter, device)
        print(f"epoch {epoch + 1}: test_acc={test_acc:.3f}")
    print(f"Finished training: train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")


def eval_test_acc(net, data_iter, device):
    net.eval()
    test_num_correct_preds = num_examples = 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)    
        cmp = y_hat.type(y.dtype) == y
        test_num_correct_preds += float(cmp.type(y.dtype).sum())
        num_examples += X.shape[0]

    return test_num_correct_preds / num_examples



if __name__ == "__main__":
    main()