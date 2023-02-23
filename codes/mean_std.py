import os
import torch
import torchvision
import torchvision.transforms as transforms

train_data_path = 'D:\\Gachon study\\Masters\\Masters\\1st sem\\Computer Vision\\Osteo\\test'

train_tranforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_tranforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)


def get_mean_std(loader):
    mean = 0.
    std = 0.
    total_image_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_image_count += image_count_in_a_batch
        mean /= total_image_count
        std /= total_image_count

        return mean, std


r=get_mean_std(train_loader)
print(r)
