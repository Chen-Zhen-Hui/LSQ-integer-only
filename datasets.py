from torchvision import datasets, transforms
from torch.utils.data import DistributedSampler
import torch
import os



def get_data_loader(dataset_type, img_size, train_batch_size, test_batch_size, distributed=False, world_size=None, rank=0):
    if dataset_type=='imagenet':
        data_root = '/pm981a/xuhr/dataset/ImageNet/'
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if img_size==224:
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif img_size==128:
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(144),
                    transforms.CenterCrop(128),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            raise ValueError('input size of imagenet should be 128 or 224')
        if distributed:
            train_sampler = DistributedSampler(train_dataset,
                                              num_replicas=world_size,
                                              rank=rank,
                                              )
            val_sampler = DistributedSampler(val_dataset, 
                                            num_replicas=world_size,
                                            rank=rank,
                                            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=test_batch_size,
            num_workers=4, pin_memory=True, sampler=val_sampler)
    elif dataset_type=='tiny-imagenet':
        data_root = "/pm981a/xuhr/dataset/tiny-imagenet-200/"
        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        train_data = datasets.ImageFolder(root=data_root+'/train', transform=transform['train'])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        val_data = datasets.ImageFolder(root=data_root+'/val', transform=transform['val'])
        test_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=False)
    elif dataset_type=='cifar10':
        data_root = '/barracuda_2T/chenzh/cifar10/'
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        trainset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        testset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,num_workers=8, pin_memory=True)
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
    return train_loader, test_loader

if __name__ == '__main__':
    import shutil
    import random

    # ImageNet验证集图片目录
    val_dir = '/hdd_raid5/xuhr/ImageNet/val'
    # 目标目录
    target_dir = 'imgs'

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 获取所有图片文件名
    all_imgs = []
    for root, dirs, files in os.walk(val_dir):
        for f in files:
            if f.lower().endswith('.jpeg'):
                all_imgs.append(os.path.join(root, f))

    print(f"val目录下图片数量: {len(all_imgs)}")

    if len(all_imgs) < 100:
        print(f"警告：val目录下只有{len(all_imgs)}张图片，全部复制。")
        sample_imgs = all_imgs
    else:
        sample_imgs = random.sample(all_imgs, 100)

    for img_path in sample_imgs:
        img_name = os.path.basename(img_path)
        dst = os.path.join(target_dir, img_name)
        shutil.copy(img_path, dst)

    print(f"已复制{len(sample_imgs)}张图片到imgs目录。")
