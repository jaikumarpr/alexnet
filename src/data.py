import datasets as ds
import torchvision.transforms.v2 as transforms
from helpers import to_rgb, HGDataset

# training dataset
__hg_train_dset = ds.load_dataset(
    'imagenet-1k', split='train', data_dir='./imagenet-1k')
# testing dataset
__hg_test_dset = ds.load_dataset(
    'imagenet-1k', split='test', data_dir='./imagenet-1k')
# validation dataset
__hg_valid_dset = ds.load_dataset(
    'imagenet-1k', split='validation', data_dir='./imagenet-1k')

image_transform = transforms.Compose([
    # dataset contains grayscale and gamma images, neeed to convert to rgb mode
    transforms.Lambda(to_rgb),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# convert hg dataset to torch datasets
__train_dataset = HGDataset(__hg_train_dset, transform=image_transform)
__test_dataset = HGDataset(__hg_test_dset, transform=image_transform)
__valid_dataset = HGDataset(__hg_valid_dset, transform=image_transform)


def dataset(set):
    if (set == 'train'):
        return __train_dataset
    elif (set == "validation"):
        return __valid_dataset
    elif (set == 'test'):
        return __test_dataset
