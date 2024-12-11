import albumentations as albu
import cv2


def get_transforms(opt):
    max_pixel_value = 1.0
    mean = [0.5]
    std = [0.5]

    train_ops = []
    if 'hflip' in opt.augs:
        train_ops.append(albu.HorizontalFlip(p=0.5))
    if 'vflip' in opt.augs:
        train_ops.append(albu.VerticalFlip(p=0.1))
    if 'shift_scale_rotate' in opt.augs:
        train_ops.append(albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.07, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))

    data_transforms = {
        'train': albu.Compose(
            [*train_ops, albu.Resize(*opt.img_size, cv2.INTER_LINEAR), albu.ToFloat(), albu.Normalize(mean, std, max_pixel_value=max_pixel_value)]
        ),
        'val': albu.Compose([albu.Resize(*opt.img_size, cv2.INTER_LINEAR), albu.ToFloat(), albu.Normalize(mean, std, max_pixel_value=max_pixel_value)]),
    }
    print('Note: Ensure the image range is in [-1, 1] after transforms(Aug)')
    return data_transforms
