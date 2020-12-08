import albumentations as A


def get_valid_transforms():
    return A.Compose(
        [
            A.Normalize()
        ],
        p=1.0)


def light_training_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.Normalize()
    ])


def medium_training_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
        A.Normalize()
    ])


def heavy_training_transforms():
    return A.Compose([
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.CLAHE(),
                A.RGBShift(),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.HueSaturationValue(),
                A.NoOp()
            ], p=0.15),
        A.OneOf(
            [
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
                A.NoOp(),
                A.ShiftScaleRotate(),
            ], p=1.0),
        A.OneOf(
            [
                A.GaussNoise(),
                A.GaussianBlur(),
                A.NoOp()
            ], p=0.15),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
        A.Normalize()
    ])


def get_valid_transforms_xray(crop_size=256):
    return A.Compose(
        [
            A.Resize(crop_size, crop_size),
        ],
        p=1.0)


def light_training_transforms_xray(crop_size=256):
    return A.Compose([
        A.RandomResizedCrop(height=crop_size, width=crop_size),
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
    ])


def medium_training_transforms_xray(crop_size=256):
    return A.Compose([
        A.RandomResizedCrop(height=crop_size, width=crop_size),
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
    ])


def heavy_training_transforms_xray(crop_size=256):
    return A.Compose([
        A.RandomResizedCrop(height=crop_size, width=crop_size),
        A.OneOf(
            [
                A.Transpose(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.RandomRotate90(),
                A.NoOp()
            ], p=1.0),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf(
            [
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                A.NoOp()
            ], p=1.0),
    ])


def get_training_trasnforms(transforms_type):
    if transforms_type == 'light':
        return(light_training_transforms())
    elif transforms_type == 'medium':
        return(medium_training_transforms())
    elif transforms_type == 'heavy':
        return(heavy_training_transforms())
    else:
        raise NotImplementedError("Not implemented transformation configuration")


def get_training_trasnforms_xray(transforms_type):
    if transforms_type == 'light':
        return(light_training_transforms_xray())
    elif transforms_type == 'medium':
        return(medium_training_transforms_xray())
    elif transforms_type == 'heavy':
        return(heavy_training_transforms_xray())
    else:
        raise NotImplementedError("Not implemented transformation configuration")
