import os
import torch
import random
import time
import yaml
import json
import cv2
import numpy as np
import torchvision

def load_configs(configs):
    with open(configs, 'r') as stream:
        try:
            x = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return x


project_name = os.path.basename(os.getcwd())


def hint(msg):
    timestamp = f'{time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))}'
    print('\033[1m' + project_name + ' >> ' + timestamp + ' >> ' + '\033[0m' + msg)


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]
    cuda = all(gpu >= 0 for gpu in gpus)
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        hint('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        hint('Launching on CPU')
    return cuda


def make_reproducible(iscuda, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if iscuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def mean(l):
    return sum(l) / len(l)


def eps(x):
    return x + 1e-8


def move_dict_to_device(dict, device, tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def color_jitter(images_array_list, brightness = 0.4, contrast = 0.4, saturation = 0.2, hue = 0.127):
    h, w, c = images_array_list[0].shape
    cj_module = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    images = np.concatenate(images_array_list, axis=0) # [16*1600, 1200, 3]
    images_t = torch.from_numpy(images.transpose([2, 0, 1]).copy())
    images_t = cj_module.forward(images_t / 255.0) * 255.0
    images = images_t.numpy().astype(np.uint8).transpose(1, 2, 0)

    return images.reshape(-1, h, w, c)

def flip_image(img):
    results = []
    for i in img:
        results.append(np.fliplr(i).copy())
    return np.stack(results, axis = 0)

def flip_point(point, image_h, image_w, cam):
    f, cx, cy = cam[0, 0], cam[0, 2], cam[1, 2]
    pc_x, pc_y, depth = point[..., 0], point[..., 1], point[..., 2]

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y
    image_x = image_w - 1 - image_x

    pc_x = (image_x - cx) * depth / f
    pc_y = (image_y - cy) * depth / f
    pc = np.stack([pc_x, pc_y, depth], axis=-1)

    return pc

def flip_j2d(j2d, image_h, image_w, cam) :
    image_x, image_y = j2d[..., 0], j2d[...,1]
    image_x = image_w - 1 - image_x
    return np.stack([image_x, image_y], axis=-1)

def random_flip(img, image_h, image_w, point, pc_num, pose, trans, joint_3d, joint_2d, cam):
    new_img = flip_image(img)
    new_point = flip_point(point, image_h, image_w, cam)
    new_trans = flip_point(trans, image_h, image_w, cam)
    new_joint_3d = flip_point(joint_3d, image_h, image_w, cam)
    new_joint_2d = flip_j2d(joint_2d, image_h, image_w, cam)
    new_pose = pose
    return new_img, new_point, new_pose, new_trans, new_joint_3d, new_joint_2d

def crop_image(img, crop_window, image_h, image_w, cam):
    x1, y1, x2, y2 = crop_window
    new_img = img[:, y1:y2, x1:x2].copy()
    cam[0, 2] = cam[0, 2] - x1
    cam[1, 2] = cam[1, 2] - y1
    return new_img, cam

def random_scale(img, point, joint_2d, cam, image_h, image_w):
    if np.random.rand() < 0.5:
        return img, point, joint_2d, cam
    scale = np.random.uniform(0.99, 1.01)
    crop_h, crop_w = int(image_h / scale), int(image_w / scale)
    x1 = np.random.randint(low=0, high=image_w - crop_w + 1)
    y1 = np.random.randint(low=0, high=image_h - crop_h + 1)
    crop_window = [x1, y1, x1 + crop_w, y1 + crop_h]
    new_img, cam = crop_image(img, crop_window, image_h, image_w, cam)
    results = []
    for img in new_img:
        results.append(cv2.resize(img, (image_w, image_h), interpolation=cv2.INTER_LINEAR))
    results = np.stack(results, axis=0)
    # resize point and joint_2d
    scale_w = (image_w - 1) / (crop_w - 1)
    scale_h = (image_h - 1) / (crop_h - 1)
    point[:, :, 0] *= scale_w
    point[:, :, 1] *= scale_h
    joint_2d[:, :, 0] *= scale_w
    joint_2d[:, :, 1] *= scale_h
    # adjust cam params
    cam[0, 2] *= scale_w
    cam[1, 2] *= scale_h
    return results, point, joint_2d, cam



