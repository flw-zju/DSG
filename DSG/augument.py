import random
import torch
import math
import torch.nn.functional as F


def DiffAugment(x, p):
    for f in AUGMENT_FNS.values():
        for fs in f:
            x = fs(x, p)
    return x.contiguous()


def translate_mat(t_x, t_y):
    batch = t_x.shape[0]
    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y), 1)
    mat[:, :2, 2] = translate
    return mat


def rotate_mat(theta):
    batch = theta.shape[0]
    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rot = torch.stack((cos_t, -sin_t, sin_t, cos_t), 1).view(batch, 2, 2)
    mat[:, :2, :2] = rot
    return mat


def scale_mat(s_x, s_y):
    batch = s_x.shape[0]
    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y
    return mat


def translate3d_mat(t_x, t_y, t_z):
    batch = t_x.shape[0]
    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y, t_z), 1)
    mat[:, :3, 3] = translate
    return mat


def rotate3d_mat(axis, theta):
    batch = theta.shape[0]
    u_x, u_y, u_z = axis
    eye = torch.eye(3).unsqueeze(0)
    cross = torch.tensor([(0, -u_z, u_y), (u_z, 0, -u_x), (-u_y, u_x, 0)]).unsqueeze(0)
    outer = torch.tensor(axis)
    outer = (outer.unsqueeze(1) * outer).unsqueeze(0)
    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)
    rot = cos_t * eye + sin_t * cross + (1 - cos_t) * outer
    eye_4 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    eye_4[:, :3, :3] = rot
    return eye_4


def scale3d_mat(s_x, s_y, s_z):
    batch = s_x.shape[0]
    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y
    mat[:, 2, 2] = s_z
    return mat


def luma_flip_mat(axis, i):
    batch = i.shape[0]
    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    flip = 2 * torch.ger(axis, axis) * i.view(-1, 1, 1)
    return eye - flip


def saturation_mat(axis, i):
    batch = i.shape[0]
    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    axis = torch.ger(axis, axis)
    saturate = axis + (eye - axis) * i.view(-1, 1, 1)
    return saturate


def lognormal_sample(size, mean=0, std=1):
    return torch.empty(size).log_normal_(mean=mean, std=std)


def category_sample(size, categories):
    category = torch.tensor(categories)
    sample = torch.randint(high=len(categories), size=(size,))
    return category[sample]


def uniform_sample(size, low, high):
    return torch.empty(size).uniform_(low, high)


def normal_sample(size, mean=0, std=1):
    return torch.empty(size).normal_(mean, std)


def bernoulli_sample(size, p):
    return torch.empty(size).bernoulli_(p)


def random_mat_apply(p, transform, prev, eye):
    size = transform.shape[0]
    select = bernoulli_sample(size, p).view(size, 1, 1)
    select_transform = select * transform + (1 - select) * eye
    return select_transform @ prev


def apply_color(img, mat):
    batch = img.shape[0]
    img = img.permute(0, 2, 3, 1)
    mat_mul = mat[:, :3, :3].transpose(1, 2).view(batch, 1, 3, 3)
    mat_add = mat[:, :3, 3].view(batch, 1, 1, 3)
    img = img @ mat_mul + mat_add
    img = img.permute(0, 3, 1, 2)
    return img


def sample_color(p, size):
    C = torch.eye(4).unsqueeze(0).repeat(size, 1, 1)
    eye = C
    axis_val = 1 / math.sqrt(3)
    axis = (axis_val, axis_val, axis_val)

    # brightness
    param = normal_sample(size, std=0.2)
    Cc = translate3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # contrast
    param = lognormal_sample(size, std=0.5 * math.log(2))
    Cc = scale3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # luma flip
    param = category_sample(size, (0, 1))
    Cc = luma_flip_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # hue rotation
    param = uniform_sample(size, -math.pi, math.pi)
    Cc = rotate3d_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # saturation
    param = lognormal_sample(size, std=1 * math.log(2))
    Cc = saturation_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    return C


def random_apply_color(img, p, C=None):
    if C is None:
        C = sample_color(p, img.shape[0])
    img = apply_color(img, C.to(img))
    return img


def rand_translation(x, ratio):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_offset(x, ratio):
    w, h = x.size(2), x.size(3)
    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio)
        max_v = int(h * ratio * ratio)
        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v
        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)
        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)
        imgs.append(img)
    return torch.stack(imgs)


def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio)


def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio)


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [random_apply_color],
    'offset': [rand_offset],
    'offset_h': [rand_offset_h],
    'offset_v': [rand_offset_v],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
