import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms


def set_model_requires_grad(params, requires_grad):
    for param in params:
        param.requires_grad = requires_grad


def angle_loss(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    loss = torch.sum(x * y, dim=1)
    return 1 - loss.mean()


def load(model, path):
    weights = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in weights.items() if
                       "from_rgb" not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class DiscriminatorLoss(nn.Module):
    def __init__(self, gp_coef=5) -> None:
        super(DiscriminatorLoss, self).__init__()
        self.gp_coef = gp_coef

    def dis_forward(self, discr_real1_pred, discr_fake1_pred,
                    real, flag):
        real_loss = F.softplus(-discr_real1_pred).mean()
        fake_loss = F.softplus(discr_fake1_pred).mean()
        d_loss = real_loss + fake_loss
        if flag:
            d_loss = d_r1_loss(discr_real1_pred, real) * self.gp_coef + d_loss
        return d_loss

    def d_reg(self, real, pred, zjr=2):
        gp = d_r1_loss(pred, real)
        return gp * self.gp_coef * zjr + 0 * pred[0]

    def gen_forward(self, discr_fake_pred):
        gen_loss = F.softplus(-discr_fake_pred).mean()
        return gen_loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.vgg.eval()
        self.criterion = nn.MSELoss(reduction='mean')
        self.style_weights = [1, 1, 0.6, 0, 0]

    def content_loss(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(self.style_weights)):
            loss += self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def style_loss(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(self.style_weights)):
            loss += self.style_weights[i] * self.criterion(self.gram(x_vgg[i]),
                                                           self.gram(y_vgg[i].detach()))
        return loss

    def gram(self, y):
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (h * w)
        return gram




