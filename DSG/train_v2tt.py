import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import make_v2tt_dataset
import argparse
from net import Net, Discriminator
from utils import DiscriminatorLoss, VGGLoss, set_model_requires_grad, angle_loss
from models.SimSwap import get_id
from model.lightcnn_v4 import LightCNN_V4
from torchvision.utils import save_image


parser = argparse.ArgumentParser()

parser.add_argument('--identity_text', type=str, default="/home/project_text/identity_text")
parser.add_argument('--celebA_root', type=str, default="/home/celebA_dataset/data/")
parser.add_argument('--tufts_vis_path', type=str, default="/home/tufts_dataset/vis")
parser.add_argument('--tufts_inf_path', type=str, default="/home/tufts_dataset/inf")
parser.add_argument('--Arcface_path', type=str, default="/home/pretrained_model/arcface_checkpoint.tar")
parser.add_argument('--semi_hfr_path', type=str, default="/home/pretrained_model/Tufts_checkpoint.pth.tar")
parser.add_argument('--swapping_net_path', type=str, default="/home/trained_model/swapping_net/")
parser.add_argument('--discriminator_path', type=str, default="/home/trained_model/swapping_net/")

parser.add_argument('--image_saving_path', type=str, default="/home/result/v2tt/")
parser.add_argument('--model_saving_path', type=str, default="/home/trained_model/v2tt/")
parser.add_argument('--epoch', type=int, default=41)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--train_mode', type=int, default=1)
parser.add_argument('--lambda_style', type=int, default=1,
                    help='style loss parameter')
parser.add_argument('--lambda_angle', type=int, default=0.7,
                    help='angle loss parameter')
parser.add_argument('--log_freq', type=int, default=500,
                    help='frequency for printing log information')
parser.add_argument('--model_freq', type=int, default=10,
                    help='frequency for saving the model')
parser.add_argument('--lr1', type=int, default=2e-4,
                    help='lr for swapping net')
parser.add_argument('--lr2', type=int, default=2e-4,
                    help='lr for discriminator')
parser.add_argument('--dim', type=int, default=512,
                    help='identity feature dimension extracted by Arcface')
args = parser.parse_args()


def train():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_d = make_v2tt_dataset(args.tufts_vis_path, args.tufts_inf_path, args.celebA_root, args.identity_text)
    train_loader = DataLoader(train_d, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = Net(args.dim, args.train_mode).to(device)
    model.swapping_net.load_state_dict(torch.load(args.swapping_net_path, map_location=device))

    opt1 = torch.optim.Adam(model.v2tt_net.parameters(), lr=args.lr1)

    dis = Discriminator().to(device)
    dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    opt2 = torch.optim.Adam(dis.parameters(), lr=args.lr2)

    dis_loss = DiscriminatorLoss().to(device)
    vgg_loss = VGGLoss(device).to(device)

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = 0.
    ada_aug_step = 0.6 / 500000
    r_t_stat = 0

    semi_hfr = LightCNN_V4(args.semi_hfr_path, device).to(device)
    semi_hfr.eval()

    id_enc = get_id(args.Arcface_path).to(device)
    id_enc.eval()

    print("")
    print("======== Training =======")
    for epoch in range(args.epoch):
        print("")
        print("....................................... ")
        print("The epoch is", epoch)
        zjr = 0

        for id_img, vis, inf in tqdm(train_loader):

            zjr += 1

            vis = vis.to(device)
            inf = inf.to(device)
            id_img = id_img.to(device)

            with torch.no_grad():
                id = id_enc(id_img)
                id = id / torch.linalg.norm(id, axis=1, keepdims=True)

            real = inf.detach().clone().requires_grad_(True)

            set_model_requires_grad(model.v2tt_net.parameters(), requires_grad=True)
            set_model_requires_grad(dis.parameters(), False)

            mod_vis, mod_inf = model(vis, id)
            fake = mod_inf.detach().clone()
            fake_score = dis(mod_inf, ada_aug_p)

            id_rgb = semi_hfr(mod_vis)[1]
            id_inf = semi_hfr(mod_inf)[1]

            loss_dis = dis_loss.gen_forward(fake_score)
            loss_style = vgg_loss.style_loss(mod_inf, inf)
            loss_angle = angle_loss(id_inf, id_rgb)
            loss = loss_dis + args.lambda_angle * loss_angle + args.lambda_style * loss_style

            opt1.zero_grad()
            loss.backward()
            opt1.step()

            set_model_requires_grad(model.v2tt_net.parameters(), requires_grad=False)
            set_model_requires_grad(dis.parameters(), True)

            r2 = dis(real, ada_aug_p)
            f2 = dis(fake, ada_aug_p)
            loss_d = dis_loss.dis_forward(r2, f2, real, zjr % 16 == 0)

            ada_augment += torch.tensor(
                (torch.sign(r2).sum().item(), r2.shape[0]), device=device
            )
            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()
                r_t_stat = pred_signs / n_pred
                if r_t_stat > 0.6:
                    sign = 1
                else:
                    sign = -1
                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

            opt2.zero_grad()
            loss_d.backward()
            opt2.step()


            if zjr % args.log_freq == 0 and epoch % args.model_freq == 0:

                save_image(torch.cat((mod_vis, mod_inf, inf, vis), dim=0),
                            args.image_saving_path + 'epochs_' + str(epoch) + '.jpg',
                            normalize=False, nrow=args.batch_size)
                print('')
                print('-----------------------------------------------------------')
                print('Discriminator loss:')
                print('loss_d:', loss_d.item())
                print('****************************************************')
                print('Generator losses:')
                print('loss_adv:', loss_dis.item())
                print('loss_style:', loss_style.item())
                print('loss_angle:', loss_angle.item())
                print('****************************************************')
        if epoch % args.model_freq == 0:
            torch.save(model.state_dict(), args.model_saving_path + 'model_tufts_' + str(epoch) + '.pt')
            torch.save(dis.state_dict(), args.model_saving_path + 'discriminator_tufts_'+str(epoch)+'.pt')


if __name__ == '__main__':
    train()