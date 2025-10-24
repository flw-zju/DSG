import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import make_celebA_dataset, make_finetune_dataset
from net import SwappingNet, Discriminator
from utils import DiscriminatorLoss, VGGLoss, set_model_requires_grad
from models.SimSwap import get_id, get_tea
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--celebA_path', type=str, default="./celebA_dataset")
parser.add_argument('--tufts_vis_path', type=str, default="./tufts_dataset/vis")
parser.add_argument('--Arcface_path', type=str, default="./pretrained_model/arcface_checkpoint.tar")
parser.add_argument('--Simswap_path', type=str, default="./pretrained_model/simswap/people/")
parser.add_argument('--image_saving_path', type=str, default="./results/swapping/")
parser.add_argument('--model_saving_path', type=str, default="./trained_model/swapping_net/")
parser.add_argument('--celebA_epoch', type=int, default=5)
parser.add_argument('--finetune_epoch', type=int, default=1)
parser.add_argument('--batch_size_1', type=int, default=64,
                    help='batch size of CelebA dataset')
parser.add_argument('--batch_size_2', type=int, default=8,
                    help='batch size of Tufts dataset')
parser.add_argument('--lambda_l1', type=int, default=5,
                    help='l1 loss parameter')
parser.add_argument('--log_freq', type=int, default=500,
                    help='frequency for printing log information')
parser.add_argument('--model_freq', type=int, default=2,
                    help='frequency for saving the model')
parser.add_argument('--lr1', type=int, default=2e-4,
                    help='lr for swapping net')
parser.add_argument('--lr2', type=int, default=2e-4,
                    help='lr for discriminator')
parser.add_argument('--dim', type=int, default=512,
                    help='identity feature dimension extracted by Arcface')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SwappingNet(args.dim).to(device)
opt1 = torch.optim.Adam(model.parameters(), lr=args.lr1)

dis = Discriminator().to(device)
opt2 = torch.optim.Adam(dis.parameters(), lr=args.lr2)

id_enc = get_id(args.Arcface_path).to(device)
id_enc.requires_grad_(False)
id_enc = id_enc.eval()

teacher_model = get_tea(args.Simswap_path).to(device)
teacher_model.requires_grad_(False)
teacher_model.eval()

dis_loss = DiscriminatorLoss().to(device)
resize_112 = transforms.Resize((112, 112))
resize_128 = transforms.Resize((128, 128))
resize_224 = transforms.Resize((224, 224))
vgg_loss = VGGLoss(device).to(device)


def train_on_celebA():

    train_d = make_celebA_dataset(args.celebA_path)
    train_loader = DataLoader(train_d, batch_size=args.batch_size_1, shuffle=True, drop_last=False)
    print("Training on CelebA Dataset !")

    for epoch in range(args.celebA_epoch):
        print("")
        print("....................................... ")
        print("The epoch is", epoch)
        zjr = 0

        for vis in tqdm(train_loader):

            zjr += 1

            vis = vis.to(device)

            batch_size = vis.shape[0]
            arange = torch.arange(batch_size).to(device)
            idx = torch.randperm(batch_size).to(device)
            while 0.0 in (idx - arange):
                idx = torch.randperm(batch_size).to(device)

            with torch.no_grad():
                id = id_enc(resize_112(vis[idx,...]))
                id_ = id / torch.linalg.norm(id, axis=1, keepdims=True)
                tar = resize_128(teacher_model(resize_224(vis), id_))
            real = tar.detach().clone().requires_grad_(True)

            set_model_requires_grad(model.parameters(), requires_grad=True)
            set_model_requires_grad(dis.parameters(), requires_grad=False)

            rec_vis = model(vis, id_)
            fake = rec_vis.detach().clone()
            rec_score = dis(rec_vis)

            loss_distillation_rec = vgg_loss.content_loss(rec_vis, tar) + args.lambda_l1 * F.l1_loss(rec_vis, tar)
            loss_distillation_adv = dis_loss.gen_forward(rec_score)
            loss = loss_distillation_adv + loss_distillation_rec

            opt1.zero_grad()
            loss.backward()
            opt1.step()

            set_model_requires_grad(model.parameters(), requires_grad=False)
            set_model_requires_grad(dis.parameters(), requires_grad=True)

            r2 = dis(real)
            f2 = dis(fake)
            loss_d = dis_loss.dis_forward(r2, f2, real, zjr % 16 == 0)

            opt2.zero_grad()
            loss_d.backward()
            opt2.step()

            if zjr % args.log_freq == 0:

                save_image(torch.cat((rec_vis, tar, vis), dim=0),
                           args.image_saving_path + 'trained_on_celebA_' + str(epoch) + '.jpg',
                           normalize=False, nrow=8)
                print('')
                print('-----------------------------------------------------------')
                print('Discriminator loss:')
                print('loss:', loss_d.item())
                print('****************************************************')
                print('Generator losses:')
                print('loss_teacher_rec:', loss_distillation_rec.item())
                print('loss_teacher_adv:', loss_distillation_adv.item())
                print('****************************************************')

        if epoch % args.model_freq == 0:
            print("")
            print("Saving model trained on CelebA dataset at epoch ", epoch)
            torch.save(model.state_dict(), args.model_saving_path + 'swapping_net_distillation_' + str(epoch) + '.pt')
            torch.save(dis.state_dict(), args.model_saving_path + 'discriminator_swapping_distillation_' + str(epoch) + '.pt')


def finetune_on_tufts():
    train_d = make_finetune_dataset(args.tufts_vis_path, args.celebA_path)
    train_loader = DataLoader(train_d, batch_size=args.batch_size_2, shuffle=True, drop_last=False)
    print(print("Finetune on Tufts Dataset !"))

    for epoch in range(args.finetune_epoch):

        print("....................................... ")
        print("The epoch is", epoch)
        zjr = 0

        for id_img, vis in tqdm(train_loader):

            zjr += 1

            vis = vis.to(device)
            id_img = id_img.to(device)

            with torch.no_grad():
                id = id_enc(resize_112(id_img))
                id_ = id / torch.linalg.norm(id, axis=1, keepdims=True)
                tar = resize_128(teacher_model(resize_224(vis), id_))
            real = tar.detach().clone().requires_grad_(True)

            set_model_requires_grad(model.parameters(), requires_grad=True)
            set_model_requires_grad(dis.parameters(), requires_grad=False)

            rec_vis = model(vis, id_)
            fake = rec_vis.detach().clone()
            rec_score = dis(rec_vis)

            loss_adv = dis_loss.gen_forward(rec_score)
            loss_rec = args.lambda_l1 * F.l1_loss(rec_vis, tar) + vgg_loss.content_loss(rec_vis, tar)
            loss = loss_adv + loss_rec

            opt1.zero_grad()
            loss.backward()
            opt1.step()

            set_model_requires_grad(model.parameters(), requires_grad=False)
            set_model_requires_grad(dis.parameters(), requires_grad=True)

            r2 = dis(real)
            f2 = dis(fake)
            loss_d = dis_loss.dis_forward(r2, f2, real, zjr % 16 == 0)

            opt2.zero_grad()
            loss_d.backward()
            opt2.step()

            if zjr % args.log_freq == 0:
                save_image(torch.cat((rec_vis, tar, vis), dim=0),
                           args.image_saving_path + 'finetune_' + str(epoch) + '.jpg',
                           normalize=False, nrow=8)
                print('')
                print('-----------------------------------------------------------')
                print('Discriminator loss:')
                print('loss:', loss_d.item())
                print('****************************************************')
                print('Generator losses:')
                print('loss_rec:', loss_rec.item())
                print('loss_adv:', loss_adv.item())
                print('****************************************************')

        if epoch % args.model_freq == 0:
            print("")
            print("Saving model fine-tuned on Tufts dataset at epoch ", epoch)
            torch.save(model.state_dict(), args.model_saving_path + 'swapping_net_finetune_' + str(epoch) + '.pt')
            torch.save(dis.state_dict(), args.model_saving_path + 'discriminator_swapping_finetune_' + str(epoch) + '.pt')


if __name__ == "__main__":
    train_on_celebA()
    finetune_on_tufts()