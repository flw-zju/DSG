import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import make_generation_dataset
import argparse
from net import Net
from models.SimSwap import get_id
import numpy as np
from PIL import Image
import os


parser = argparse.ArgumentParser()

parser.add_argument('--identity_text', type=str, default="/home/project_text/identity_text")
parser.add_argument('--celebA_root', type=str, default="/home/celebA_dataset/data/")
parser.add_argument('--tufts_vis_text', type=str, default="/home/tufts_list/vis_train.txt")
parser.add_argument('--Arcface_path', type=str, default="/home/pretrained_model/arcface_checkpoint.tar")
parser.add_argument('--net_path', type=str, default="/home/trained_model/provided/net1_40.pt")
parser.add_argument('--vis_saving_path', type=str, default="/home/generated_images/vis/train/")
parser.add_argument('--inf_saving_path', type=str, default="/home/generated_images/inf/train/")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--train_mode', type=int, default=2)
parser.add_argument('--log_freq', type=int, default=60,
                    help='frequency for printing generated image numbers')
parser.add_argument('--dim', type=int, default=512,
                    help='identity feature dimension extracted by Arcface')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tufts_dataset, celebA_dataset = make_generation_dataset(args.tufts_vis_text, args.celebA_root, args.identity_text)
celebA_loader = DataLoader(celebA_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
tufts_loader = DataLoader(tufts_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)


model = Net(args.dim, args.train_mode).to(device)
model.load_state_dict(torch.load(args.net_path, map_location=device))  # net1_40
model.eval()

id_enc = get_id(args.Arcface_path)
id_enc = id_enc.to(device)
id_enc = id_enc.eval()


with torch.no_grad():
    zjr = 0
    if not os.path.exists(args.vis_saving_path):
        os.makedirs(args.vis_saving_path)
        print(f"folder is created: {args.vis_saving_path}")
    if not os.path.exists(args.inf_saving_path):
        os.makedirs(args.inf_saving_path)
        print(f"folder is created: {args.inf_saving_path}")
    for id_img, id_label in tqdm(celebA_loader):
        for vis, vis_label in tufts_loader:
            id_img = id_img.to(device)
            vis = vis.to(device)
            vis = vis.squeeze(0)
            id = id_enc(id_img)
            id = id / torch.linalg.norm(id, axis=1, keepdims=True)
            id = id.repeat(vis.shape[0], 1)
            fake_vis, fake_inf = model(vis, id)

            zjr += vis.shape[0]

            for i in range(vis.shape[0]):
                save_name = str(int(id_label)) + "_" + str(int(vis_label)) + "_" + str(i) + ".jpg"

                save_img = fake_inf[i, :, :, :].detach().cpu().numpy()
                save_img = np.asarray(save_img, dtype=np.float32)
                save_img = np.rint(255 * save_img).clip(0, 255).astype(np.uint8)
                output = save_img.transpose(1, 2, 0)
                Image.fromarray(output, 'RGB').save(os.path.join(args.inf_saving_path, save_name))

                save_img = fake_vis[i, :, :, :].detach().cpu().numpy()
                save_img = np.asarray(save_img, dtype=np.float32)
                save_img = np.rint(255 * save_img).clip(0, 255).astype(np.uint8)
                output = save_img.transpose(1, 2, 0)
                Image.fromarray(output, 'RGB').save(os.path.join(args.vis_saving_path, save_name))
            break

    print(str(zjr) + " images with " + str(len(celebA_loader)) + " identities generated.")













