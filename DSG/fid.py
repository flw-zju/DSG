from pytorch_fid import fid_score
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fid_value1 = fid_score.calculate_fid_given_paths(
    ["/home/tufts_dataset/vis/data", '/home/generated_images/vis/train'],
    batch_size=50,
    device=device,
    dims=2048
)
print(f'FID: {fid_value1}')

fid_value2 = fid_score.calculate_fid_given_paths(
    ["/home/tufts_dataset/inf/data", '/home/generated_images/inf/train'],
    batch_size=50,
    device=device,
    dims=2048
)
print(f'FID: {fid_value2}')

print(f'Average FID: {(fid_value1 + fid_value2) /2.}')