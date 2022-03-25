import torch
from torch.utils import data
from dataset import Dataset
from matplotlib import pyplot as plt
from RGBLNet import Model
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./DATA/2-GWD-EC', type=str, help='path to dataset')
parser.add_argument('--save_path', default='./checkpoint/2-GWD-EC', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

test_dataset = Dataset(args.data_path, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])

model.eval()
with torch.no_grad():
    mae, mse = 0.0, 0.0
    for i, (images, gt) in enumerate(test_loader):
        rgb = images[0].to(device)
        v = images[1].to(device)
        images = torch.cat((rgb, v), 1)
        # images = [rgb, v]
        gt = gt.to(device)

        predict = model(images)
        print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
        mae += torch.abs(predict.sum() - gt).item()
        mse += ((predict.sum() - gt) ** 2).item()
        predict = predict.squeeze(0).squeeze(0).cpu().numpy()

    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
    print('MAE:', mae, 'MSE:', mse)
