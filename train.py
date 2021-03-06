import torch
from torch import nn
from torch import optim
from torch.utils import data
from dataset import Dataset

from model import RGBLNet

from tensorboardX import SummaryWriter
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='train epochs')
parser.add_argument('--data_path', default='./DATA/3-EC', type=str, help='path to dataset')
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--load', default=False, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default='./checkpoint/3-EC', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=2, type=int, help='gpu id')
parser.add_argument('--log_path', default='./checkpoint/3-EC', type=str, help='path to log')

args = parser.parse_args()

train_dataset = Dataset(args.data_path, True)
train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_dataset = Dataset(args.data_path, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = RGBLNet().to(device)

writer = SummaryWriter(args.log_path)

mseloss = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

if args.load:
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_mae = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))['mae']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_mae = 999999
    start_epoch = 0

for epoch in range(start_epoch, start_epoch + args.epoch):
    loss_avg, loss_att_avg = 0.0, 0.0

    for i, (images, density, _) in enumerate(tqdm(train_loader)):
        rgb = images[0].to(device)
        v = images[1].to(device)
        images = torch.cat((rgb, v), 1)
        density = density.to(device)
        outputs = model(images)

        loss = mseloss(outputs, density) / args.bs
        loss_sum = loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        loss_avg += loss.item()

    writer.add_scalar('loss/train_loss', loss_avg / len(train_loader), epoch)

    model.eval()
    with torch.no_grad():
        mae, mse = 0.0, 0.0
        for i, (images, gt) in enumerate(tqdm(test_loader)):
            rgb = images[0].to(device)
            v = images[1].to(device)
            images = torch.cat((rgb, v), 1)
            gt = gt.to(device)

            predict = model(images)

            mae += torch.abs(predict.sum() - gt).item()
            mse += ((predict.sum() - gt) ** 2).item()

        mae /= len(test_loader)
        mse /= len(test_loader)
        mse = mse ** 0.5
        print('Epoch:', epoch, 'MAE:', mae, 'MSE:', mse)
        writer.add_scalar('eval/MAE', mae, epoch)
        writer.add_scalar('eval/MSE', mse, epoch)

        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mae': mae,
                 'mse': mse}
        torch.save(state, os.path.join(args.save_path, 'checkpoint_latest.pth'))

        if mae < best_mae:
            best_mae = mae
            torch.save(state, os.path.join(args.save_path, 'checkpoint_best.pth'))
    model.train()

writer.close()
