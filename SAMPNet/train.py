import sys,os
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch
import time
import shutil
from torch.utils.data import DataLoader
import csv

from samp_net import EMDLoss, AttributeLoss, SAMPNet
from config import Config
from cadb_dataset import CADBDataset
from test import evaluation_on_cadb

def calculate_accuracy(predict, target, threhold=2.6):
    assert target.shape == predict.shape, '{} vs. {}'.format(target.shape, predict.shape)
    bin_tar = target > threhold
    bin_pre = predict > threhold
    correct = (bin_tar == bin_pre).sum()
    acc     = correct.float() / target.size(0)
    return correct,acc

def build_dataloader(cfg):
    trainset = CADBDataset('train', cfg)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=cfg.num_workers,
                             drop_last=False)
    return trainloader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model
        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        self.trainloader = build_dataloader(cfg)
        self.optimizer = self.create_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer, mode='min', patience=5)
        self.epoch = 0
        self.iters = 0

        self.avg_mse = 0.
        self.avg_emd = 0.
        self.avg_acc = 0.
        self.avg_att = 0.

        self.smooth_coe = 0.4
        self.smooth_mse = None
        self.smooth_emd = None
        self.smooth_acc = None
        self.smooth_att = None

        self.mse_loss = torch.nn.MSELoss()
        self.emd_loss = EMDLoss()

        self.test_acc = []
        self.test_emd1 = []
        self.test_emd2 = []
        self.test_mse = []
        self.test_srcc = []
        self.test_lcc = []

        if cfg.use_attribute:
            self.att_loss = AttributeLoss(cfg.attribute_weight)

        self.least_metric = 1.
        self.writer = self.create_writer()

    def create_optimizer(self):
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False
        bb_params = list(map(id, self.model.backbone.parameters()))
        lr_params = filter(lambda p:id(p) not in bb_params, self.model.parameters())
        params = [
            {'params': lr_params, 'lr': self.cfg.lr},
            {'params': self.model.backbone.parameters(), 'lr': self.cfg.lr * 0.01}
            ]
        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(params,
                                   weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == 'sgd':
            optimizer = optim.SGD(params,
                                  momentum=self.cfg.momentum,
                                  weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"not such optimizer {self.cfg.optimizer}")
        return optimizer

    def create_writer(self):
        print('Create tensorboardX writer...', self.cfg.log_dir)
        writer = SummaryWriter(log_dir=self.cfg.log_dir)
        return writer

    def run(self):
        for epoch in range(self.cfg.max_epoch):
            self.run_epoch()
            self.epoch += 1
            self.scheduler.step(metrics=self.least_metric)
            self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.epoch)
            if self.epoch % self.cfg.save_epoch == 0:
                checkpoint_path = os.path.join(self.cfg.checkpoint_dir, 'model-{epoch}.pth')
                torch.save(self.model.state_dict(), checkpoint_path.format(epoch=self.epoch))
                print('Save checkpoint...')
            if self.epoch % self.cfg.test_epoch == 0:
                test_emd = self.eval_training()
                if test_emd < self.least_metric:
                    self.least_metric = test_emd
                    checkpoint_path = os.path.join(self.cfg.checkpoint_dir, 'model-best.pth')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print('Update best checkpoint...')
                self.writer.add_scalar('Test/Least EMD', self.least_metric, self.epoch)


    def eval_training(self):
        avg_acc, avg_r1_emd, avg_r2_emd, avg_mse, SRCC, LCC = \
            evaluation_on_cadb(self.model, self.cfg)
        self.writer.add_scalar('Test/Average EMD(r=2)', avg_r2_emd, self.epoch)
        self.writer.add_scalar('Test/Average EMD(r=1)', avg_r1_emd, self.epoch)
        self.writer.add_scalar('Test/Average MSE', avg_mse, self.epoch)
        self.writer.add_scalar('Test/Accuracy', avg_acc, self.epoch)
        self.writer.add_scalar('Test/SRCC', SRCC, self.epoch)
        self.writer.add_scalar('Test/LCC', LCC, self.epoch)
        error = avg_r1_emd

        self.test_acc.append(avg_acc)
        self.test_emd1.append(avg_r1_emd)
        self.test_emd2.append(avg_r2_emd)
        self.test_mse.append(avg_mse)
        self.test_srcc.append(SRCC)
        self.test_lcc.append(LCC)
        self.write2csv()
        return error

    def write2csv(self):
        csv_path = os.path.join(self.cfg.exp_path, '..', '{}.csv'.format(self.cfg.exp_name))
        header = ['epoch', 'Accuracy', 'EMD r=1', 'EMD r=2', 'MSE', 'SRCC', 'LCC']
        epoches = list(range(len(self.test_acc)))
        metrics = [epoches, self.test_acc, self.test_emd1, self.test_emd2,
                   self.test_mse, self.test_srcc, self.test_lcc]
        rows = [header]
        for i in range(len(epoches)):
            row = [m[i] for m in metrics]
            rows.append(row)
        for name, m in zip(header, metrics):
            if name == 'epoch':
                continue
            index = m.index(min(m))
            if name in ['Accuracy', 'SRCC', 'LCC']:
                index = m.index(max(m))
            title = 'best {} (epoch-{})'.format(name, index)
            row = [l[index] for l in metrics]
            row[0] = title
            rows.append(row)
        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ', csv_path)

    def dist2ave(self, pred_dist):
        pred_score = torch.sum(pred_dist* torch.Tensor(range(1,6)).to(pred_dist.device), dim=-1, keepdim=True)
        return pred_score

    def run_epoch(self):
        self.model.train()
        for batch, data in enumerate(self.trainloader):
            self.iters += 1
            image = data[0].to(self.device)
            score = data[1].to(self.device)
            score_dist = data[2].to(self.device)
            saliency = data[3].to(self.device)
            attributes = data[4].to(self.device)
            weight = data[5].to(self.device)

            pred_weight, pred_atts, pred_dist = self.model(image, saliency)

            if self.cfg.use_weighted_loss:
                dist_loss = self.emd_loss(score_dist, pred_dist, weight)
            else:
                dist_loss = self.emd_loss(score_dist, pred_dist)

            if self.cfg.use_attribute:
                att_loss = self.att_loss(attributes, pred_atts)
                loss = dist_loss + att_loss
            else:
                loss = dist_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.avg_emd += dist_loss.item()
            self.avg_att += att_loss.item()
            pred_score = self.dist2ave(pred_dist)
            correct, accuracy = calculate_accuracy(pred_score, score)
            self.avg_acc += accuracy.item()
            if (self.iters+1) % self.cfg.display_steps == 0:
                print('ground truth:   average={}'.format(score.view(-1)))
                print('prediction:     average={}'.format(pred_score.view(-1)))

                self.avg_emd = self.avg_emd / self.cfg.display_steps
                self.avg_acc = self.avg_acc / self.cfg.display_steps
                if self.cfg.use_attribute:
                    self.avg_att = self.avg_att / self.cfg.display_steps

                if self.smooth_emd != None:
                    self.avg_emd = (1-self.smooth_coe) * self.avg_emd + self.smooth_coe * self.smooth_emd
                    self.avg_acc = (1-self.smooth_coe) * self.avg_acc + self.smooth_coe * self.smooth_acc
                    if self.cfg.use_attribute:
                        self.avg_att = (1-self.smooth_coe) * self.avg_att + self.smooth_coe * self.smooth_att
                        self.writer.add_scalar('Train/AttributeLoss', self.avg_att, self.iters)

                self.writer.add_scalar('Train/EMD_Loss', self.avg_emd, self.iters)
                self.writer.add_scalar('Train/Accuracy', self.avg_acc, self.iters)

                if self.cfg.use_attribute:
                    print('Traning Epoch:{}/{} Current Batch: {}/{} EMD_Loss:{:.4f} Attribute_Loss:{:.4f}  ACC:{:.2%} lr:{:.6f} '.
                        format(
                        self.epoch, self.cfg.max_epoch,
                        batch, len(self.trainloader),
                        self.avg_emd, self.avg_att,
                        self.avg_acc,
                        self.optimizer.param_groups[0]['lr']))
                else:
                    print(
                        'Traning Epoch:{}/{} Current Batch: {}/{} EMD_Loss:{:.4f} ACC:{:.2%} lr:{:.6f} '.
                            format(
                            self.epoch, self.cfg.max_epoch,
                            batch, len(self.trainloader),
                            self.avg_emd, self.avg_acc,
                            self.optimizer.param_groups[0]['lr']))

                self.smooth_emd = self.avg_emd
                self.smooth_acc = self.avg_acc

                self.avg_mse = 0.
                self.avg_emd = 0.
                self.avg_acc = 0.
                if self.cfg.use_attribute:
                    self.smooth_att = self.avg_att
                    self.avg_att = 0.
                print()

if __name__ == '__main__':
    cfg = Config()
    cfg.create_path()
    device = torch.device('cuda:{}'.format(cfg.gpu_id))
    # evaluate(cfg)
    for file in os.listdir('./'):
        if file.endswith('.py'):
            shutil.copy(file, cfg.exp_path)
            print('Backup ', file)

    model = SAMPNet(cfg)
    model = model.train().to(device)
    trainer = Trainer(model, cfg)
    trainer.run()