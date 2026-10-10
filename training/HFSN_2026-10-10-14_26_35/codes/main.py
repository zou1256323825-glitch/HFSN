# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: train.py
@time: 2018/3/20

"""

import argparse
import copy
import os
import shutil
import socket
import time

import numpy as np
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from noise_layer import noiser
from torchvision import transforms
from data.ImageFolderDataset import MyImageFolder
from models.Encoder import Encoder
from models.Decoder import Decoder
from SSIM import SSIM
from lpips import lpips

import config as cfg


if cfg.SAVED_MODE:
    checkpoint_load = torch.load(cfg.SAVED_MODE_PATH)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=cfg.DATASET,
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=cfg.WORKERS,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=cfg.BATCH_SIZE,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=cfg.IMAGE_SIZE,
                    help='the number of frames')
parser.add_argument('--message_length', type=int, default=cfg.MESSAGE_LENGTH, help='the size of bit')
parser.add_argument('--niter', type=int, default=cfg.NITER,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=cfg.LR,
                    help='learning rate, default=0.001')
parser.add_argument('--loss', default=cfg.LOSS, help='l1 or l2')
parser.add_argument('--vgg_loss', default=cfg.VGG_LOSS, help='if use vgg_loss')
parser.add_argument('--decay_round', type=int, default=cfg.DECAY_ROUND,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=cfg.BETA1,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=cfg.CUDA,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=cfg.NGPU,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default=cfg.HNET_PATH,
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default=cfg.RNET_PATH,
                    help="path to Revealnet (to continue training)")
parser.add_argument('--bs_secret', type=int, default=cfg.BS_SECRET, help='batch size for ')
parser.add_argument('--trainpics', default=cfg.SAVE_ROOT,
                    help='folder to output training images')
parser.add_argument('--validationpics', default=cfg.SAVE_ROOT,
                    help='folder to output validation images')
parser.add_argument('--testPics', default=cfg.SAVE_ROOT,
                    help='folder to output test images')
parser.add_argument('--outckpts', default=cfg.SAVE_ROOT,
                    help='folder to output checkPoints')
parser.add_argument('--outlogs', default=cfg.SAVE_ROOT,
                    help='folder to output images')
parser.add_argument('--outcodes', default=cfg.SAVE_ROOT,
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=cfg.BETA,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default=cfg.REMARK, help='comment')
parser.add_argument('--test', default=cfg.TEST, help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--hostname', default=cfg.HOSTNAME, help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=cfg.DEBUG, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=cfg.LOG_FREQUENCY, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=cfg.RESULT_PIC_FREQUENCY, help='the frequency of save the resultPic')
parser.add_argument('--jpeg_quality', type=int, default=cfg.JPEG_QUALITY, help='JPEG quality')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/train.py
    cur_work_dir, mainfile = os.path.split(main_file_path)  # eg：/n/liyz/videosteganography/

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    model_dir1 = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir1, new_model_dir_path)


def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss, \
        optimizer, scheduler, DATA_DIR, mse_loss, vgg_loss, ssim_loss, lpips_loss, mse_loss, MRG_loss, \
        Noiser, mse_weight, ssim_weight, lpips_weight, decode_weight, mse_weight
    lpips_weight = cfg.LPIPS_WEIGHT
    ssim_weight = cfg.SSIM_WEIGHT
    mse_weight = cfg.MSE_WEIGHT
    decode_weight = cfg.DECODE_WEIGHT
    #################  output configuration   ###############
    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True
    DATA_DIR = cfg.DATA_ROOT
    assert DATA_DIR

    ############  create dirs to save the result #############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs1"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    if opt.debug:
        logPath = './debug/debug_logs/debug.txt'
    print_log(str(opt), logPath)

    save_current_codes(opt.outcodes)

    # tensorboardX writer
    writer = SummaryWriter(comment='_' + opt.remark)
    ##############   get dataset   ############################
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')
    train_dataset = MyImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize([opt.imageSize, opt.imageSize]),  # resize to a given size
            transforms.ToTensor(),
        ]))
    val_dataset = MyImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize([opt.imageSize, opt.imageSize]),
            transforms.ToTensor(),
        ]))
    assert train_dataset
    assert val_dataset

    # 定义网络结构
    Hnet = Encoder()
    Rnet = Decoder()

    Hnet.cuda()
    Hnet.apply(weights_init)
    Rnet.cuda()
    Rnet.apply(weights_init)

    if cfg.SAVED_MODE:
        Hnet.load_state_dict(checkpoint_load['netH'])
        Rnet.load_state_dict(checkpoint_load['netR'])
    # 冻结参数
    # for name, parameter in Hnet.named_parameters():
    #     parameter.requires_grad = False
    # for name, parameter in Rnet.named_parameters():
    #     parameter.requires_grad = False


    # 损失函数
    ssim_loss = SSIM().cuda()
    lpips_loss = lpips.LPIPS(net='alex').cuda()
    mse_loss = nn.MSELoss().cuda()

    # 噪声层定义
    Noiser = noiser.Noiser(cfg.NOISE_LAYERS)
    # Noiser = noiser.Noiser(['Identity'])
    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, weight_decay=1e-3, betas=(opt.beta1, 0.999))
        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, weight_decay=1e-3, betas=(opt.beta1, 0.999))

        if cfg.SAVED_MODE:
            optimizerH.load_state_dict(checkpoint_load['optimizerH'])
            optimizerR.load_state_dict(checkpoint_load['optimizerR'])

        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=int(opt.workers), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                                shuffle=False, num_workers=int(opt.workers), drop_last=True)
        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)
        for epoch in range(1, opt.niter+1):
            ######################## train ##########################################
            train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet)

            ####################### validation  #####################################
            validation(val_loader, epoch, Hnet=Hnet, Rnet=Rnet)

            # save the best NOSFTAdaptADDMoire parameters
            if epoch % 10 == 0:
                checkpoint = {
                    'netH': Hnet.state_dict(),
                    'netR': Rnet.state_dict(),
                    # 'netD': Dnet.state_dict(),
                    'optimizerH': optimizerH.state_dict(),
                    'optimizerR': optimizerR.state_dict(),
                    # 'optimizerD': optimizerD.state_dict(),
                    'schedulerH': schedulerH.state_dict(),
                    'schedulerR': schedulerR.state_dict(),
                    # 'schedulerD': schedulerD.state_dict(),
                }
                torch.save(checkpoint, f'{opt.outckpts}/checkpoint_{epoch}.pth')
                # torch.save(Hnet.state_dict(),
                #            '%s/epoch_%d.pth' % (
                #                opt.outckpts, epoch))
                # torch.save(Rnet.state_dict(),
                #            '%s/epoch_%d.pth' % (
                #                opt.outckpts, epoch))

        writer.close()


def forward_pass(cover_img, secret_img, Hnet, Rnet, mode='train'):
    batch_size_cover, channel_cover, _, _ = cover_img.size()
    container_img = Hnet(cover_img, secret_img)
    # encoded loss
    # loss_enc = mse_loss(container_img, cover_img)
    loss_enc_ssim = ssim_loss(container_img, cover_img)
    loss_enc_mse = mse_loss(cover_img, container_img)
    loss_enc_lpips = lpips_loss.forward(container_img, cover_img)

    NoisedAndCover, noise_idx = Noiser(container_img, cover_img, mode)
    container_img_noised = NoisedAndCover[0]

    rev_secret_img = Rnet(container_img_noised)

    # decoded loss
    loss_dec = mse_loss(rev_secret_img, secret_img)  # Reveal net

    return container_img, container_img_noised, rev_secret_img, loss_enc_ssim, \
           loss_enc_mse, loss_enc_lpips, loss_dec, noise_idx


def train(train_loader, epoch, Hnet, Rnet):
    mode = 'val'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    H_loss_enc = AverageMeter()
    H_loss_enc_ssim = AverageMeter()
    H_loss_enc_mse = AverageMeter()
    H_loss_enc_lpips = AverageMeter()
    R_loss_dec = AverageMeter()
    SumLosses = AverageMeter()  # record Hloss + β*Rloss
    # switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()

        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0])  # get true batch size of this step

        if opt.cuda:
            cover_img = all_pics.cuda()
            message = torch.Tensor(np.random.choice([0, 1], (cover_img.shape[0], opt.message_length))).cuda()

        container_img, container_img_noised, rev_secret_img, loss_enc_ssim, \
        loss_enc_mse, loss_enc_lpips, loss_dec, noise_idx = forward_pass(cover_img, message, Hnet, Rnet, mode)

        H_loss_enc_ssim.update(loss_enc_ssim.item(), this_batch_size)  # R loss
        H_loss_enc_lpips.update(torch.mean(loss_enc_lpips.view(this_batch_size)).item(), this_batch_size)
        H_loss_enc_mse.update(loss_enc_mse.item(), this_batch_size)
        R_loss_dec.update(loss_dec.item(), this_batch_size)

        err_sum = ssim_weight * (1 - loss_enc_ssim)+ \
                  + lpips_weight * torch.mean(loss_enc_lpips.view(this_batch_size)) \
                  + mse_weight * loss_enc_mse \
                  + decode_weight * (loss_dec)

        SumLosses.update(err_sum.item(), this_batch_size)
        err_sum.backward()
        optimizerH.step()
        optimizerR.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\t H_loss_enc_ssim: %.4f  H_loss_enc_lpips: %.4f  H_loss_enc_mse: %.4f ' \
              'R_loss_dec: %.4f\tSumLosses: %.4f \tdatatime: %.4f \tbatchtime: %.4f \t noise_idx:%.1f' % (
                  epoch, opt.niter, i, len(train_loader),
                  H_loss_enc_ssim.val, H_loss_enc_lpips.val, H_loss_enc_mse.val, R_loss_dec.val, SumLosses.val,
                  data_time.val,
                  batch_time.val, noise_idx)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)


    # epcoh log
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_H_loss_enc=%.6f\tepoch_H_loss_enc_ssim=%.6f\tepoch_R_loss_dec=%.6f\tepoch_sumLoss=%.6f" % (
        H_loss_enc.avg, H_loss_enc_ssim.avg, R_loss_dec.avg, SumLosses.avg)
    print_log(epoch_log, logPath)

    if not opt.debug:
        # record lr
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        # record loss
        writer.add_scalar('train/H_loss_enc', H_loss_enc.avg, epoch)
        writer.add_scalar('train/H_loss_enc_ssim', H_loss_enc_ssim.avg, epoch)
        writer.add_scalar('train/R_loss_dec', R_loss_dec.avg, epoch)
        writer.add_scalar('train/Sum_loss', SumLosses.avg, epoch)


def validation(val_loader, epoch, Hnet, Rnet):
    print(
        "#################################################### validation begin ########################################################")
    mode='val'
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    batch_time = AverageMeter()
    H_loss_enc_lpips = AverageMeter()
    H_loss_enc_mse = AverageMeter()
    H_loss_enc_ssim = AverageMeter()
    R_loss_dec = AverageMeter()
    SumLosses = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            all_pics = data
            this_batch_size = int(all_pics.size()[0])


            if opt.cuda:
                cover_img = all_pics.cuda()
                message = torch.Tensor(np.random.choice([0, 1], (cover_img.shape[0], opt.message_length))).cuda()
                # message = message.view(opt.batchSize, 1, np.sqrt(opt.message_length), np.sqrt(opt.message_length))

            container_img, container_img_noised, rev_secret_img, loss_enc_ssim, \
            loss_enc_mse, loss_enc_lpips, loss_dec, noise_idx = forward_pass(cover_img, message, Hnet, Rnet, mode)

            H_loss_enc_ssim.update(loss_enc_ssim.item(), this_batch_size)  # R loss
            H_loss_enc_mse.update(loss_enc_mse.item(), this_batch_size)
            H_loss_enc_lpips.update(torch.mean(loss_enc_lpips.view(this_batch_size)).item(), this_batch_size)
            R_loss_dec.update(loss_dec.item(), this_batch_size)


            if i % 500 == 0:
                save_result_pic(this_batch_size, cover_img, container_img.data, container_img_noised, R_loss_dec.val,
                                epoch, i, opt.validationpics)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            val_log = "validation[%d] val_H_loss_enc_ssim = %.6f\t H_loss_enc_lpips: %.4f\t" \
                      "H_loss_enc_mse: %.4f\t val_R_loss_dec = %.6f\t batch time=%.2f \tnoise_idx: noise_idx:%.1f" \
                      % (
                          epoch, H_loss_enc_ssim.val, H_loss_enc_lpips.val, H_loss_enc_mse.val, R_loss_dec.val,
                          batch_time.val, noise_idx)
            if i % opt.logFrequency == 0:
                print_log(val_log, logPath)

        if not opt.debug:
            writer.add_scalar('validation/H_loss_jndenc', H_loss_enc_mse.avg, epoch)
            writer.add_scalar('validation/H_loss_enc_ssim', H_loss_enc_ssim.avg, epoch)
            writer.add_scalar('validation/R_loss_dec', R_loss_dec.avg, epoch)
            writer.add_scalar('validation/Sum_loss', SumLosses.avg, epoch)

        print(
            "#################################################### validation end ########################################################")


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(this_batch_size, originalLabelv, ContainerImg, noisedimg, Decloss, epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize).cuda()
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize).cuda()
        # secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize).cuda()
        # revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize).cuda()
        noisedcontainerFrames = F.interpolate(noisedimg, (opt.imageSize, opt.imageSize), mode='bilinear').cuda()
        # noisedcontainerFrames = noisedimg.cuda()

        showContainer = torch.cat([originalFrames, containerFrames, noisedcontainerFrames], 0)
        # showReveal = torch.cat([secretFrames, revSecFrames], 0)
        # resultImg contains four rows: coverImg, containerImg, secretImg, RevSecImg, total this_batch_size columns
        # resultImg = torch.cat([showContainer, secretFrames], 0)
        resultImgName = '%s/epoch%03d_%06f.png' % (save_path, epoch, Decloss)
        vutils.save_image(showContainer, resultImgName, nrow=this_batch_size, padding=1, normalize=False)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
