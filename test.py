import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 
from utils.dataset_utils import DerainDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model_moe import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR()
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]




def test_Derain(net, dataset, task="derain"):
    output_path = testopt.output_path
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    # psnr = AverageMeter()
    # ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch, original_size) in tqdm(testloader):
            original_h, original_w = original_size

            h, w = degrad_patch.shape[-2], degrad_patch.shape[-1]

            multiple = 16

            new_h = max(multiple, (h // multiple) * multiple)
            new_w = max(multiple, (w // multiple) * multiple)

            if h % multiple != 0 or w % multiple != 0:
                degrad_patch_resized = F.interpolate(degrad_patch, size=(new_h, new_w), mode='bilinear', align_corners=False)
                clean_patch_resized = F.interpolate(clean_patch, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                degrad_patch_resized = degrad_patch
                clean_patch_resized = clean_patch

            degrad_patch_resized, clean_patch_resized = degrad_patch_resized.cuda(), clean_patch_resized.cuda()

            restored_resized = net(degrad_patch_resized)

            if new_h != h or new_w != w:
                restored = F.interpolate(restored_resized, size=(h, w), mode='bilinear', align_corners=False)
            else:
                restored = restored_resized

            save_image_tensor(restored, output_path + degraded_name[0] + '.jpg')
        print("Over")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=1, help='')
    parser.add_argument('--derain_path', type=str, default="/root/autodl-tmp/Dataset/test/RAIN/", help='test images')
    parser.add_argument('--output_path', type=str, default="weatherbench_output/RAIN/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default=".ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = "weatherbench_ckpt/epochepoch=080-avgPSNRval_avg_psnr=28.22" + testopt.ckpt_name

    derain_splits = [""]

    derain_tests = []

    print("CKPT name : {}".format(ckpt_path))

    net  = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    
    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDataset(testopt,addnoise=False,sigma=15)
            test_Derain(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif testopt.mode == 3:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} Derain...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")