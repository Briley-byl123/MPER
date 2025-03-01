from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory, BCP_net_3d,initialize_3D_prototypes
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../Datasets/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=1300, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=8000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
#prototype
parser.add_argument('--proto_head', type=int,  default=1, help='whether use prototype head')
parser.add_argument('--proto_unpdate_momentum', type=float,  default=0.999, help='prototype update momentum')
parser.add_argument('--num_micro_proto', type=int,  default=3, help='number of micro prototypes')
parser.add_argument('--confidence_threshold', type=float,  default=0.9, help='confidence threshold')
parser.add_argument('--loss_ppc_weight', type=float,  default=0.02, help='weight of pixel prototype classification loss')
parser.add_argument('--loss_ppd_weight', type=float,  default=0.02, help='weight of pixel prototype distance loss')
parser.add_argument('--temperature', type=float,  default=0.1, help='temperature of contrastive loss')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def PPD_loss(contrast_logits,contrast_target):

    contrast_logits = contrast_logits[contrast_target != 0, :]
    contrast_target = contrast_target[contrast_target != 0]

    logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())#logits:(N,1)
    loss_ppd = (1 - logits).pow(2).mean()

    return loss_ppd

def PPC_loss(output, target,temperature):
    loss_ppc = F.cross_entropy(output/temperature, target.long(),ignore_index=0)
    return loss_ppc
def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    current_state_dict = net.state_dict()

    net.load_state_dict(state['net'], strict=False)
    en_decoder = net.en_decoder
    conv_layer = getattr(en_decoder, 'representation')
    if conv_layer is not None and isinstance(conv_layer, nn.Conv3d): 

        out_channels, _, height, width,depth = conv_layer.weight.data.size()

        identity_matrix = torch.eye(out_channels).unsqueeze(2).unsqueeze(3).unsqueeze(4) 
        identity_matrix = identity_matrix.expand(-1, -1, height, width,depth) 
        conv_layer.weight.data.copy_(identity_matrix)
        

        if conv_layer.bias is not None:
            conv_layer.bias.data.zero_()
    
    current_optimizer_state_dict = optimizer.state_dict()
    pretrained_optimizer_state_dict = state['opt']
    

    for key, value in pretrained_optimizer_state_dict['state'].items():
        if key in current_optimizer_state_dict['state'] and 'params' in value:

            if len(value['params']) == len(current_optimizer_state_dict['state'][key]['params']):

                current_optimizer_state_dict['state'][key]['params'] = value['params']
            else:

                pass
    

    optimizer.load_state_dict(state['opt'])
    #optimizer.load_state_dict(current_optimizer_state_dict)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'], strict=False)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def pre_train(args, snapshot_path,init_proto):
    model = BCP_net_3d(args,in_chns=1, class_num=num_classes,ema=False, proto_head=False,output_proto=False)
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    #数据加载器
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs= model(volume_batch)
            loss_ce = F.cross_entropy(outputs['pred'], label_batch)
            loss_dice = DICE(outputs['pred'], label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:#如果当前迭代次数是 200 的倍数，则进行模型评估
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))#根据迭代次数和最佳 Dice 系数构建模型保存路径 save_mode_path 和 save_best_path。
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()
    
    if init_proto:
        logging.info('Start initializing prototypes')
        pre_trained_model = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
        proto_model = BCP_net_3d(args,in_chns=1, class_num=num_classes,ema=False,proto_head=True,output_proto=True)#student network
        load_net(proto_model, pre_trained_model)
        init_path='../model/model_LA_PPC_{}/LA_{}_{}_labeled/pre_train/{}_init_prototype.pkl'.format(args.loss_ppc_weight,args.exp, args.labelnum, args.model)
        initialize_3D_prototypes(args,proto_model, trainloader, init_path)
        #print("proto_list.shape",proto_list.shape)
        logging.info('Finish initializing prototypes')
        

def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = BCP_net_3d(args,in_chns=1, class_num=num_classes,ema=False, proto_head=True,output_proto=True,init_proto=True)#创建 3D BCP 网络模型 model。
    ema_model = BCP_net_3d(args,in_chns=1, class_num=num_classes,ema=True, proto_head=True,output_proto=True)

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    #load_net(model, pretrained_model)
    load_net_opt(model, optimizer, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            with torch.no_grad():
                unoutput_a = ema_model(unimg_a,eval=True)
                unoutput_b= ema_model(unimg_b,eval=True)
                plab_a = get_cut_mask(unoutput_a['pred'], nms=1)
                plab_b = get_cut_mask(unoutput_b['pred'], nms=1)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

            outputs_l= model(mixl_img,eval=False,unlab_flag=True,label=lab_a,plab=plab_a,pre=unoutput_a['pred'],select_mask=1-img_mask)
            outputs_u = model(mixu_img,eval=False,unlab_flag=True,label=lab_b,plab=plab_b,pre=unoutput_b['pred'],select_mask=img_mask)
            
            
            
            loss_l = mix_loss(outputs_l['pred'], lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u['pred'], plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            loss_l_proto= mix_loss(outputs_l['proto'], lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u_proto= mix_loss(outputs_u['proto'], plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            unl_ppc_loss = PPC_loss(outputs_u['contrast_logits'],outputs_u['contrast_target'],args.temperature)
            l_ppc_loss = PPC_loss(outputs_l['contrast_logits'],outputs_l['contrast_target'],args.temperature)

            loss = loss_l + loss_u + consistency_weight*(loss_l_proto+loss_u_proto+args.loss_ppc_weight * (unl_ppc_loss + l_ppc_loss))

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_l_proto', loss_l_proto, iter_num)
            writer.add_scalar('Self/loss_u_proto', loss_u_proto, iter_num)
            writer.add_scalar('Self/unl_ppc_loss', unl_ppc_loss, iter_num)
            writer.add_scalar('Self/l_ppc_loss', l_ppc_loss, iter_num)

            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f'%(iter_num, loss, loss_l, loss_u))

            update_ema_variables(model, ema_model, 0.99)

             # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save({'model_state_dict': model.state_dict(),'proto_list': model.proto_net.proto_list}, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            
            if iter_num % 200 == 1:
                ins_width = 2
                B,C,H,W,D = outputs_l['pred'].size()
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

                snapshot_img[:,:, H:H+ ins_width,:] = 1
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l['pred'], dim=1)
                seg_out = outputs_l_soft[0,1,...].permute(2,0,1) # y
                target =  mixl_lab[0,...].permute(2,0,1)
                train_img = mixl_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
                writer.add_images('Epoch_%d_Iter_%d_labeled'% (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u['pred'], dim=1)
                seg_out = outputs_u_soft[0,1,...].permute(2,0,1) # y
                target =  mixu_lab[0,...].permute(2,0,1)
                train_img = mixu_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "../model/model_LA_PPC_{}/LA_{}_{}_labeled/pre_train".format(args.loss_ppc_weight,args.exp, args.labelnum)
    self_snapshot_path = "../model/model_LA_PPC_{}/LA_{}_{}_labeled/self_train".format(args.loss_ppc_weight, args.exp,args.labelnum)

    print("Strating BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/LA_BCP_train.py', self_snapshot_path)
    
    
 # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path,init_proto=True) 
    
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
