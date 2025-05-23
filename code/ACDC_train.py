import argparse
#from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
#import cv2
#import matplotlib.pyplot as plt
#import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory, initialize_prototypes
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
print("torch version:", torch.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../Datasets/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
#prototype
parser.add_argument('--proto_head', type=int,  default=1, help='whether use prototype head')
parser.add_argument('--proto_unpdate_momentum', type=float,  default=0.999, help='prototype update momentum')
parser.add_argument('--num_micro_proto', type=int,  default=3, help='number of micro prototypes')
parser.add_argument('--confidence_threshold', type=float,  default=0.9, help='confidence threshold')
parser.add_argument('--loss_ppc_weight', type=float,  default=0.001, help='weight of pixel prototype classification loss')
parser.add_argument('--loss_ppd_weight', type=float,  default=0.02, help='weight of pixel prototype distance loss')
parser.add_argument('--temperature', type=float,  default=0.1, help='temperature of contrastive loss')

# label and unlabel                  
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')

args = parser.parse_args()

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path):
    
    state = torch.load(str(path))
    print(state.keys())

    net.load_state_dict(state['net'], strict=False)

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    current_state_dict = net.state_dict()

    net.load_state_dict(state['net'], strict=False)
    en_decoder = net.en_decoder
    conv_layer = getattr(en_decoder, 'representation')
    if conv_layer is not None and isinstance(conv_layer, nn.Conv2d): 

        out_channels, _, height, width = conv_layer.weight.data.size()
        identity_matrix = torch.eye(out_channels).unsqueeze(2).unsqueeze(3)
        identity_matrix = identity_matrix.expand(-1, -1, height, width) 
        conv_layer.weight.data.copy_(identity_matrix)
        
        if conv_layer.bias is not None:
            conv_layer.bias.data.zero_()


    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []  
    N = segmentation.shape[0]  
    for i in range(0, N):
        class_list = []  
        for c in range(1, 4):  
            temp_seg = segmentation[i]  
            temp_prob = torch.zeros_like(temp_seg)  
            temp_prob[temp_seg == c] = 1 
            temp_prob = temp_prob.detach().cpu().numpy()  
            labels = label(temp_prob) 

            if labels.max() != 0:  
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1  
                class_list.append(largestCC * c)  
            else:
                class_list.append(temp_prob)  

        n_batch = class_list[0] + class_list[1] + class_list[2] 
        batch_list.append(n_batch)  

    return torch.Tensor(batch_list).cuda()  
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    ema_model_state = ema_model.state_dict()
    new_dict = {}

    for key in model_state:
        if key in ema_model_state and model_state[key].shape == ema_model_state[key].shape:
            new_dict[key] = alpha * ema_model_state[key] + (1 - alpha) * model_state[key]

    ema_model.load_state_dict(new_dict, strict=False)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]#6,1,256,256
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()
 
def PPD_loss(contrast_logits,contrast_target):
    contrast_logits = contrast_logits[contrast_target != 0, :]
    contrast_target = contrast_target[contrast_target != 0]

    logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
    loss_ppd = (1 - logits).pow(2).mean()

    return loss_ppd

def PPC_loss(output, target,temperature):
    loss_ppc = F.cross_entropy(output/temperature, target.long(),ignore_index=0)
    return loss_ppc



def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path,init_proto):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
      

    model = BCP_net(args,in_chns=1, class_num=num_classes, ema=False, proto_head=False,output_proto=False)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))#Total slices is: 1312, labeled slices is:136
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100


    iterator = tqdm(range(max_epoch), ncols=70)
    
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]#bs, 1, 256, 256->6,6
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)#img_mask:(256,256);loss_mask:(bs, 256, 256)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)#bs, 256, 256




            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input,eval=False)



     
            loss_dice, loss_ce = mix_loss(out_mixl['pred'], lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2      
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl['pred'], dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:

            
            iterator.close()
            break

        
    writer.close()
   
    logging.info('Finish pre_training')
    
    if init_proto:
        logging.info('Start initializing prototypes')
        pre_trained_model = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
        proto_model = BCP_net(args,in_chns=1, class_num=num_classes,ema=False,proto_head=True,output_proto=True)#student network
        load_net(proto_model, pre_trained_model)
        init_path='../model/model_ACDC_PPC_{}/ACDC_{}_{}_labeled/pre_train/{}_init_prototype.pkl'.format(args.loss_ppc_weight,args.exp, args.labelnum, args.model)

        initialize_prototypes(args,proto_model, trainloader, init_path)
        logging.info('Finish initializing prototypes')


        
    

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)#6,6
     
    model = BCP_net(args,in_chns=1, class_num=num_classes,ema=False,proto_head=True,output_proto=True,init_proto=True)#student network
    ema_model = BCP_net(args,in_chns=1, class_num=num_classes, ema=True,proto_head=True,output_proto=True)#teacher network

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()#student
    ema_model.train()#teacher

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']#数据类型分别为：torch.Size([24, 1, 256, 256])，torch.Size([24, 256, 256])
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]#6,1,256,256
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                pre_a = ema_model(uimg_a,eval=True)
                pre_b = ema_model(uimg_b,eval=True)
                plab_a = get_ACDC_masks(pre_a['pred'], nms=1)
                plab_b = get_ACDC_masks(pre_b['pred'], nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl,eval=False,unlab_flag=True,label=lab_a,plab=plab_a,pre=pre_a['pred'],select_mask=img_mask)
            out_l = model(net_input_l,eval=False,unlab_flag=True,label=lab_b,plab=plab_b,pre=pre_b['pred'],select_mask=1-img_mask)
            
            linear_out_unl=out_unl["pred"]
            linear_out_l=out_l["pred"]
            proto_out_unl = out_unl["proto"]
            proto_out_l = out_l["proto"]


            proto_out = proto_out_unl.clone()
            proto_out = get_ACDC_masks(proto_out, nms=1)


            linear_out = linear_out_unl.clone()
            linear_out = get_ACDC_masks(linear_out, nms=1)

            proto_out = proto_out[1].detach().cpu().numpy().astype(np.float32)
            linear_out = linear_out[1].detach().cpu().numpy().astype(np.float32)
            data_range = np.max([proto_out.max(), linear_out.max()]) - np.min([proto_out.min(), linear_out.min()])
            ssim_score = ssim(proto_out ,linear_out, data_range=data_range)
            mse_score = mse(proto_out, linear_out)


            unl_proto_dice,unl_proto_ce=mix_loss(proto_out_unl,plab_a,lab_a,loss_mask,u_weight=args.u_weight,unlab=True)       
            l_proto_dice,l_proto_ce=mix_loss(proto_out_l,lab_b,plab_b,loss_mask,u_weight=args.u_weight)    


            unl_ppc_loss = PPC_loss(out_unl['contrast_logits'],out_unl['contrast_target'],args.temperature)
            l_ppc_loss = PPC_loss(out_l['contrast_logits'],out_l['contrast_target'],args.temperature)

            #ppd_loss
            #unl_ppd_loss = PPD_loss(out_unl['contrast_logits'],out_unl['contrast_target'])
            #l_ppd_loss = PPD_loss(out_l['contrast_logits'],out_l['contrast_target'])


            unl_dice, unl_ce = mix_loss(linear_out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(linear_out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)

            #sum
            loss_ce = unl_ce + l_ce+(unl_proto_ce+l_proto_ce)*consistency_weight
            loss_dice = unl_dice + l_dice + (unl_proto_dice + l_proto_dice)*consistency_weight

            loss_ppc = unl_ppc_loss + l_ppc_loss

            loss = (loss_dice + loss_ce) / 2 + args.loss_ppc_weight*loss_ppc *consistency_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl['pred'], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l['pred'], dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)


                outputs_proto = torch.argmax(torch.softmax(out_unl['proto'], dim=1), dim=1, keepdim=True)#维度为：bs, 1, 256, 256
                writer.add_image('train/Un_Prototype', outputs_proto[1, ...] * 50, iter_num)
                outputs_proto_l = torch.argmax(torch.softmax(out_l['proto'], dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prototype', outputs_proto_l[1, ...] * 50, iter_num)
                

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/SSIM', ssim_score, iter_num)
                writer.add_scalar('info/MSE', mse_score, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save({'model_state_dict': model.state_dict(),'proto_list': model.proto_net.proto_list}, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":


    pre_snapshot_path = "../model/model_ACDC_PPC_{}/ACDC_{}_{}_labeled/pre_train".format(args.loss_ppc_weight,args.exp, args.labelnum)
    self_snapshot_path = "../model/model_ACDC_PPC_{}/ACDC_{}_{}_labeled/self_train".format(args.loss_ppc_weight, args.exp,args.labelnum)

    
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/ACDC_BCP_train.py', self_snapshot_path)


    

    #Pre_train


    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path,init_proto=True) 

    



    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)




