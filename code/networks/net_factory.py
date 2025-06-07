from networks.unet import UNet, UNet_2d, CosProto_Module
from networks.VNet import VNet, CosProto_Module_3D
import torch.nn as nn
import torch
import torch.nn as nn
import importlib
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def net_factory(args=None,net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = NNet_2d(args=args,in_chns=in_chns, class_num=class_num,proto_head =True,output_proto=True,init_proto=False).cuda()
 
    if net_type == "VNet" and mode == "test" and tsne==0:
        net = NNet_3d(args=args,in_chns=in_chns, class_num=class_num, proto_head =True,output_proto=True,init_proto=False,normalization='batchnorm', has_dropout=False).cuda()
    return net



def BCP_net(args,in_chns=1, class_num=2, ema=False, proto_head=False,output_proto=False,init_proto=False):
    net = NNet_2d(args,in_chns=in_chns, class_num=class_num, proto_head = proto_head,output_proto=output_proto,init_proto=init_proto).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def BCP_net_3d(args,in_chns=1, class_num=2, ema=False, proto_head=False,output_proto=False,init_proto=False,normalization='batchnorm', has_dropout=True):
    net = NNet_3d(args,in_chns=in_chns, class_num=class_num, proto_head = proto_head,output_proto=output_proto,init_proto=init_proto,normalization=normalization,has_dropout=has_dropout).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

class NNet_2d(nn.Module):
    def __init__(self,args,in_chns, class_num ,proto_head, output_proto,init_proto):
        super(NNet_2d, self).__init__()
        self.in_chns = in_chns
        self.class_num = class_num
        self.args = args
        self.proto_head = proto_head
        self.output_proto = output_proto
        self.num_micro_proto = args.num_micro_proto
        self.init_proto = init_proto
        if self.proto_head:
            self.proto_net = CosProto_Module(args=args,in_planes = 16, num_classes =class_num , num_micro_proto = self.num_micro_proto, proto_update_momentum=0.99,init_proto=init_proto)

            self.en_decoder = UNet_2d(in_chns=in_chns, class_num=class_num,rep_clf=True,rep_head=True).cuda()
        else:
            self.en_decoder = UNet_2d(in_chns=in_chns, class_num=class_num,rep_clf=True,rep_head=True).cuda()

    def forward(self, x, eval=False, unlab_flag=None,label=None, plab=None,pre=False, select_mask=None):
        outs = self.en_decoder(x)
        if eval:
            # evaluation phase do not need to do sampling
            pred_proto,_,_ = self.proto_net(outs["rep_clf"])
            outs.update({"proto": pred_proto})    
            return outs#[outs["pred"],outs["rep_clf"]
        if self.proto_head and self.output_proto:
            pred_proto,proto_match_idx, contrast_logits= self.proto_net(outs["rep_clf"])
            outs.update({"proto_match_idx": proto_match_idx})
            
            
            with torch.no_grad():
                if unlab_flag:
                    gt_seg=plab*select_mask+label*(1-select_mask)
                    gt_seg = gt_seg.view(-1)
                    
                    pred_seg =torch.max(pred_proto,1)[1]
                    b,h,w=pred_seg.shape
                    pred_seg_expanded = pred_seg.unsqueeze(-1)
                    predicted_idx = torch.gather(proto_match_idx.view(b,h,w,-1), -1, pred_seg_expanded)
                    predicted_idx = predicted_idx.squeeze(-1).view(-1)
                    
                    pred_seg = pred_seg.view(-1)
                    mask = (gt_seg == pred_seg )
                    mask=mask.view(b,h,w)
                    
                    proto_target= predicted_idx.float()+self.num_micro_proto*pred_seg


                    probablity= F.softmax(pre, dim=1)#softmax
                    confidence_threshold = self.args.confidence_threshold
                    confidence, predicted_class = torch.max(probablity, 1)
                    
                    high_confidence_mask= confidence > confidence_threshold

                    #unlabel
                    merged_mask_un =mask&high_confidence_mask
                    self.update_proto(outs["rep"], proto_match_idx, plab, select_mask,merged_mask_un)
                    #label
                    bs,h,w=label.shape
                    label_cond = torch.ones(bs, h, w, dtype=torch.bool).cuda()
                    merged_mask_l =mask&label_cond
                    self.update_proto(outs["rep"], proto_match_idx, label, 1-select_mask,merged_mask_l)
                    outs.update({"contrast_logits":contrast_logits,"contrast_target":proto_target})
            outs.update({"proto": pred_proto})
        return outs
    
    def update_proto(self, rep, proto_match_idx, target, select_mask,cond):


        bs, in_planes, H, W = rep.shape
        proto_match_idx = proto_match_idx.reshape(bs, H, W, 4)#(bs, H, W, 4)

        rep_selected = rep.permute(0, 2, 3, 1)[:, select_mask == 1]
        bs, hw, c = rep_selected.shape
        rep_selected = rep_selected.reshape(bs*hw, c)
        target_selected = target[:, select_mask == 1].view(-1)
        cond = cond[:, select_mask == 1].view(-1)#(bs*H*W_new)

        
        valid_mask = target_selected != 0

              
        rep_selected_valid = rep_selected[valid_mask]
        target_selected_valid = target_selected[valid_mask]

        proto_match_idx_valid = proto_match_idx[:,select_mask==1]
        bs,hw,c = proto_match_idx_valid.shape
        proto_match_idx_valid = proto_match_idx_valid.reshape(bs*hw, c)

        proto_match_idx_valid = proto_match_idx_valid[valid_mask]
        cond_valid = cond[valid_mask]
 
        proto_match_idx_valid_target = proto_match_idx_valid[torch.arange(len(target_selected_valid)), target_selected_valid.long()]

        rep_selected_valid_cond = rep_selected_valid[cond_valid,:]
        proto_match_idx_valid_target_cond = proto_match_idx_valid_target[cond_valid]
        target_selected_valid_cond = target_selected_valid[cond_valid]

        if cond_valid.sum() > 0:
           shot_cls = target_selected_valid_cond.unique()
           for _cls in shot_cls:
              for proto_idx in range(self.num_micro_proto):
                   _candidate_mask = torch.logical_and(target_selected_valid_cond == _cls, proto_match_idx_valid_target_cond == proto_idx)
                   if _candidate_mask.sum() > 0:
                       self.proto_net.update_proto(rep_selected_valid_cond[_candidate_mask, : ].mean(0), _cls, proto_idx)
        pass

class NNet_3d(nn.Module):
    def __init__(self,args,in_chns, class_num ,proto_head, output_proto,init_proto,normalization,has_dropout):
        super(NNet_3d, self).__init__()
        self.in_chns = in_chns
        self.class_num = class_num
        self.args = args
        self.proto_head = proto_head
        self.output_proto = output_proto
        self.num_micro_proto = args.num_micro_proto
        self.init_proto = init_proto
        if self.proto_head:
            self.proto_net = CosProto_Module_3D(args=args,in_planes = 16, num_classes =class_num , num_micro_proto = self.num_micro_proto, proto_update_momentum=0.99,init_proto=init_proto)

            self.en_decoder = VNet(n_channels=in_chns, n_classes=class_num, normalization=normalization, has_dropout=has_dropout).cuda()
        else:
            self.en_decoder = VNet(n_channels=in_chns, n_classes=class_num, normalization=normalization, has_dropout=has_dropout).cuda()                                                                    
    
    def forward(self, x, eval=False, unlab_flag=None,label=None, plab=None,pre=False, select_mask=None):
        outs = self.en_decoder(x)
        if eval:
 
            return outs
        
        if self.proto_head and self.output_proto:
            # training phase need to do sampling
            pred_proto,proto_match_idx, contrast_logits= self.proto_net(outs["rep_clf"])
            outs.update({"proto_match_idx": proto_match_idx})

            with torch.no_grad():
                if unlab_flag:
                    #contrast_logits
                    gt_seg=plab*select_mask+label*(1-select_mask)
                    gt_seg = gt_seg.view(-1)
                    
                    
                    pred_seg =torch.max(pred_proto,1)[1]
                    b,h,w,d=pred_seg.shape
                    pred_seg_expanded = pred_seg.unsqueeze(-1)
                    predicted_idx = torch.gather(proto_match_idx.view(b,h,w,d,-1), -1, pred_seg_expanded)
                    predicted_idx = predicted_idx.squeeze(-1).view(-1)
                    
                    pred_seg = pred_seg.view(-1)
                    mask = (gt_seg == pred_seg )
                    mask=mask.view(b,h,w,d)
                    
                    #proto_target = pred_seg .clone().float()
                    proto_target= predicted_idx.float()+self.num_micro_proto*pred_seg

                    probablity= F.softmax(pre, dim=1)
                    confidence_threshold = self.args.confidence_threshold
                    confidence, predicted_class = torch.max(probablity, 1)
                    
                    high_confidence_mask= confidence > confidence_threshold

                    merged_mask_un =mask&high_confidence_mask
                    self.update_proto(outs["rep"], proto_match_idx, plab, select_mask,merged_mask_un)
                    bs,h,w,d=label.shape
                    label_cond = torch.ones(bs, h, w, d,dtype=torch.bool).cuda()
                    merged_mask_l =mask&label_cond
                    self.update_proto(outs["rep"], proto_match_idx, label, 1-select_mask,merged_mask_l)
                    outs.update({"contrast_logits":contrast_logits,"contrast_target":proto_target})
        

            outs.update({"proto": pred_proto})
        return outs
    
    def update_proto(self, rep, proto_match_idx, target, select_mask,cond):
        bs, in_planes, H, W, D = rep.shape
        proto_match_idx = proto_match_idx.reshape(bs, H, W, D, self.class_num)

        rep_selected = rep.permute(0, 2, 3, 4, 1)[:, select_mask == 1]
        bs, hwd, c = rep_selected.shape
        rep_selected = rep_selected.reshape(bs*hwd, c)
        target_selected = target[:, select_mask == 1].view(-1)
        cond = cond[:, select_mask == 1].view(-1)

        valid_mask = target_selected != 0

        rep_selected_valid = rep_selected[valid_mask]
        target_selected_valid = target_selected[valid_mask]

        proto_match_idx_valid = proto_match_idx[:,select_mask==1]
        bs,hwd,c = proto_match_idx_valid.shape
        proto_match_idx_valid = proto_match_idx_valid.reshape(bs*hwd, c)
        proto_match_idx_valid = proto_match_idx_valid[valid_mask]
        cond_valid = cond[valid_mask]

        proto_match_idx_valid_target = proto_match_idx_valid[torch.arange(len(target_selected_valid)), target_selected_valid.long()]
        rep_selected_valid_cond = rep_selected_valid[cond_valid,:]
        proto_match_idx_valid_target_cond = proto_match_idx_valid_target[cond_valid]
        target_selected_valid_cond = target_selected_valid[cond_valid]

        if cond_valid.sum() > 0:
           shot_cls = target_selected_valid_cond.unique()
           for _cls in shot_cls:
              for proto_idx in range(self.num_micro_proto):
                   _candidate_mask = torch.logical_and(target_selected_valid_cond == _cls, proto_match_idx_valid_target_cond == proto_idx)
                   if _candidate_mask.sum() > 0:
                       self.proto_net.update_proto(rep_selected_valid_cond[_candidate_mask, : ].mean(0), _cls, proto_idx)
        pass



def initialize_prototypes(args,model, trainloader, init_path):

    model.eval()

    proto_features = []
    label_list=[]

    for _, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        img_a= volume_batch[:args.labeled_bs]
        lab_a= label_batch[:args.labeled_bs]

        if volume_batch.size(0) != label_batch.size(0):
            raise ValueError("The batch size of images and labels must be the same.")
        
        with torch.no_grad():
            outs = model(img_a)
            proto_features.append(outs["rep_clf"].detach())
            label_list.append(lab_a.detach())
        
    proto_features = torch.cat(proto_features, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # 初始化原型
    total_samples=proto_features.shape[0]
    in_planes = proto_features.shape[1]
    h=proto_features.shape[2]
    w=proto_features.shape[3]
    #pixel2sample
    label_list=label_list.view(-1)
    proto_features=proto_features.permute(0,2,3,1)
    proto_features=proto_features.reshape(total_samples*h*w,-1)


    #按类别将像素分类
    class_protos = [[] for _ in range(args.num_classes)]
    for i in range(proto_features.shape[0]):
        label = label_list[i].item()
        class_protos[label].append(proto_features[i])

    print("len(class_protos)",len(class_protos))#4
    print("class_protos[0][0].shape",len(class_protos[0][0]))#16

    prototypes = []
    labels = []  # 记录每个样本的聚类标签
    for class_pixels in class_protos:
        if len(class_pixels) > 0:
            '''
            kmeans = KMeans(n_clusters=args.num_micro_proto,random_state=args.seed)
            class_pixels_cpu = [tensor.cpu() for tensor in class_pixels]
            kmeans.fit(class_pixels_cpu)
            class_prototypes = kmeans.cluster_centers_
            prototypes.append(class_prototypes)  
            '''
            MiniBatch = MiniBatchKMeans(n_clusters=args.num_micro_proto,random_state=args.seed)
            class_pixels_cpu = [tensor.cpu() for tensor in class_pixels]
            MiniBatch.fit(class_pixels_cpu)
            class_prototypes = MiniBatch.cluster_centers_
            prototypes.append(class_prototypes)



    prototypes_tensor = [torch.from_numpy(proto) for proto in prototypes]
    init_proto = torch.cat(prototypes_tensor, dim=0)
    print("init_proto.shape",init_proto.shape)#(12, 16)
 
    with open(init_path, 'wb') as f:
        pickle.dump(init_proto, f)

def initialize_3D_prototypes(args,model, trainloader, init_path):

    model.eval()

    proto_features = []
    label_list=[]

    for _, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        img_a= volume_batch[:args.labeled_bs]
        lab_a= label_batch[:args.labeled_bs]



        if volume_batch.size(0) != label_batch.size(0):
            raise ValueError("The batch size of images and labels must be the same.")
        

        with torch.no_grad():
            outs = model(img_a)
            proto_features.append(outs["rep_clf"].detach())
            label_list.append(lab_a.detach())
        

    proto_features = torch.cat(proto_features, dim=0)
    label_list = torch.cat(label_list, dim=0)
    print("label_list.shape",label_list.shape)


    total_samples=proto_features.shape[0]
    in_planes = proto_features.shape[1]
    h=proto_features.shape[2]
    w=proto_features.shape[3]
    d=proto_features.shape[4]
    #pixel2sample
    label_list=label_list.view(-1)
    proto_features=proto_features.permute(0,2,3,4,1)
    proto_features=proto_features.reshape(total_samples*h*w*d,-1)
    print("proto_features.shape",proto_features.shape)


    class_protos = [[] for _ in range(args.num_classes)]
    for i in range(proto_features.shape[0]):
        label = label_list[i].item()
        class_protos[label].append(proto_features[i])
    print("len(class_protos)",len(class_protos))#4
    print("class_protos[0][0].shape",len(class_protos[0][0]))

    prototypes = []
    labels = [] 
    for class_pixels in class_protos:
        if len(class_pixels) > 0:
            '''
            kmeans = KMeans(n_clusters=args.num_micro_proto,random_state=args.seed)
            class_pixels_cpu = [tensor.cpu() for tensor in class_pixels]
            kmeans.fit(class_pixels_cpu)
            class_prototypes = kmeans.cluster_centers_
            prototypes.append(class_prototypes)  
            '''
            MiniBatch = MiniBatchKMeans(n_clusters=args.num_micro_proto,random_state=args.seed)
            class_pixels_cpu = [tensor.cpu() for tensor in class_pixels]
            MiniBatch.fit(class_pixels_cpu)
            class_prototypes = MiniBatch.cluster_centers_
            prototypes.append(class_prototypes)


    prototypes_tensor = [torch.from_numpy(proto) for proto in prototypes]
    init_proto = torch.cat(prototypes_tensor, dim=0)
    print("init_proto.shape",init_proto.shape)
    with open(init_path, 'wb') as f:
        pickle.dump(init_proto, f)
