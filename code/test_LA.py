import os
import argparse
import torch
import pdb
import torch.nn as nn

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../Datasets/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str,  default='VNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--num_micro_proto', type=int,  default=3, help='number of micro prototypes')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
#prototype
parser.add_argument('--proto_head', type=int,  default=1, help='whether use prototype head')
parser.add_argument('--proto_unpdate_momentum', type=float,  default=0.999, help='prototype update momentum')
parser.add_argument('--confidence_threshold', type=float,  default=0.9, help='confidence threshold')
parser.add_argument('--loss_ppc_weight', type=float,  default=0.02, help='weight of pixel prototype classification loss')
parser.add_argument('--loss_ppd_weight', type=float,  default=0.2, help='weight of pixel prototype distance loss')
parser.add_argument('--temperature', type=float,  default=0.1, help='temperature of contrastive loss')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/model_LA_PPC_{}/LA_{}_{}_labeled/{}".format(FLAGS.loss_ppc_weight,FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
test_save_path = "../model/model_LA_PPC_{}/LA_{}_{}_labeled/{}_predictions/".format(FLAGS.loss_ppc_weight,FLAGS.exp, FLAGS.labelnum, FLAGS.model)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric():
    model = net_factory(args=FLAGS,net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")

    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    checkpoint=torch.load(save_model_path)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    proto_list_tensor = checkpoint['proto_list']
    proto_list_parameter = nn.Parameter(proto_list_tensor)


    model.proto_net.proto_list = proto_list_parameter

    model.proto_net.proto_list.requires_grad = False
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    model.load_state_dict(torch.load(save_model_path),strict=False)
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric

import os

if __name__ == '__main__':
    if os.getenv('CAPTURE_METRIC_ONLY'):
        metric = test_calculate_metric()
        print(metric)
    else:
        metric = test_calculate_metric()
        # 其他输出
        print("其他信息")
        print(metric)

# python test_LA.py --model 0214_re01 --gpu 0
