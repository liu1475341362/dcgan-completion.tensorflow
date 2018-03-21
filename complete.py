#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

#LBN  2018年3月16日 11:17:39
# 整个代码作用详细参见paper文件夹里面的python：argparse
# 用argparse来定义一些默认命令，通常是全局变量，也是用作和系统命令之间交互的全局设置



import argparse
import os
import tensorflow as tf
import itertools
from glob import glob



from model import DCGAN

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


parser = argparse.ArgumentParser()

parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')   #修复完的图片的文件夹
parser.add_argument('--outInterval', type=int, default=50)
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres'],
                    default='center')
parser.add_argument('--centerScale', type=float, default=0.25)
parser.add_argument('--imgs',type=str, nargs='+')      #parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #True，这个是原来的参数，因为没有gpu，改成0


#整个项目代码从这个地方开始，DCGAN在model文件里面，快捷键ctrl+b
with tf.Session(config=config) as sess:
    # batch_size=min(64, len(args.imgs)),

    #读入我的测试文件
    args.imgs = dataset_files("F:\Python_Project\Dataset\Test")

    dcgan = DCGAN(sess, image_size=args.imgSize,
                  batch_size=min(64, len(args.imgs)),
                  checkpoint_dir=args.checkpointDir, lam=args.lam)

    dcgan.complete(args)

