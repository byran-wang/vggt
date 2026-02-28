from glob import glob
from PIL import Image
import numpy as np
import os
import numpy as np
import os.path as op
from tqdm import tqdm
import sys
import pickle

sys.path = ['../code'] + sys.path


def object_intrinsic_gen(args):
    seq_name = args.seq_name
    out_name = args.out_name
    print(f"Processing {seq_name}")


    intrinsic_f = args.intrinsic_f
    meta = pickle.load(open(intrinsic_f,'rb'))
    K = meta['camMat']
    fmt = '%.12f'
    os.makedirs(f'data/{seq_name}/processed/{out_name}', exist_ok=True)
    np.savetxt(f'data/{seq_name}/processed/{out_name}/intrins.txt', K, fmt=fmt, delimiter=' ')
    print('Done!')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsic_f", type=str, default=None)
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    object_intrinsic_gen(args)
