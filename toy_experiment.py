import os
import torch
import random
import argparse
import numpy as np
from metric import compute_pprecision_precall, compute_prdc

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_toydataset(dim = 64, datanum = 20000, u = 0, v = 1):
    real = np.random.randn(datanum, dim)
    fake = torch.empty((datanum, dim))
    torch.nn.init.normal_(fake, u, v)
    fake = fake.numpy()

    return real, fake

def main(args):
    """
    Toy-experiment (Gaussian Distribution) for testing evaluation metric
    args:
        setting:
            outlier_f : Investigate the behavior of metric when there exist outlier in real distribution
            outlier_d : Investigate the behavior of metric when there exist outlier in fake distribution
            trade_off : Investigate the behavior of metric when fidelity and diversity trade-off
            hyperparameter : Investigate the behavior of metric for varying hyperparameters
        enable_gpu: Enable GPU calculation of PP&PR (default : False). Use for fast calculation with
                     small dataset size, e.g., 10k, due to memory of GPU.
    
    Original Code for IP&IR and D&C is available at https://github.com/clovaai/generative-evaluation-prdc
    """
    
    random_seed(args.seed)
    
    if args.gpu_enable:
        if torch.cuda_is_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.ngpu
        else:
            raise ValueError('GPU calculation is not valid')
    
    if args.setting == 'outlier_f':
        outlier = -2 + np.random.randn(1, args.dim)
        for sh in np.arange(-3.0, 3.2, 0.2):
            print('Shifting Factor : ', sh)
            real, fake = get_toydataset(args.dim, args.datanum, u = sh)
            real[0] = outlier

            # PP&PR
            p_precision, p_recall = compute_precision_precall(real, fake, a = args.scale, kth = args.kth, gpu = args.gpu_enable)
            print('p_precision : {:.5f}, \t p_recall : {:.5f}'.format(p_precision, p_recall))

            #IP&IR / D&C
            score = compute_prdc(real, fake, args.kth)
            print(score)

    elif args.setting == 'outlier_d':
        outlier = 2 + np.random.randn(1, args.dim)
        for sh in np.arange(-3.0, 3.2, 0.2):
            print('Shifting Factor : ', sh)
            fake, real = get_toydataset(args.dim, args.datanum, u = sh)
            fake[0] = outlier

            # PP&PR
            p_precision, p_recall = compute_pprecision_precall(real, fake, a = args.scale, kth = args.kth, gpu = args.gpu_enable)
            print('p_precision : {:.5f}, \t p_recall : {:.5f}'.format(p_precision, p_recall))

            #IP&IR / D&C
            score = compute_prdc(real, fake, args.kth)
            print(score)

    elif args.setting == 'trade_off':
        for v in np.arange(0.2, 1.6, 0.1):
            print('Variance : ', v)
            real, fake = get_toydataset(args.dim, args.datanum, v = v)

            # PP&PR
            p_precision, p_recall = compute_pprecision_precall(real, fake, a = args.scale, kth = args.kth, gpu = args.gpu_enable)
            print('p_precision : {:.5f}, \t p_recall : {:.5f}'.format(p_precision, p_recall))

            #IP&IR / D&C
            score = compute_prdc(real, fake, args.kth)
            print(score)
            
    elif args.setting == 'hyperparameter':
        a_list = [1.2, 1.5, 1.7, 2.0]
        k_list = [2, 3, 4, 5, 6]
        for a in a_list:
            print('a : ', a)
            for k in k_list:
                print('k : ', k)
                real, fake = get_toydataset(args.dim, args.datanum)
                p_precision, p_recall = compute_pprecision_precall(real, fake, a = a, kth = k, gpu = args.gpu_enable)
                print('p_precision : {:.5f}, \t p_recall : {:.5f}'.format(p_precision,p_recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toy-experiment with Evaluation Metrics')
    parser.add_argument('--seed', type = int, default = 777, help = 'Fixing the seed')
    parser.add_argument('--dim', type = int, default = 64, help = 'Dimension for feature embedding')
    parser.add_argument('--datanum', type = int, default = 10000, help = 'Number of dataset')
    parser.add_argument('--kth', type = int, default = 4, help = 'hyperparameter for KNN')
    parser.add_argument('--scale', type = int, default= 1.2, help = 'hyperparameter for hypersphere f')
    parser.add_argument('--setting', type = str, default= 'outlier_f', help = 'settings for toy-experiment')
    parser.add_argument('--gpu_enable', action = 'store_true', default= False, help = 'Enable for GPU calculation (default = False)')
    parser.add_argument('--ngpu', type = str, default = '0', help = 'GPU device number')
    args = parser.parse_args()

    main(args)

        
