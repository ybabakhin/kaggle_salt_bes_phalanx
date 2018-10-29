from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd

from utils import RLenc

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='res34', type=str, help='Model version')
parser.add_argument('--save_pred', default='../pred/', type=str, help='prediction save space')
args = parser.parse_args()
args.save_pred += args.model + '/'

train_id = pd.read_csv('../input/train.csv')['id'].values
depth_id = pd.read_csv('../input/depths.csv')['id'].values
test_id = np.setdiff1d(depth_id, train_id)

if __name__ == '__main__':
    a = np.load(args.save_pred +'pred0.npy')
    b = np.load(args.save_pred +'pred1.npy')
    c = np.load(args.save_pred +'pred2.npy')
    d = np.load(args.save_pred +'pred3.npy')
    e = np.load(args.save_pred +'pred4.npy')

    a = (a+b+c+d+e) / 5.0
    np.save(args.save+'all_fold_prediction', a)
    
    #a = np.where(a >= 0.52, 1, 0)
    #pred_dict = {fn: RLenc(np.round(a[i])) for i , fn in tqdm(enumerate(test_id))}
    #sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    #sub.index.names = ['id']
    #sub.columns = ['rle_mask']
    #sub.to_csv('./submission_' + args.model + '.csv')
