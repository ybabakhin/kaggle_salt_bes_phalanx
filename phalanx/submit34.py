import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='res34', type=str, help='Model version')
parser.add_argument('--save_pred', default='pred/', type=str, help='prediction save space')
parser.add_argument('--pred_path', default='../predictions/pred.npy', type=str, help='final prediction')
args = parser.parse_args()
args.save_pred += args.model + '/'

train_id = pd.read_csv('../data/train.csv')['id'].values
depth_id = pd.read_csv('../data/depths.csv')['id'].values
test_id = np.setdiff1d(depth_id, train_id)

if __name__ == '__main__':
    a = np.load(args.save_pred +'pred0.npy')
    b = np.load(args.save_pred +'pred1.npy')
    c = np.load(args.save_pred +'pred2.npy')
    d = np.load(args.save_pred +'pred3.npy')
    e = np.load(args.save_pred +'pred4.npy')

    a = (a+b+c+d+e) / 5.0
    np.save(args.pred_path, a)
