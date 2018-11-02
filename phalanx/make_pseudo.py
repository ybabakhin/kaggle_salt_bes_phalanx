import numpy as np
import pandas as pd

train_id = pd.read_csv('../data/train.csv')['id'].values
depth_id = pd.read_csv('../data/depths.csv')['id'].values
test_id = np.setdiff1d(depth_id, train_id)

if __name__ == '__main__':
    df = pd.read_csv('../data/pseudolabels.csv')
    pseudo_id = df['id'].values
    pseudo_pix = df['mask_pixels'].values
    pseudo_conf = df['confidence'].values
    pseudo_unique = df['unique_pixels'].values

    split0 = [] # salt_size == 0
    split1 = [] # 20 <= salt_size <= 500
    split2 = [] # 500 < salt_size <= 101*101/4
    split3 = [] # 101*101/4 < salt_size <= 101*101/2
    split4 = [] # 101*101/2 < salt_size <= 101*101*3/4
    split5 = [] # 101*101*3/4 < salt_size < 101*101*0.9

    for idx, mask_id in enumerate(pseudo_id):
        if pseudo_conf[idx] >= 0.97:
            if pseudo_pix[idx] == 0:
                if pseudo_unique[idx] != 1:
                    split0.append(mask_id+'.png')
            elif 20 <= pseudo_pix[idx] and pseudo_pix[idx] <= 500:
                if pseudo_conf[idx] >= 0.99:
                    split1.append(mask_id+'.png')
            elif 500 < pseudo_pix[idx] and pseudo_pix[idx] <= 101*101/4:
                split2.append(mask_id+'.png')
            elif 101*101/4 < pseudo_pix[idx] and pseudo_pix[idx] <= 101*101/2:
                split3.append(mask_id+'.png')
            elif 101*101/2 < pseudo_pix[idx] and pseudo_pix[idx] <= 101*101*3/4:
                split4.append(mask_id+'.png')
            elif 101*101*3/4 < pseudo_pix[idx] and pseudo_pix[idx] < 101*101 * 0.9:
                split5.append(mask_id+'.png')
    split0 = split0[::2]

    fold0 = split0[:600] + split1[:110] + split2[:300] + split3[:240] + split4[:200] + split5[:130]
    fold1 = split0[600:1200] + split1[110:220] + split2[300:600] + split3[240:480] + split4[200:400] + split5[130:260]
    fold2 = split0[1200:1800] + split1[220:330] + split2[600:900] + split3[480:740] + split4[400:600] + split5[260:390]
    fold3 = split0[1800:2400] + split1[330:440] + split2[900:1200] + split3[740:980] + split4[600:800] + split5[390:520]
    fold4 = split0[2400:] + split1[440:] + split2[1200:] + split3[980:] + split4[800:] + split5[520:]

    np.save('../data/stage2_fold0', fold0)
    np.save('../data/stage2_fold1', fold1)
    np.save('../data/stage2_fold2', fold2)
    np.save('../data/stage2_fold3', fold3)
    np.save('../data/stage2_fold4', fold4)