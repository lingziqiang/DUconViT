import os


def make_train_list(train_data_path, out_path):
    with open(out_path, 'w') as f:
        for name in os.listdir(train_data_path):
            f.write(name.split('.')[0]+'\n')

def make_test_vol_list(test_path, out_path):
    with open(out_path, 'w') as f:
        for name in os.listdir(test_path):
            f.write(name.split('.')[0]+'\n')
if __name__=='__main__':
    out_path = './lists/train.txt'
    make_train_list('./npz_h5_data/train_npz', out_path)

    out_path2 = './lists/test_vol.txt'
    make_train_list('./npz_h5_data/test_vol_h5', out_path2)