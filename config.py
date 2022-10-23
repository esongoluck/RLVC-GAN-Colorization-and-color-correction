import numpy as np
Config = {
    #'distribute_strategy': tf.distribute.OneDeviceStrategy(device='/gpu:0'),
    #'set_visible_devices':([list_physical_devices("GPU")[0]],"GPU"),
    'device':'cuda:0',#  'cuda:0',或 'cpu'

    'mood':"train", #train / test

    'epochs': 50,
    'batch_size': 1,
    'D_lr': 0.0002,
    'G_lr': 0.0002,
    'C_lr': 0.0002,
    'image_size':512,##GPU使用量为16*batch_size*image_size*image_size B
    'error_size':8,
    'date_path':"my_dataset",

    'img_depth': 8,
    'mid_L':0.1,
    'size_blur':3,

    'x_path':"/img-ir",
    'y_path':"/img-rgb",

    'train_csv': 1,  # 设置是否初始化csv文件ssh -p 35548 root@region-3.autodl.c
    'val_csv': 1,
    'test_csv': 1,

    'checkpoint_save_path' : "./checkpoint/ResNet18.ckpt",
    'image_list': './demo_outputs',
    'model_dir': 'model_files',
    'tensorboard_log_dir': 'logs',
    'checkpoint_prefix': 'ckpt',
    'restore_parameters': False,
    'mode': 'train',
    'MD_sig':'False',#True为模型中使用sigmoid函数
    'half':'True',#True为使用半精度计算
    'log_i':'None',
    'stat':'False',
    'x_corners' : np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]),  # 左上 右上    右下  左下
    'y_corners' : np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]),
    'the_corners' : np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]),
    'be_corners' : np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]),
    'tr_mat_x' : [],
    'tr_mat_y' : [],
}
