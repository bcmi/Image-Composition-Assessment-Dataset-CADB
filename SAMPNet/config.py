import os,time

class Config:
    # setting for dataset and dataloader
    dataset_path = '/workspace/composition/CADB_Dataset'
    assert os.path.exists(dataset_path), dataset_path + 'not found'

    batch_size = 16
    gpu_id = 0
    num_workers = 8

    # setting for training and optimization
    max_epoch = 50
    save_epoch = 1
    display_steps = 10
    score_level = 5

    use_weighted_loss = True
    use_attribute = True
    use_channel_attention = True
    use_saliency = True
    use_multipattern = True
    use_pattern_weight = True
    # AADB attributes: "Light" "Symmetry" "Object"
    # "score" "RuleOfThirds" "Repetition"
    # "BalacingElements" "ColorHarmony" "MotionBlur"
    # "VividColor" "DoF" "Content"
    attribute_types = ['RuleOfThirds', 'BalacingElements','DoF',
                       'Object', 'Symmetry', 'Repetition']
    num_attributes = len(attribute_types)
    attribute_weight = 0.1

    optimizer = 'adam' # or sgd
    lr = 1e-4
    weight_decay = 5e-5
    momentum = 0.9
    # setting for cnn
    image_size = 224
    resnet_layers = 18
    dropout = 0.5
    pool_dropout = 0.5
    pattern_list = [1, 2, 3, 4, 5, 6, 7, 8]
    pattern_fuse = 'sum'

    # setting for testing
    test_epoch = 1

    # setting for expriments
    if len(pattern_list) == 1:
        exp_root = os.path.join(os.getcwd(), './experiments/single_pattern')
        prefix = 'pattern{}'.format(pattern_list[0])
    else:
        exp_root = os.path.join(os.getcwd(), './experiments/')
        prefix = 'resnet{}'.format(resnet_layers)
        if use_multipattern:
            if use_pattern_weight and use_saliency:
                prefix += '_samp'
            elif use_pattern_weight:
                prefix += '_weighted_mpp'
            elif use_saliency:
                prefix += '_saliency_mpp'
            else:
                prefix += '_mpp'
        if use_attribute:
            if use_channel_attention:
                prefix += '_aaff'
            else:
                prefix += '_attr'
        if use_weighted_loss:
            prefix += '_wemd'
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + str(index)
        exp_path = os.path.join (exp_root, exp_name)
    print('Experiment name {} \n'.format(os.path.basename(exp_path)))
    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

if __name__ == '__main__':
    cfg = Config()