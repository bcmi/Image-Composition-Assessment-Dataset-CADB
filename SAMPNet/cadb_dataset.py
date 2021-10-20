from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from PIL import Image
import os, json
import torchvision.transforms as transforms
import random
import numpy as np
from config import Config
import cv2

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

random.seed(1)
torch.manual_seed(1)
cv2.setNumThreads(0)

# Refer to: Saliency detection: A spectral residual approach
def detect_saliency(img, scale=6, q_value=0.95, target_size=(224,224)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    W, H = img_gray.shape
    img_resize = cv2.resize(img_gray, (H // scale, W // scale), interpolation=cv2.INTER_AREA)

    myFFT = np.fft.fft2(img_resize)
    myPhase = np.angle(myFFT)
    myLogAmplitude = np.log(np.abs(myFFT) + 0.000001)
    myAvg = cv2.blur(myLogAmplitude, (3, 3))
    mySpectralResidual = myLogAmplitude - myAvg

    m = np.exp(mySpectralResidual) * (np.cos(myPhase) + complex(1j) * np.sin(myPhase))
    saliencyMap = np.abs(np.fft.ifft2(m)) ** 2
    saliencyMap = cv2.GaussianBlur(saliencyMap, (9, 9), 2.5)
    saliencyMap = cv2.resize(saliencyMap, target_size, interpolation=cv2.INTER_LINEAR)
    threshold = np.quantile(saliencyMap.reshape(-1), q_value)
    if threshold > 0:
        saliencyMap[saliencyMap > threshold] = threshold
        saliencyMap = (saliencyMap - saliencyMap.min()) / threshold
    # for debugging
    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(saliencyMap, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return saliencyMap

class CADBDataset(Dataset):
    def __init__(self, split, cfg):
        self.data_path  = cfg.dataset_path
        self.image_path = os.path.join(self.data_path, 'images')
        self.score_path = os.path.join(self.data_path, 'composition_scores.json')
        self.split_path = os.path.join(self.data_path, 'split.json')
        self.attr_path  = os.path.join(self.data_path, 'composition_attributes.json')
        self.weight_path= os.path.join(self.data_path, 'emdloss_weight.json')
        self.split      = split
        self.attr_types = cfg.attribute_types

        self.image_list  = json.load(open(self.split_path, 'r'))[split]
        self.comp_scores = json.load(open(self.score_path, 'r'))
        self.comp_attrs  = json.load(open(self.attr_path,  'r'))
        if self.split == 'train':
            self.image_weight = json.load(open(self.weight_path, 'r'))
        else:
            self.image_weight = None

        self.image_size = cfg.image_size
        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_path, image_name)
        assert os.path.exists(image_file), image_file + ' not found'
        src = Image.open(image_file).convert('RGB')
        im = self.transformer(src)

        score_mean = self.comp_scores[image_name]['mean']
        score_mean = torch.Tensor([score_mean])
        score_dist = self.comp_scores[image_name]['dist']
        score_dist = torch.Tensor(score_dist)

        attrs = torch.tensor(self.get_attribute(image_name))
        src_im  = np.asarray(src).copy()
        sal_map = detect_saliency(src_im, target_size=(self.image_size, self.image_size))
        sal_map = torch.from_numpy(sal_map.astype(np.float32)).unsqueeze(0)

        if self.split == 'train':
            emd_weight = torch.tensor(self.image_weight[image_name])
            return im, score_mean, score_dist, sal_map, attrs, emd_weight
        else:
            return im, score_mean, score_dist, sal_map, attrs

    def get_attribute(self, image_name):
        all_attrs = self.comp_attrs[image_name]
        attrs = [all_attrs[k] for k in self.attr_types]
        return attrs

    def normbboxes(self, bboxes, w, h):
        bboxes = bboxes.astype(np.float32)
        center_x = (bboxes[:,0] + bboxes[:,2]) / 2. / w
        center_y = (bboxes[:,1] + bboxes[:,3]) / 2. / h
        norm_w = (bboxes[:,2] - bboxes[:,0]) / w
        norm_h = (bboxes[:,3] - bboxes[:,1]) / h
        norm_bboxes = np.column_stack((center_x, center_y, norm_w, norm_h))
        norm_bboxes = np.clip(norm_bboxes, 0, 1)
        assert norm_bboxes.shape == bboxes.shape, '{} vs. {}'.format(bboxes.shape, norm_bboxes.shape)
        # print(w,h,bboxes[0],norm_bboxes[0])
        return norm_bboxes

    def scores2dist(self, scores):
        scores = np.array(scores)
        count = [(scores == i).sum() for i in range(1,6)]
        count = np.array(count)
        assert count.sum() == 5, scores
        distribution = count.astype(np.float) / count.sum()
        distribution = distribution.tolist()
        return distribution

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        if self.split == 'train':
            assert (not self.need_image_path) and (not self.need_mask) \
                   and (not self.need_proposal), 'Multi-scale training not implement'
            self.image_size = random.choice(range(self.min_image_size,
                                                  self.max_image_size+1,
                                                  16))
            batch[0] = [resize(im, self.image_size) for im in batch[0]]
        batch = [torch.stack(data, dim=0) for data in batch]
        return  batch


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

if __name__ == '__main__':
    cfg = Config()
    train_dataset = CADBDataset('train', cfg)
    train_loader = DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=True,
                            num_workers=8,
                            drop_last=True,
                            collate_fn=None)
    test_dataset = CADBDataset('test', cfg)
    test_loader = DataLoader(test_dataset,
                            batch_size=2,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)

    print("training set size {}, test set size {}".format(len(train_dataset), len(test_dataset)))
    for batch, data in enumerate(train_loader):
        im,score,dist,mask,attrs,weight = data
        print('train', im.shape, score.shape, dist.shape, mask.shape, attrs.shape, weight.shape)


    for batch, data in enumerate(test_loader):
        im,score,dist,mask,attrs = data
        print('test', im.shape, score.shape, dist.shape, mask.shape, attrs.shape)
        break