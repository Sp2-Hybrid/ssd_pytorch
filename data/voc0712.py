"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
#ET.parse(xml_file_path).getroot() 获取第一标签
'''
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
'''
VOC_CLASSES = (
    'core',
    'coreless'
)

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join('/home', 'songpeng', 'ssd', 'ssd.pytorch.2', 'data', 'VOCdevkit',)


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, targets, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for line in targets:
            line = line.strip()
            line = line.split()
            if line == "":
                continue
            
            if line[1] == '带电芯充电宝':
                name  = 'core' 
            elif line[1] == '不带电芯充电宝':
                name = "coreless" 
            else:
                continue
            bndbox = []
            xmin = int(line[2]) -1 
            ymin = int(line[3]) -1
            xmax = int(line[4]) -1 
            ymax = int(line[5]) -1
            
            xmin = xmin / width
            xmax = xmax / width
            ymin = ymin / height
            ymax = ymax / height 
            bndbox.append(xmin)
            bndbox.append(xmax)
            bndbox.append(ymin)
            bndbox.append(ymax)
            
            label_idx = self.class_to_ind[name] 
            bndbox.append(label_idx)
            res += [bndbox]
        return res


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        if image_sets[0][1] == 'test':
            self._annopath = osp.join('%s', 'eval', 'Anno_test', '%s.txt')
            self._imgpath = osp.join('%s', 'eval', 'Image_test', '%s.jpg')
            self.ids = list()
            for (year, name) in image_sets:
                for line in open(osp.join(self.root, 'eval', 'core_coreless_test.txt')):
                    self.ids.append((self.root, line.strip()))
        else:
            self._annopath = osp.join('%s', 'Annotations_txt', '%s.txt')
            self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
            
            self.ids = list()
            for (year, name) in image_sets:
                #rootpath = osp.join(self.root, 'VOC' + year)
                rootpath = osp.join(self.root, 'VOC')
                for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((rootpath, line.strip()))
        

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        #target = ET.parse(self._annopath % img_id).getroot()
        with open(self._annopath % img_id, 'r', encoding='utf-8') as f:
            target = f.readlines()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            #调用__call__
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        with open(self._annopath % img_id, 'r', encoding='utf-8') as f:
            anno = f.readlines()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
