class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return '/home/twsf/data/VOC2012'
            # return '/home/twsf/work/DeepLab-V3Plus/data/VOC2012'
        elif dataset == 'coco':
            return ''
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
    
    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return "/home/twsf/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth"
        else:
            print('weights {} not available.'.format(backbone))