class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return 'C:\\Users\\TWSF\\Desktop\\ResNet\\data\\VOC2012'
        elif dataset == 'coco':
            return ''
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
    
    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return 'C:\\Users\\TWSF\\Desktop\\ResNet\\weights\\resnet101.pth'
        else:
            print('weights {} not available.'.format(backbone))
            raise NotImplementedError