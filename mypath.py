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