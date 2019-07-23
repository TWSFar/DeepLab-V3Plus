from dataloaders.datasets import coco, voc
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset == 'voc':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            raise NotImplementedError

        num_classes = trian_set.NUM_CLASSES
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  **kwargs=kwargs)
        val_loader = DataLoader(val_loader,
                                batch_size=args.batch_size,
                                shuffle=False,
                                **kwargs=kwargs)
        test_loader = None
    
        return train_loader, val_loader, test_loader, num_classes


    elif args.dataset == 'coco':
        raise NotImplementedError
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    
