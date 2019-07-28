import os
import os.path as osp
import torch

from mypath import Path
from dataloaders import make_data_loader
from modeling.deeplab import *

class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # Define Tensorboard Summary


        # Define Dataloader
        kwargs = {'num_works': args.works, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.output_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # Whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset))
            if osp.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weights_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLoss(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Resuming checkpoint
        if args.resume is not None
        
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)



        # Clear start epoch if fine-tuning


    def training(self, epoch):
        pass
    
    def validation(self, epoch):
        pass


def main():
    from utils.hyp import args
    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval -  1):
            trainer.validation(epoch)
    
    trainer.writer.close()


if __name__ == "__main__":
    main()