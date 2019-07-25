import torch


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


        # Define Criterion

        
        # Define Evaluator

        # Define lr scheduler

        
        # Using cuda


        # Resuming checkpoint


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