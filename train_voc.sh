CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.007 --workers 4 --epochs 70 --batch-size 4 --checkname deeplab-resnet --eval-interval 1 --dataset voc --resume /home/twsf/work/DeepLab-V3Plus/run/voc/deeplab-resnet/experiment_10/checkpoint.path.tar
