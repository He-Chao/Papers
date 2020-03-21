from trainers import *
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main():
    opts = BaseOptions()
    args = opts.parse()
    
    # args.resume = '../debug/checkpoints.pth' #断点训练，不可用
    #是否进行预训练
    args.pretrain = True
    if args.pretrain:
        args.batch_size = 256
        args.epochs = 60
        args.lr = 1e-3
        args.scala_ce = 1
        args.pretrain_path = '../data/resnet50-19c8e357.pth'
        
    logger = Logger(args.save_path) #创建日志文件
    opts.print_options(logger) #打印日志文件中的参数

    #加载数据集
    source_loader, _, _, _ = get_transfer_dataloaders(args.source, args.target, args.img_size, args.crop_size, args.padding, args.batch_size // 2, False)
    args.num_classes = 4101 #辅助数据集的ID数

    if args.resume:
        #断点训练
        trainer, start_epoch = load_checkpoint(args, logger)
    else:
        trainer = ReidTrainer(args, logger) #类class
        start_epoch = 0

    total_epoch = args.epochs

    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, total_epoch):

        #计算需要的时间
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - epoch))
        need_time = 'Stage 1, [Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger.print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, total_epoch, need_time))
        
        #Train
        meters_trn = trainer.pre_MSMT(source_loader, epoch)
        logger.print_log('  **Train**  ' + create_stat_string(meters_trn))

        epoch_time.update(time.time() - start_time)
        start_time = time.time()


if __name__ == '__main__':
    main()
