from trainers import *
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main():
    opts = BaseOptions()
    args = opts.parse()
    logger = Logger(args.save_path) #创建日志文件
    opts.print_options(logger) #打印日志文件中的参数
    
    # args.resume = '../debug/checkpoints.pth' #断点训练，不可用

    #加载数据集
    source_loader, target_loader, gallery_loader, probe_loader = get_transfer_dataloaders(args.source, args.target, args.img_size, args.crop_size, args.padding, args.batch_size // 2, False)
    args.num_classes = 4101 #辅助数据集的ID数
    '''
    辅助数据集:MSMT17.mat：124068张图片，4101个人,只在训练阶段使用
    目标数据集:Market.mat: 32217,1501个人
        - target_data : (12936, 3, 128, 384) 训练数据集
        - gallery_data: (15913, 3, 128, 384) 测试数据集
        - probe_data  : (3368 , 3, 128, 384) 查询数据集
    '''

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
        meters_trn = trainer.train_epoch(source_loader, target_loader, epoch)
        logger.print_log('  **Train**  ' + create_stat_string(meters_trn))

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #Test
        meters_val = trainer.eval_performance(target_loader, gallery_loader, probe_loader)
        logger.print_log('  **Test**  ' + create_stat_string(meters_val))


if __name__ == '__main__':
    main()
