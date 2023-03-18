import os
import time
from options import TrainOptions, TestOptions
from framework import SketchModel
from utils import load_data
from writer import Writer
from evaluate import run_eval
import numpy as np
# import torchsnooper


def run_train(train_params=None, test_params=None):
    opt = TrainOptions().parse(train_params)
    testopt = TestOptions().parse(test_params)
    testopt.timestamp = opt.timestamp
    testopt.batch_size = opt.batch_size

    # model init
    model = SketchModel(opt)
    model.print_detail()

    writer = Writer(opt)

    # data load
    trainDataloader = load_data(opt, datasetType='train', shuffle=opt.shuffle)
    trainDataloader_2 = load_data(opt, datasetType='train', shuffle=True)
    trainDataloader_3 = load_data(opt, datasetType='train', shuffle=True)
    testDataloader = load_data(opt, datasetType='test')

    # train epoches
    # with torchsnooper.snoop():
    ii = 0
    min_test_avgloss = 100
    min_test_avgloss_epoch = 0
    best_p_metric = 0
    best_p_metric_epoch = 0
    best_c_metric = 0
    best_c_metric_epoch = 0
    for epoch in range(opt.epoch):
        data_last = None
        
        for i, (data, data_2, data_3) in enumerate(zip(trainDataloader, trainDataloader_2, trainDataloader_3)):
            if i == 0:
                data_last = data_3
            else:
                model.step_pi(data_2, data_last, data_3)
                data_last = data_3
            model.step_theta(data)

            if ii % opt.plot_freq == 0:
                writer.plot_train_loss(model.loss, ii)
            if ii % opt.print_freq == 0:
                writer.print_train_loss(epoch, i, model.loss)

            ii += 1

        model.update_learning_rate()
        if opt.plot_weights:
            writer.plot_model_wts(model, epoch)
        
        # test
        if epoch % opt.run_test_freq == 0:
            model.save_network('latest')
            loss_avg, P_metric, C_metric = run_eval(
                opt=testopt,
                loader=testDataloader, 
                dataset='test',
                write_result=False)
            writer.print_test_loss(epoch, 0, loss_avg)
            writer.plot_test_loss(loss_avg, epoch)
            writer.print_eval_metric(epoch, P_metric, C_metric)
            writer.plot_eval_metric(epoch, P_metric, C_metric)
            if loss_avg < min_test_avgloss:
                min_test_avgloss = loss_avg
                min_test_avgloss_epoch = epoch
                print('saving the model at the end of epoch {} with test best avgLoss {}'.format(epoch, min_test_avgloss))
                model.save_network('bestloss')
            
            if C_metric > best_c_metric:
                best_c_metric = C_metric
                best_c_metric_epoch = epoch
                print('saving the model at the end of epoch {} with test best C_metric {}'.format(epoch, best_c_metric))
                model.save_network('best_c')
            
            if P_metric > best_p_metric:
                best_p_metric = P_metric
                best_p_metric_epoch = epoch
                print('saving the model at the end of epoch {} with test best P_metric {}'.format(epoch, best_p_metric))
                model.save_network('best_p')

    testopt.which_epoch = 'latest'
    testopt.metric_way = 'wlen'
    loss_avg, P_metric, C_metric = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)
    
    testopt.which_epoch = 'bestloss'
    testopt.metric_way = 'wlen'
    loss_avg_2, P_metric_2, C_metric_2 = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)
    
    testopt.which_epoch = 'best_c'
    testopt.metric_way = 'wlen'
    loss_avg_3, P_metric_3, C_metric_3 = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)

    testopt.which_epoch = 'best_p'
    testopt.metric_way = 'wlen'
    loss_avg_4, P_metric_4, C_metric_4 = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)

    record_list = {
        'p_metric': round(P_metric*100, 2),
        'c_metric': round(C_metric*100, 2),
        'loss_avg': round(loss_avg, 4),
        'best_epoch': min_test_avgloss_epoch,
        'p_metric_2': round(P_metric_2*100, 2),
        'c_metric_2': round(C_metric_2*100, 2),
        'loss_avg_2': round(loss_avg_2, 4),
        'best_c_epoch': best_c_metric_epoch,
        'p_metric_3': round(P_metric_3*100, 2),
        'c_metric_3': round(C_metric_3*100, 2),
        'loss_avg_3': round(loss_avg_3, 4),
        'best_p_epoch': best_p_metric_epoch,
        'p_metric_4': round(P_metric_4*100, 2),
        'c_metric_4': round(C_metric_4*100, 2),
        'loss_avg_4': round(loss_avg_4, 4),
    }
    writer.train_record(record_list=record_list)
    writer.close()
    
    return record_list, opt.timestamp, opt.creativity_alpha, opt.creativity_beta

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    record_list, _, creativity_alpha, creativity_beta = run_train()
    print(record_list)
    print("creativity_alpha: ", creativity_alpha)
    print("creativity_beta: ", creativity_beta)
