# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle
import logging
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import data_handler
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.tensorboard import SummaryWriter # JULIA

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode, log=False, julia_datasetname='ICEWS18', julia_multistep_bool=True, julia_exp_nr=0):
    """
    :param model: model used to test
    :param history_list: all input history snap shot list, not include output label train list or valid list
    :param test_list: test triple snap shot list
    :param num_rels:  number of relations
    :param num_nodes: number of nodes
    :param use_cuda:
    :param all_ans_list: dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list: dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    ### ADDED JULIA
    #for logging scores
    import inspect
    import sys
    import os
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # parentdir = os.path.dirname(currentdir)
    import evaluation_utils 
    sys.path.insert(1, currentdir) 
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))        
    
    exp_nr =julia_exp_nr #seed
    if julia_multistep_bool== True:
        steps = 'multistep'        
    else:
        steps ='singlestep'

    method = 'regcn'
    filter = 'raw'
    logname = method + '-' + julia_datasetname + '-' +str(exp_nr) + '-' +steps + '-' + filter
    #renet-ICEWS18-multistep-raw-modifiedpredict_xxx_1_3206_6624.pt
    ## END ADDED JULIA

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    new_input_list = []
    for inp in input_list:
        # mask = (inp[:, 1] == 4)  | (inp[:, 1] == 8)  | (inp[:, 1] == 23) | (inp[:, 1] == 27) #4,8,23,27
        # new_inp = inp[mask]
        new_inp = inp
        new_input_list.append(new_inp)

    input_list = new_input_list

    julia_logging_dict = {}
    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        # mask = (test_triples_input[:, 1] == 0)  | (test_triples_input[:, 1] == 5)
        # test_triples_input = test_triples_input[mask]
        if len(test_triples_input) > 0:
            # queries_of_interest = output[0][mask]
            test_triples_input = test_triples_input.to(args.gpu)

            test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, use_cuda, subbatch=10)
            #TODO coypu: hier die test_tiples, final_score und final_r_score extrahieren die ich brauche; bzw einfach test_triples_input anpassen

            ## ADDED JULIA
            # # # logging scores
            if log == True:
                print("logging results")
                for triple, subobscores in zip(test_triples, final_score):
                    quad = triple.tolist()
                    quad.append(time_idx)

                    query_name, gt_test_query_ids = evaluation_utils.query_name_from_quadruple(quad, num_rels)

                    julia_logging_dict[query_name] = [subobscores.cpu().detach().numpy(), gt_test_query_ids]# liste mit element 0: scores, element 1:gt 
                    # evaluation_utils.store_scores(currentdir, logname,  query_name, julia_datasetname, gt_test_query_ids, subobscores)

            ## END ADDED JULIA
            mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

            # used to global statistic
            ranks_raw.append(rank_raw)
            ranks_filter.append(rank_filter)
            # used to show slide results
            mrr_raw_list.append(mrr_snap)
            mrr_filter_list.append(mrr_filter_snap)

            # relation rank
            ranks_raw_r.append(rank_raw_r)
            ranks_filter_r.append(rank_filter_r)
            mrr_raw_list_r.append(mrr_snap_r)
            mrr_filter_list_r.append(mrr_filter_snap_r)

            # reconstruct history graph list
            if args.multi_step:
                if not args.relation_evaluation:    
                    predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk) #TODO: HERE THEY PREDICT A FULL GRAPH?!
                else:
                    predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
                if len(predicted_snap):
                    input_list.pop(0)
                    input_list.append(predicted_snap)
            else:
                input_list.pop(0)
                input_list.append(test_snap)
        else:
            # print("no relations of interest in snapshot: ", time_idx)
            a = time_idx
        idx += 1
    mrr_raw = 0
    mrr_raw = utils.stat_ranks(ranks_raw, "Entity Prediction Raw", log)
    mrr_filter = utils.stat_ranks(ranks_filter, "Entity Prediction Filter", log)
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "Relation Prediction Raw", log)
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "Relation Prediction Filter", log) 

    #JULIA
    if log == True:
        import pathlib
        dirname = os.path.join(pathlib.Path().resolve(), 'results' )
        juliafilename = os.path.join(dirname, logname + ".pkl")
        print('dumping pkl file to: ', juliafilename)
        # if not os.path.isfile(juliafilename):
        with open(juliafilename,'wb') as file:
            pickle.dump(julia_logging_dict, file, protocol=4) 
        file.close()
    #END JULIA

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None, train_history_len=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if train_history_len:
        args.train_history_len = train_history_len
        args.test_history_len = train_history_len

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    logging.debug('Train: {}\t Valid: {}\t Test: {}'.format(len(data.train), len(data.valid), len(data.test)))

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    # model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}.pth"\
    #     .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len,
    #             args.weight, args.discount, args.angle, args.dropout, args.input_dropout, args.hidden_dropout,
    #             args.feat_dropout, args.gpu)
    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight-{}-discount-{}-angle-{}-dp{}-{}-{}-{}-gpu{}-run{}.pth"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len,
                args.weight, args.discount, args.angle, args.dropout, args.input_dropout, args.hidden_dropout,
                args.feat_dropout, args.gpu, args.runnr)  #CHANGED JULIA


    model_state_file = os.getcwd()[:-4] + '/models/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                            num_nodes,
                            num_rels,
                            num_static_rels,
                            num_words,
                            args.n_hidden,
                            args.opn,
                            sequence_len=args.train_history_len,
                            num_bases=args.n_bases,
                            num_basis=args.n_basis,
                            num_hidden_layers=args.n_layers,
                            dropout=args.dropout,
                            self_loop=args.self_loop,
                            skip_connect=args.skip_connect,
                            layer_norm=args.layer_norm,
                            input_dropout=args.input_dropout,
                            hidden_dropout=args.hidden_dropout,
                            feat_dropout=args.feat_dropout,
                            aggregation=args.aggregation,
                            weight=args.weight,
                            discount=args.discount,
                            angle=args.angle,
                            use_static=args.add_static_graph,
                            entity_prediction=args.entity_prediction,
                            relation_prediction=args.relation_prediction,
                            use_cuda=use_cuda,
                            gpu = args.gpu,
                            analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    ## JULIA
    writer = SummaryWriter(comment=model_name)  # setup the tensorboard writer path # JULIA
    mrrs = []
    epoch = 0
    if args.multi_step:
        print('julia multistep multistep')
        julia_multistep_bool =True
    else:
        print('julia multistep not multistep')
        julia_multistep_bool = False
    ## END

    if args.test and os.path.exists(model_state_file):
        print(args.dataset)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list, 
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph, 
                                                            "test",
                                                            log=True,
                                                            julia_datasetname=args.dataset, 
                                                            julia_multistep_bool=julia_multistep_bool, 
                                                            julia_exp_nr=args.runnr) #this line was added by julia for logging
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        print("number of epochs: ", args.n_epochs)
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []
            valid_losses_e = []
            valid_losses_r = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1] #quasi naechster zeitschritt oder?! also target graph
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                input_list_backup = input_list
                new_input_list = []
                
                for inp in input_list:
                    # mask = (inp[:, 1] == 4)  | (inp[:, 1] == 8)  | (inp[:, 1] == 23) | (inp[:, 1] == 27)
                    # new_inp = inp[mask]
                    new_inp = inp #[mask]
                    new_input_list.append(new_inp)

                input_list = new_input_list


                # generate history graph #the history-glist has for each of the train_sample_num (6) timesteps a grraph num+edges = 2*num_triples; aber: brauchen wir keine self-loops??
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                # mask = (output[0][:, 1] == 0)  | (output[0][:, 1] == 5)
                # queries_of_interest = output[0][mask]
                queries_of_interest = output[0]
                if len(queries_of_interest) > 0:
                    # TODO coypu: ich muss glaube ich nur output[0] anpassen hier!
                    loss_e, loss_r, loss_static = model.get_loss(history_glist, queries_of_interest, static_graph, use_cuda)
                    loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static
                    

                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                else:

                    # print("no relations of interest in snapshot: ", train_sample_num)     
                    a = train_sample_num         

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            
            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    use_cuda, 
                                                                    all_ans_list_valid, 
                                                                    all_ans_list_r_valid, 
                                                                    model_state_file, 
                                                                    static_graph, 
                                                                    mode="train") #TODO: julia added , test_loss_e, test_loss_r

                mrrs.append(mrr_raw.item()) #JULIA
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            print("BREAKING BEC mrr_raw smaller best mrr, epoch ", epoch)
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                        print('Model Saved')
                        time.sleep(5)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            print("BREAKING BEC mrr_raw_r smaller best mrr, epoch ", epoch)
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                        print('Model Saved')
                        time.sleep(5)

                ##JULIA
                writer.add_scalar('Mean training loss', np.mean(losses), epoch)
                writer.add_scalar('Mean training loss_e', np.mean(losses_e), epoch)
                writer.add_scalar('Mean training loss_r', np.mean(losses_r), epoch)
                writer.add_scalar('Mean training loss_s', np.mean(losses_static), epoch)
                writer.add_scalar('Mean valid loss_e', np.mean(valid_losses_e), epoch)
                writer.add_scalar('Mean valid loss_r', np.mean(valid_losses_r), epoch)
                writer.add_scalar('Mean MRR validation', np.mean(mrrs), epoch)
                writer.add_scalar('MRR', mrr_raw, epoch)

        validation_mrr_raw = best_mrr #added julia
                #END  JULIA
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph, 
                                                            mode="test",
                                                            log=True,
                                                            julia_datasetname=args.dataset, 
                                                            julia_multistep_bool=julia_multistep_bool, 
                                                            julia_exp_nr=args.runnr)
        test_mrr = mrr_raw
    writer.close()
    print('finished after epoch ', epoch)
    return validation_mrr_raw, test_mrr


if __name__ == '__main__':

    n = 'RE-GCN'
    log_dir = f'../../logs/{n}.log'
    logging.basicConfig(filename=log_dir, filemode='a',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='REGCN')

    logging.debug('Note: Results for YAGO, WIKI and GDELT have been reported without the static graph constraint as they are missing in the dataset itself.')

    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1, #500,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str,  default='crisis2023', #ICEWS14', #required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False, 
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False, 
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1.0,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200, #200, #embedding dim emb-dim
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=True, #default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True, #default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=100, #500, #500,#default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,#default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int,  default=7,  #6,# default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int,  default=7, # 6,#default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases,train_history_len",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")
    ##ADDED JULIA
    # args for logging the scores
    parser.add_argument("--runnr", default=0, type=int) 
    ##END ADDED JULIA

    args = parser.parse_args()
    # args.test = True
    print(args)
    if args.grid_search:
        
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')
    
        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()
    
        for i, grid_entry in enumerate(list(grid)):
            start_time = time.time()
            o_f = open(out_log, 'a')
            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")

            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            validation_mrr_raw, test_mrr = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4])
            print("MRR (raw) valid: {:.6f}".format(validation_mrr_raw))
            print("MRR (raw) test: {:.6f}".format( test_mrr))
            o_f.write("MRR (raw) valid: {:.6f}\n".format(validation_mrr_raw))
            o_f.write("MRR (raw) test: {:.6f}\n".format(test_mrr))
            # for hit in hits:
            #     avg_count = torch.mean((ranks <= hit).float())
            #     print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
            #     o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
            end_time = time.time()
            # Calculate training time
            training_time = end_time - start_time

            # Log the training time
            o_f.write("Training time:"+str(training_time)+"seconds\n")
    # single run
    else:
        run_experiment(args)
    sys.exit()



