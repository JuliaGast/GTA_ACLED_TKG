import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #added julia - set gpu id
import hydra 
from omegaconf import DictConfig

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseDataset, QuadruplesDataset
from model.agent import Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
from model.dirichlet import Dirichlet
import os
import pickle


def get_model_config(args, num_ent, num_rel):
    config = {
        'cuda': args.cuda,  # whether to use GPU or not.
        'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations
        'time_dim': args.time_dim,  # Embedding dimension of the timestamps
        'state_dim': args.state_dim,  # dimension of the LSTM hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman Eq.
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used
    }
    return config

@hydra.main(version_base=None, config_path='', config_name="conf2")
def main(config:DictConfig):

    args = config
    
    #######################Set Logger#################################
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    data_path = os.path.join('data', args.data_path) #modified julia

    #######################Create DataLoader#################################
    train_path = os.path.join(data_path, 'train.txt') #modified julia -> args.data_path to data_path
    test_path = os.path.join(data_path, 'test.txt') #modified julia -> args.data_path to data_path
    stat_path = os.path.join(data_path, 'stat.txt') #modified julia -> args.data_path to data_path
    valid_path = os.path.join(data_path, 'valid.txt') #modified julia -> args.data_path to data_path

    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset  = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)
    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ######################Creat the agent and the environment###########################
    config = get_model_config(args, baseData.num_e, baseData.num_r)
    logging.info(config)
    logging.info(args)

    # creat the agent
    agent = Agent(config)

    # creat the environment
    state_actions_path = os.path.join(data_path, args.state_actions_path) #modified julia -> args.data_path to data_path
    if not os.path.exists(state_actions_path):
        state_action_space = None
        print(state_actions_path, ' load did not work')
    else:
        state_action_space = pickle.load(open(os.path.join(data_path, args.state_actions_path), 'rb')) #modified julia -> args.data_path to data_path
    env = Env(baseData.allQuadruples, config, state_action_space)

    # Create episode controller
    episode = Episode(env, agent, config)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    model_path_name = args.data_path + '_time_singlestep_checkpoint_270.pth'
    # load_model_path = os.path.join(args.save_path, model_path_name)
    load_model_path = os.path.join(args.load_model_path, model_path_name)

    print(load_model_path)
    if os.path.isfile(load_model_path): #modified julia: load_model_path set manually instead of args
        print("loaded pretraine model from ", load_model_path)
        params = torch.load(load_model_path) #modified julia: load_model_path set manually instead of args
        episode.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(load_model_path))
    else:
        print("loading model did NOT work ", load_model_path)

    ######################Training and Testing###########################
    if args.reward_shaping:
        alphas = pickle.load(open(os.path.join(data_path, args.alphas_pkl), 'rb')) #modified julia -> args.data_path to data_path
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None
    trainer = Trainer(episode, pg, optimizer, args, distributions)
    tester = Tester(episode, args, baseData.train_entities, baseData.RelEntCooccurrence)
    best_valid_mrr = 0
    best_epoch = 0
    if args.do_train:
        
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            if i % args.save_epoch == 0 and i != 0:
                # trainer.save_model('checkpoint_{}.pth'.format(i)) #
                model_name = str(args.data_path) +  '_' + args.setting + '_' + args.singleormultistep + '_' #added julia
                trainer.save_model(model_name, 'checkpoint_{}.pth'.format(i)) #modified julia
                logging.info('Save Model in {}'.format(args.save_path))
            # a=1
            if i % args.valid_epoch == 0 and i != 0:                
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                if metrics['MRR'] > best_valid_mrr:
                    best_valid_mrr = metrics['MRR']
                    best_epoch = i
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))
        model_name = str(args.data_path) +  '_' + args.setting + '_' + args.singleormultistep + '_' #added julia
        trainer.save_model(model_name) #modified julia: model_name
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        logging.info('Start Testing......')
        metrics = tester.test(test_dataloader,
                              testDataset.__len__(),
                              baseData.skip_dict,
                              config['num_ent'],
                              log_scores_flag=True,                         # added julia
                              dataset_dir=data_path,                        # added julia
                              dataset_name=args.data_path,                  # added julia
                              setting=args.setting,                         # added julia
                              singleormultistep=args.singleormultistep,
                              save_path=args.save_path)     # added julia
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))
    with open('valid_mrrs.txt', 'a') as file:
    # Append a new line to the file
        stringlist = [str(c) for c in config.items()]

        stringstring = ''
        for c in args.items():
            stringstring += str(c)
        print(stringstring)
        file.write(str(best_valid_mrr) +'\t'+str(best_epoch)+'\t'+ stringstring +'\n')
    return best_valid_mrr

if __name__ == '__main__':
    # args = parse_args()
    main()
