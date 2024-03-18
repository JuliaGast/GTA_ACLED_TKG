import torch
import tqdm
import numpy as np

class Tester(object):
    def __init__(self, model, args, train_entities, RelEntCooccurrence):
        self.model = model
        self.args = args
        self.train_entities = train_entities
        self.RelEntCooccurrence = RelEntCooccurrence

    def get_rank(self, score, answer, entities_space, num_ent):
        """Get the location of the answer, if the answer is not in the array,
        the ranking will be the total number of entities.
        Args:
            score: list, entity score
            answer: int, the ground truth entity
            entities_space: corresponding entity with the score
            num_ent: the total number of entities
        Return: the rank of the ground truth.
        """
        if answer not in entities_space:
            rank = num_ent
        else:
            answer_prob = score[entities_space.index(answer)]
            score.sort(reverse=True)
            rank = score.index(answer_prob) + 1
        return rank

    def test(self, dataloader, ntriple, skip_dict, num_ent, log_scores_flag=False, dataset_dir=None, dataset_name = None, setting='time', singleormultistep='singlestep', save_path=''):
        """Get time-aware filtered metrics(MRR, Hits@1/3/10).
        Args:
            ntriple: number of the test examples.
            skip_dict: time-aware filter. Get from baseDataset
            num_ent: number of the entities.
        Return: a dict (key -> MRR/HITS@1/HITS@3/HITS@10, values -> float)
        """
        # log_scores_flag = True #added julia
        # dataset_name = "ICEWS14"
        julia_logging_dict = {} #added julia
        if log_scores_flag: #added julia
            import inspect
            import sys
            import os
            currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            # parentdir = os.path.dirname(currentdir)
            sys.path.insert(1, currentdir) 
            sys.path.insert(1, os.path.join(sys.path[0], '../../..'))        
            sys.path.insert(1, os.path.join(sys.path[0], '..'))    
            sys.path.insert(1, os.path.join(sys.path[0], '../..'))
            import evaluation_utils 
            # dataset_dir = os.path.join('tKGR', 'data', dataset_name) #modified julia
            # dataset_dir = os.path.join('TITer-master', 'data', dataset_name) #modified julia
            num_nodes, num_rels = evaluation_utils.get_total_number(dataset_dir, 'stat.txt')
            #end added julia
        self.model.eval()
        logs = []
        with torch.no_grad():
            with tqdm.tqdm(total=ntriple, unit='ex') as bar:
                current_time = 0
                cache_IM = {}  # key -> entity, values: list, IM representations of the co-o relations.
                for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                    if rel_batch > 20:
                        a =1
                    mask = (rel_batch[:] == 4)  | (rel_batch[:]== 8)  | (rel_batch[:] == 24) | (rel_batch[:]== 28)
                    src_batch = src_batch[mask]
                    rel_batch = rel_batch[mask]
                    dst_batch = dst_batch[mask]
                    time_batch = time_batch[mask]
                    if len(src_batch) > 0:
                        if rel_batch > 10:
                            a=1
                        batch_size = dst_batch.size(0)

                        if self.args.IM:  # inductive 
                            src = src_batch[0].item()
                            rel = rel_batch[0].item()
                            dst = dst_batch[0].item()
                            time = time_batch[0].item()

                            # representation update
                            if current_time != time:
                                current_time = time
                                for k, v in cache_IM.items():
                                    ims = torch.stack(v, dim=0)
                                    self.model.agent.update_entity_embedding(k, ims, self.args.mu) #12 inductive
                                cache_IM = {}

                            if src not in self.train_entities and rel in self.RelEntCooccurrence['subject'].keys():  #inductive; self.RelEntCoocurrence is from train set only
                                im = self.model.agent.get_im_embedding(list(self.RelEntCooccurrence['subject'][rel])) # co ocurrence relations- list that contains the trained entities with the co-occurrence relation.
                                if src in cache_IM.keys():
                                    cache_IM[src].append(im)
                                else:
                                    cache_IM[src] = [im]

                                # prediction shift
                                self.model.agent.entities_embedding_shift(src, im, self.args.mu) # 13

                        if self.args.cuda:
                            src_batch = src_batch.cuda()
                            rel_batch = rel_batch.cuda()
                            dst_batch = dst_batch.cuda()
                            time_batch = time_batch.cuda()

                        current_entities, beam_prob = \
                            self.model.beam_search(src_batch, time_batch, rel_batch)                    

                        if self.args.IM and src not in self.train_entities:
                            # We do this
                            # because events that happen at the same time in the future cannot see each other.
                            self.model.agent.back_entities_embedding(src)                                    

                        if self.args.cuda:
                            current_entities = current_entities.cpu()
                            beam_prob = beam_prob.cpu()

                        current_entities = current_entities.numpy()
                        beam_prob = beam_prob.numpy()

                        MRR = 0
                        for i in range(batch_size):
                            if log_scores_flag:
                                scores_julia = -10000000000.0*np.ones(num_nodes, dtype=np.float32) #added julia -  added one more node for logging they return nr_nodes+1 score
                            candidate_answers = current_entities[i]
                            candidate_score = beam_prob[i]


                            # sort by score from largest to smallest
                            idx = np.argsort(-candidate_score)
                            candidate_answers = candidate_answers[idx]
                            candidate_score = candidate_score[idx]

                            # remove duplicate entities
                            candidate_answers, idx = np.unique(candidate_answers, return_index=True) #julia: ialso candidate answers mit id = 7128 (where 7128 num_nodes) predicted
                            candidate_answers = list(candidate_answers)
                            candidate_score = list(candidate_score[idx])

                            src = src_batch[i].item()
                            rel = rel_batch[i].item()
                            dst = dst_batch[i].item()
                            time = time_batch[i].item()

                            if log_scores_flag: #added julia for logging
                                if np.max(candidate_answers) >= num_nodes:
                                    if candidate_answers[-1] == num_nodes:
                                        logging_score_answers = candidate_answers[0:-1]
                                        logging_score = candidate_score[0:-1]
                                    else:
                                        print("Problem with the score ids", np.max(candidate_answers))
                                else:
                                    logging_score_answers = candidate_answers
                                    logging_score = candidate_score

                                scores_julia[logging_score_answers] = logging_score #added julia for logging
                                test_query = [src, rel, dst, time] #added julia
                                query_name, gt_test_query_ids = evaluation_utils.query_name_from_quadruple(test_query, num_rels, plus_one_flag=True) #added julia for logging
                                if query_name.startswith('xxx'):
                                    a = 1
                                julia_logging_dict[query_name] = [scores_julia, gt_test_query_ids]#added julia for logging
                                # end added julia

                            # get inductive inference performance.
                            # Only count the results of the example containing new entities.
                            if self.args.test_inductive and src in self.train_entities and dst in self.train_entities:
                                continue

                            filter = skip_dict[(src, rel, time)]  # a set of ground truth entities
                            tmp_entities = candidate_answers.copy()
                            tmp_prob = candidate_score.copy()
                            # time-aware filter
                            for j in range(len(tmp_entities)):
                                if tmp_entities[j] in filter and tmp_entities[j] != dst:
                                    candidate_answers.remove(tmp_entities[j])
                                    candidate_score.remove(tmp_prob[j])

                            ranking_raw = self.get_rank(candidate_score, dst, candidate_answers, num_ent)

                            logs.append({
                                'MRR': 1.0 / ranking_raw,
                                'HITS@1': 1.0 if ranking_raw <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking_raw <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking_raw <= 10 else 0.0,
                            })
                            MRR = MRR + 1.0 / ranking_raw #TODO check
                    else:
                        batch_size = 0.00001
                        MRR = 0
                    bar.update(batch_size)
                    bar.set_postfix(MRR='{}'.format(MRR / batch_size))
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)


            # #Julia
        if log_scores_flag:
            import pathlib
            import pickle
            dirname = os.path.join(pathlib.Path().resolve(), 'results' )

            logname = 'titer' + '-' + dataset_name + '-' + 'roi' +singleormultistep + '-' + setting +save_path
            juliafilename = os.path.join(dirname, logname + ".pkl")

            with open(juliafilename,'wb') as file:
                pickle.dump(julia_logging_dict, file, protocol=4) 
            file.close()

        return metrics
