'''
       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
'''

import numpy as np
import seasonal
from networkx.algorithms import approximation as app
import networkx as nx
from matplotlib import pyplot as plt

class TsAnalyser():
    """ For time series extraction """
    def __init__(self, device='cpu'):
        self.device = device
        self.seasonal_markers_dict = {}

    def preprocess_dataset(self, timesteps_all, graph_dict, dataset_name):
        graph_ts = list(graph_dict.keys())

        self._timestep_indexer = {value: idx for idx, value in enumerate(timesteps_all)}  # maps ts2indeces
        self._timestep_indexer_inv = {idx: value for idx, value in enumerate(timesteps_all)}  # maps indeces2ts

        all_features = self.extract_timeseries_from_graphs(graph_dict) 
        # extract: [num_triples_all, num_nodes, max_deg, mean_deg, mean_deg_c, max_deg_c, min_deg_c, density]

        self._min_list_from_train = [np.min(all_features[i]) for i in range(len(all_features))]
        self._max_list_from_train = [np.max(all_features[i]) for i in range(len(all_features))]

        seasonality =self.estimate_seasons(all_features[0])     # estimate seasonality 

        # create seasonal markers dict (for all timesteps)
        seasonal_markers_dict = self.assign_seasonal_markers(timesteps_all, seasonality)         
        self.seasonal_markers_dict = seasonal_markers_dict

        names = ['Number of Triples', 'Number of Nodes', 'Max Node Degree', 'Mean Node Degree',         
                 'Mean Degree Centrality', 'Max Degree Centrality', 'Min Degree Centrality', 'Density']

        for timeseries, name in zip(all_features, names):
            self.plot_feature_figure(timesteps_all, timeseries, name, seasonality, dataset_name)

        feature_extension_dict = self.extend_features(seasonal_markers_dict, all_features, graph_ts)        

        return seasonality, seasonal_markers_dict, feature_extension_dict

        

    def extend_features(self, seasonal_markers_dict, all_features, timesteps_of_interest):
        """ dict with key: timestep, values: all the extended features.
        """

        extended_features = {}
        for ts in timesteps_of_interest:
            index = self._timestep_indexer[ts]
            features_ts = [all_features[feat][index] for feat in range(len(all_features))]
            features_ts.append(seasonal_markers_dict[ts])
            extended_features[ts] = features_ts
        
        return extended_features

    def extract_timeseries_from_graphs(self, graph_dict):
        """ extracts multivariate timeseries from quadruples based on graph params

        :param graph_dict: dict, with keys: timestep, values: triples; training quadruples.

        """
        num_nodes = []
        num_triples = []
        max_deg = []
        mean_deg = []
        mean_deg_c = [] 
        max_deg_c = [] 
        min_deg_c = [] 
        density = []

        for ts, triples_snap in graph_dict.items():

            # create graph for that timestep
            e_list_ts = [(triples_snap[line][0], triples_snap[line][2]) for line in range(len(triples_snap))]
            G = nx.MultiGraph()
            G.add_nodes_from(graph_dict[ts][:][ 0])
            G.add_nodes_from(graph_dict[ts][:][2])
            G.add_edges_from(e_list_ts)  # default edge data=1

            # extract relevant parameters and append to list
            num_nodes.append(G.number_of_nodes())
            num_triples.append(G.number_of_edges())

            # degree
            deg_list = list(dict(G.degree(G.nodes)).values())
            max_deg.append(np.max(deg_list))
            mean_deg.append(np.mean(deg_list))
            
            # degree centrality
            deg_clist = list(dict(nx.degree_centrality(G)).values())
            mean_deg_c.append(np.mean(deg_clist))
            max_deg_c.append(np.max(deg_clist))
            min_deg_c.append(np.min(deg_clist))
            
            density.append(nx.density(G))
 
        return [num_triples, num_nodes, max_deg, mean_deg, mean_deg_c, max_deg_c, min_deg_c, density]

    def estimate_seasons(self, train_data):
        """ Estimate seasonal effects in a series.
               
        Estimate the major period of the data by testing seasonal differences for various period lengths and returning 
        the seasonal offsets that best predict out-of-sample variation.   
            
        First, a range of likely periods is estimated via periodogram averaging. Next, a time-domain period 
        estimator chooses the best integer period based on cross-validated residual errors. It also tests
        the strength of the seasonal effect using the R^2 of the leave-one-out cross-validation.

        :param data: list, data to be analysed, time-series;
        :return: NBseason int. if no season found: 1; else: seasonality that was discovered (e.g. if seven and 
                time granularity is daily: weekly seasonality)
        """
        seasons, trended = seasonal.fit_seasons(train_data)
        
        if seasons is None:
            Nbseason = int(1)
        else: 
            Nbseason = len(seasons)
            
        return Nbseason

    def extract_num_triples(self, triple_dict):
        num_triples_all = []
        for graph in triple_dict.values():
            num_triples_all.append(len(graph))
        return num_triples_all

    def assign_seasonal_markers(self, timesteps_all, seasonality):
        """ lookup dict that says which element of a seasonal period we are in for each timestep.
        e.g. timestep[0]:0 ( monday);  timestep[1]:1 ( tuesday); ...  timestep[7]:0 ( monday)
        :param timesteps_all: list with timesteps
        :param seasonality: int, seasonality that we have (e.g. if seven and time granularity 
                is daily: weekly seasonality)
        :return: seasonal_markers_dict; dict with keys: timesteps, values: season_index (0:seasonality)
        """

        seasonal_markers_dict = {}
        seasons = range(0,seasonality)
        seasons = seasons/np.max(seasons)
        season_idx = 0
        for ts in timesteps_all:
            seasonal_markers_dict[ts] = seasons[season_idx]
            season_idx+=1
            if season_idx==seasonality: # start from beginning
                season_idx = 0


        return seasonal_markers_dict

    def plot_feature_figure(self, timesteps, timeseries, name, seasonality, dataset_name):
        """ plot timeseries with graph params
            one figure per feature (ie one call of this function per feature)
        """
        plt.figure(figsize=(int(35)/5,int(18/5)))
        plt.plot(timesteps, timeseries, marker='.', markersize=2)
        
        tslist = []
        timeserieslist = []
        for i in range(2, len(timesteps), 7):
            tslist.append(timesteps[i])
            timeserieslist.append(timeseries[i])
        plt.scatter(tslist, timeserieslist, s=15.0, label = 'Sundays', color = 'grey', alpha= 0.4)
        min_val = np.min(timeseries) - 0.05*np.median(timeseries)
        max_val = np.max(timeseries) + 0.05*np.median(timeseries)
        plt.vlines(tslist, min_val, max_val, colors='grey', linestyles='solid', label='', alpha = 0.4)
        plt.ylabel(name)
        title_dict = {'Number of Triples': 'a) ', 'Number of Nodes': 'b) ', 'Max Node Degree': 'e) ', 
                      'Mean Node Degree': 'd) ', 'Mean Degree Centrality': None, 'Max Degree Centrality': None, 
                      'Min Degree Centrality': None, 'Density': 'c) '}
        new_name = name
        if name in title_dict.keys():
            if title_dict[name] != None:
                new_name = title_dict[name] +name
            
        plt.title(new_name + ' over Time')

        
        plt.legend()
        
        months = ['Jan', 'Mar',  'May',  'Jul',  'Sep',  'Nov',  'Jan']
        plt.xticks(np.linspace(0,365,7) , months)

        plt.savefig('./figs/' +dataset_name.split('/')[0]+name+'.pdf')
        print("Stored the analysis figures in ", './figs/' +dataset_name+name+'.pdf')

