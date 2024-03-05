
import data_handler as data_handler
from ts_analysis import TsAnalyser


def main():
    # dataset
    datasetid = 'crisis2023/all.txt' #name of the dataset
    dataset = (datasetid, 3) # identifier, timestamp_column_idx

    # load dataset & group by timestamp
    data = data_handler.load(dataset[0])

    all_dict= [data_handler.group_by(entry, dataset[1]) for entry in [data]][0]
    timesteps_all = list(all_dict.keys())

    preprocessor = TsAnalyser()
    tmp = preprocessor.preprocess_dataset(timesteps_all, all_dict, datasetid)
    seasonality, seasonal_markers_dict, feature_extension_dict  = tmp #for downstream analysis if desired
    feature_extension_size = len(feature_extension_dict[0])

    print("We found a seasonality of ", seasonality, " and have extracted ", feature_extension_size, " graph timeseries features ")

if __name__ == "__main__":
    main()

