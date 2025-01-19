import numpy as np
import torch


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data, max_length):
    pad_length = max_length - data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.full(pad_shape, np.nan)
        return np.concatenate((data, Nan_pad), axis=-1)


def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n', '').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i
                break
    return label_dict


def get_data_and_label_from_ts_file(file_path, label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data' in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n', '')])
                data_tuple = [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1] > max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data, max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length > max_length:
                    max_length = max_channel_length

        Data_list = [fill_out_with_Nan(data, max_length) for data in Data_list]
        X = np.concatenate(Data_list, axis=0)
        Y = np.asarray(Label_list)

        return np.float32(X), Y


def TSC_multivariate_data_loader(dataset_path, dataset_name):
    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path, label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path, label_dict)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    dataset_path = './dataset/General/'
    dataset_name = 'JapaneseVowels'
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_test.dtype)
