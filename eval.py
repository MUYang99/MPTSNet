from os.path import dirname
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from data_provider import TSC_multivariate_data_loader, fill_out_with_Nan
from utils import get_confmat_metrics, fft_main_periods_wo_duplicates
from model.MPTSNet import Model


dataset_path = dirname("./dataset/General/")
dataset_name_list = [
"JapaneseVowels",
"SelfRegulationSCP1",
"Handwriting",
"UWaveGestureLibrary",
]

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")


Result_log_folder = "./results/0721/MPTSNet/"
for dataset_name in dataset_name_list:
    print('evaluating at:', dataset_name)
    # load multivariate data
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
    print('[INFO] running at:', dataset_name)
    # load multivariate data
    print('test data shape', X_test.shape)
    print('test label shape', y_test.shape)
    print('unique test label', np.unique(y_test))

    if X_train.shape[-1] != X_test.shape[-1]:
        print('[INFO]: seq length between train and test unmatched')
        target_length = max(X_train.shape[-1], X_test.shape[-1])
        if X_train.shape[-1] > X_test.shape[-1]:
            X_test = fill_out_with_Nan(X_test, target_length)
        else:
            X_train = fill_out_with_Nan(X_train, target_length)

    print('train data shape', X_train.shape)
    print('test data shape', X_test.shape)

    num_channels = X_train.shape[1]
    seq_length = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    embed_dim = max(min(num_channels * 4, 256), 64)
    print(f"Adaptive embed_dim: {embed_dim}")
    embed_dim_t = max(min(embed_dim * 4, 512), 256)
    print(f"Adaptive embed_dim_t: {embed_dim_t}")

    X_test = torch.from_numpy(X_test).float()
    X_test[torch.isnan(X_test)] = 0
    X_test.requires_grad = False
    if len(X_test.shape) == 3:
        X_test = X_test.to(device)
    else:
        X_test = X_test.unsqueeze_(1).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    X_train_fft = X_train.permute(0, 2, 1).detach().cpu().numpy()
    periods = fft_main_periods_wo_duplicates(X_train_fft, 5, dataset_name)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, 32)), 2),
                             shuffle=False)

    n_input_channel = X_test.shape[1]
    n_class = max(y_test) + 1

    flag_DE_1 = False
    if dataset_name in ['PEMS-SF']:
        flag_DE_1 = True

    # load saved model
    print('[INFO] Loading Model...')
    # receptive_field_shape = min(int(X_train.shape[-1] / quarter_or_half), Max_kernel_size)
    # layer_parameter_list = generate_layer_parameter_list(start_kernel_size, receptive_field_shape,
    #                                                      paramenter_number_of_layer_list, in_channel=1)
    model_save_path = Result_log_folder + dataset_name + '/' + 'best_model'
    # model = OS_CNN(layer_parameter_list, n_class.item(), n_input_channel, True).to(device)
    model = Model(periods=periods, flag=flag_DE_1, num_channels=num_channels, seq_length=seq_length, num_classes=num_classes, embed_dim=embed_dim,
                               embed_dim_t=embed_dim_t, num_heads=4, ff_dim=256, num_layers=1).to(device)
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)

    # new_state_dict = model.state_dict()
    # checkpoint = torch.load(model_save_path, map_location=device)
    # for name, param in checkpoint.named_parameters():
    #     if name in new_state_dict:
    #         new_state_dict[name] = param
    # model.load_state_dict(new_state_dict)

    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        confusion_matrix = torch.zeros(n_class, n_class)
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            target = target.to(device)

            output = model(data.float())
            pred = output.data.max(1, keepdim=False)[1]
            count += target.shape[0]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print("Final result: ")
        print('\nTest set:  Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, count,
            100. * correct / count))

        confusion_matrix = confusion_matrix.numpy()
        precision, recall, f1 = get_confmat_metrics(confusion_matrix)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)




