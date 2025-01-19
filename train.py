import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import gc
import pandas as pd
from model.MPTSNet import Model
from data_provider import TSC_multivariate_data_loader, fill_out_with_Nan
from utils import eval_condition, eval_model, save_to_log, fft_main_periods_wo_duplicates


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")


if __name__ == '__main__':
    print('[INFO] Loading data...')
    dataset_path = './dataset/General'
    dataset_name_list = [
        # "Heartbeat",
        "EthanolConcentration",
        # "Handwriting",
        # "JapaneseVowels",
        # "PEMS-SF",
        # "SelfRegulationSCP1",
        # "SelfRegulationSCP2",
        # "UWaveGestureLibrary",
        # "SpokenArabicDigits",
        # "FaceDetection",
    ]
    acc = []
    Result_log_folder = "./results/0803/MPTSNet/"
    for dataset_name in dataset_name_list:
        X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
        print('[INFO] running at:', dataset_name)
        # load multivariate data
        print('train data shape', X_train.shape)
        print('train label shape', y_train.shape)
        print('test data shape', X_test.shape)
        print('test label shape', y_test.shape)
        print('unique train label', np.unique(y_train))
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
        embed_dim = max(min(num_channels * 4, 256), 64)
        print(f"Adaptive embed_dim: {embed_dim}")
        embed_dim_t = max(min(embed_dim * 4, 512), 256)
        print(f"Adaptive embed_dim_t: {embed_dim_t}")

        seq_length = X_train.shape[2]
        num_classes = len(np.unique(y_train))

        # covert numpy to pytorch tensor
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

        if dataset_name in ['Heartbeat', 'SelfRegulationSCP1', 'SelfRegulationSCP2']:
            print('[INFO] Z-norm for stable convergence: ')
            mean = X_train.mean(dim=(0, 2), keepdim=True)
            std = X_train.std(dim=(0, 2), keepdim=True)

            X_train = (X_train - mean) / (std + 1e-5)
            X_test = (X_test - mean) / (std + 1e-5)

        flag_DE_1 = False
        if dataset_name in ['PEMS-SF']:
            flag_DE_1 = True

        # replace NaN with 0
        X_train[torch.isnan(X_train)] = 0
        X_test[torch.isnan(X_test)] = 0

        # covert numpy to pytorch tensor and put into gpu
        X_train.requires_grad = False
        if len(X_train.shape) == 3:
            X_train = X_train.to(device)
        else:
            X_train = X_train.unsqueeze_(1).to(device)
        y_train = torch.LongTensor(y_train).to(device)

        X_test.requires_grad = False
        if len(X_test.shape) == 3:
            X_test = X_test.to(device)
        else:
            X_test = X_test.unsqueeze_(1).to(device)
        y_test = torch.LongTensor(y_test).to(device)

        X_train_fft = X_train.permute(0, 2, 1).detach().cpu().numpy()
        periods = fft_main_periods_wo_duplicates(X_train_fft, 5, dataset_name)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        print('[INFO] Training Model...')
        if not os.path.exists(Result_log_folder + dataset_name + '/'):
            os.makedirs(Result_log_folder + dataset_name + '/')
        model_save_path = Result_log_folder + dataset_name + '/' + 'best_model'

        model = Model(periods=periods, flag=flag_DE_1, num_channels=num_channels, seq_length=seq_length, num_classes=num_classes, embed_dim=embed_dim,
                               embed_dim_t=embed_dim_t, num_heads=4, ff_dim=256, num_layers=1).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total parameters of model: {total_params}")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10,
                                                               min_lr=0.0001)
        best_test_acc = 0
        lr_history = []

        model.train()
        patience = 20
        cnt = 0
        for i in range(200):
            running_loss = 0.0
            batch_count = 0
            for sample in train_loader:
                optimizer.zero_grad()
                y_predict = model(sample[0])
                loss = criterion(y_predict, sample[1])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / batch_count
            scheduler.step(avg_loss)

            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            if len(lr_history) > 1 and lr_history[-1] != lr_history[-2]:
                print(f"Epoch {i + 1}: Learning rate changed to {current_lr}")

            if eval_condition(i, 1):
                for param_group in optimizer.param_groups:
                    print('epoch =', i, 'lr = ', param_group['lr'])
                model.eval()
                acc_train = eval_model(model, train_loader)
                acc_test = eval_model(model, test_loader)
                model.train()
                print('train_acc=\t', acc_train, '\t test_acc=\t', acc_test, '\t loss=\t', avg_loss)
                temp_result = 'train_acc=\t' + str(acc_train) + '\t test_acc=\t' + str(acc_test)
                save_to_log(temp_result, Result_log_folder, dataset_name)

                if acc_test > best_test_acc:
                    best_test_acc = acc_test
                    torch.save(model.state_dict(), model_save_path)
                    print(f'New best model saved with test_acc={acc_test}')
                    cnt = 0
                else:
                    cnt += 1
                    if cnt >= patience:
                        print('Early stopping after', i + 1, 'epochs.')
                        break
        print('[INFO] Best test accuracy on ', dataset_name, ": ", best_test_acc)
        final_result = 'Best test accuracy on ' + str(dataset_name) + ": " + str(best_test_acc)
        save_to_log(final_result, Result_log_folder, dataset_name)
        acc.append(round(best_test_acc, 3))

        # release memory
        del X_train, y_train, X_test, y_test
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        gc.collect()

    acc.append(round(np.average(acc), 3))
    columns = dataset_name_list + ["Average acc"]
    df = pd.DataFrame([acc], columns=columns)
    df.to_csv(Result_log_folder + 'accuracy.csv', sep='\t', index=False)
    print(acc)