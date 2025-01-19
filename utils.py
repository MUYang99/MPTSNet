import matplotlib.pyplot as plt
from data_provider import TSC_multivariate_data_loader, fill_out_with_Nan
import json
from bunch import Bunch
import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import pywt

# device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
# print(f"Running on {device}")


# Function to plot time series data
def plot_time_series(data, num_series=3, series_length=26):
    fig, axes = plt.subplots(num_series, 1, figsize=(15, num_series * 5))
    for i in range(min(num_series, data.shape[0])):
        for j in range(data.shape[1]):
            axes[i].plot(data[i, j, :series_length], label=f'Channel {j+1}')
        axes[i].legend()
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
    plt.tight_layout()
    plt.show()


def get_config_from_json(json_file):
    """
    :param json_file
    :return: config class
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config


def get_confmat_metrics(confusion_matrix):
    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)  # TP/P
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)  # TP/T
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def save_to_log(sentence, Result_log_folder, dataset_name):
    father_path = Result_log_folder + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + '.txt'
    # print(path)
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')


def eval_model(model, dataloader):
    predict_list = np.array([])
    label_list = np.array([])
    for sample in dataloader:
        y_predict = model(sample[0])
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, sample[1].detach().cpu().numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    return acc


def eval_condition(iepoch, print_result_every_x_epoch):
    if (iepoch + 1) % print_result_every_x_epoch == 0:
        return True
    else:
        return False


def instance_norm(case):
    mean = case.mean(0, keepdim=True)
    case = case - mean
    stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
    case /= stdev
    return case


def layer_norm(case):
    mean = case.mean(dim=(1, 2), keepdim=True)
    std = case.std(dim=(1, 2), keepdim=True)
    return (case - mean) / (std + 1e-5)


def get_adaptive_embed_dim(num_channels, base_factor=4, min_dim=32, max_dim=256):
    embed_dim = num_channels * base_factor
    embed_dim = max(min_dim, embed_dim)
    embed_dim = min(max_dim, embed_dim)
    return embed_dim


def get_compatible_embed_dim(dim, num_channels, num_heads):
    # if dim % num_heads == 0:
    #     return dim
    # else:
    if dim*num_heads*num_channels < 96000:
        return dim*num_heads
    else:
        return (dim // num_heads + 1) * num_heads


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def fft(data):
    N = data.shape[1]  # sequence length
    T = 1.0 / N  # sampling interval

    # average batch_size and num_channels dim
    averaged_data = data.mean(axis=0).mean(axis=-1)

    # compute FFT
    yf = np.fft.fft(averaged_data)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    power_spectrum = 2.0 / N * np.abs(yf[:N // 2])
    # power_spectrum = np.abs(yf[:N//2])

    return xf, power_spectrum


def fft_main_periods(data, k, dataset_name):
    xf, power_spectrum = fft(data)
    N = data.shape[1]
    averaged_data = data.mean(axis=0).mean(axis=-1)

    # # filter zero frequency
    # nonzero_indices = xf > 0
    # xf = xf[nonzero_indices]
    # power_spectrum = power_spectrum[nonzero_indices]

    # filter zero frequency and frequency equals to 1
    valid_indices = (xf > 0) & (xf != 1)
    xf = xf[valid_indices]
    power_spectrum = power_spectrum[valid_indices]

    # top k amplitudes and frequencies
    indices = np.argsort(power_spectrum)[-k:][::-1]
    main_frequencies = xf[indices]
    main_amplitudes = power_spectrum[indices]
    main_periods = (1 / main_frequencies * N).astype(int)

    # Print main periods and amplitudes
    print("Main periods and amplitudes: ")
    for i, (period, freq, amp) in enumerate(zip(main_periods, main_frequencies, main_amplitudes)):
        print(f"period {i + 1}: {period}, amplitude: {int(amp)}")

    # Plot time series and power spectrum
    time = np.arange(N)
    plt.figure(figsize=(10, 14))

    plt.subplot(2, 1, 1)
    plt.plot(time, averaged_data)
    plt.title(dataset_name+' Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    plt.plot(xf, power_spectrum)
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.scatter(main_frequencies, main_amplitudes, color='red', zorder=5)  # mark main frequency points

    for i, (freq, amp) in enumerate(zip(main_frequencies, main_amplitudes)):
        plt.annotate(f'{int(freq)} Hz\n{int(amp)}', xy=(freq, amp), xytext=(freq, amp + 0.02),
                     textcoords='data', ha='center', color='red')
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()
    return main_periods


def fft_find_amplitude(data, target_period):
    N = data.shape[1]  # sequence length
    T = 1.0 / N  # sampling interval

    # average batch_size and num_channels dim
    averaged_data = data.mean(axis=0).mean(axis=-1)

    # compute FFT
    yf = np.fft.fft(averaged_data)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    power_spectrum = 2.0 / N * np.abs(yf[:N // 2])

    # calculate target frequency
    target_frequency = N / target_period
    # find closet index
    closest_index = np.argmin(np.abs(xf - target_frequency))
    closest_frequency = xf[closest_index]
    closest_amplitude = power_spectrum[closest_index]

    # print(f"target period: {int(target_period)} , closest_amplitude: {int(closest_amplitude)}")

    return closest_amplitude


def fft_main_periods_wo_duplicates(data, k, dataset_name):
    xf, power_spectrum = fft(data)
    N = data.shape[1]
    averaged_data = data.mean(axis=0).mean(axis=-1)

    # filter zero frequency and frequency equals to 1
    valid_indices = (xf > 0) & (xf != 1)
    xf = xf[valid_indices]
    power_spectrum = power_spectrum[valid_indices]

    # top amplitudes and frequencies
    indices = np.argsort(power_spectrum)[::-1]  # rank from high to low
    main_frequencies = xf[indices]
    main_amplitudes = power_spectrum[indices]

    unique_periods = []
    unique_amplitudes = []
    unique_frequencies = []
    used_periods = set()

    i = 0
    while len(unique_periods) < k and i < len(main_frequencies):
        period = np.round(1 / main_frequencies[i] * N).astype(int)
        if period not in used_periods:
            unique_periods.append(period)
            unique_amplitudes.append(main_amplitudes[i])
            unique_frequencies.append(main_frequencies[i])
            used_periods.add(period)
        i += 1

    # Print main periods and amplitudes
    print("Main periods and amplitudes: ")
    for i, (period, freq, amp) in enumerate(zip(unique_periods, unique_frequencies, unique_amplitudes)):
        print(f"period {i + 1}: {period}, amplitude: {int(amp)}")

    # # Plot time series and power spectrum
    # time = np.arange(N)
    # plt.figure(figsize=(14, 20))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(time, averaged_data)
    # plt.title(dataset_name + ' Time Series')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(xf, power_spectrum)
    # plt.title('Power Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')
    # plt.scatter(unique_frequencies, unique_amplitudes, color='red', zorder=5)  # mark main frequency points
    #
    # for i, (freq, amp) in enumerate(zip(unique_frequencies, unique_amplitudes)):
    #     plt.annotate(f'{int(freq)} Hz\n{int(amp)}', xy=(freq, amp), xytext=(freq, amp + 0.02),
    #                  textcoords='data', ha='center', color='red')
    # plt.subplots_adjust(hspace=0.4)
    # # plt.tight_layout()
    # plt.show()
    return unique_periods


def fft_find_each_amplitude(data, target_period):
    '''
    For each element in a batch
    :param data:
    :param target_period:
    :return: target amplitudes
    '''
    batch_size = data.shape[0]
    sequence_length = data.shape[1]
    T = 1.0 / sequence_length  # sampling interval

    # Initialize an array to store the amplitude for each batch element
    amplitudes = torch.zeros((batch_size, 1))

    for i in range(batch_size):
        # For each batch element, average the num_channels dimension
        averaged_data = data[i].mean(axis=-1)

        # Compute FFT
        yf = np.fft.fft(averaged_data)
        xf = np.fft.fftfreq(sequence_length, T)[:sequence_length // 2]
        power_spectrum = 2.0 / sequence_length * np.abs(yf[:sequence_length // 2])

        # Calculate the target frequency
        target_frequency = sequence_length / target_period
        # Find the closest frequency index
        closest_index = np.argmin(np.abs(xf - target_frequency))
        closest_amplitude = power_spectrum[closest_index]

        # Store the amplitude of the current batch element
        amplitudes[i] = closest_amplitude

    return amplitudes


if __name__ == '__main__':
    device = torch.device('cpu')
    dataset_path = './dataset/General/'
    dataset_name_list = [
        "EthanolConcentration",
        "FaceDetection",
        "Handwriting",
        "Heartbeat",
        "JapaneseVowels",
        "PEMS-SF",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "UWaveGestureLibrary",
    ]
    for dataset_name in dataset_name_list:
        X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
        print('[INFO] running at:', dataset_name)
        # load multivariate data
        print('train data shape', X_train.shape)

        if X_train.shape[-1] != X_test.shape[-1]:
            print('[INFO]: seq length between train and test unmatched')
            target_length = max(X_train.shape[-1], X_test.shape[-1])
            if X_train.shape[-1] > X_test.shape[-1]:
                X_test = fill_out_with_Nan(X_test, target_length)
            else:
                X_train = fill_out_with_Nan(X_train, target_length)

        print('train data shape', X_train.shape)

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

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

        X_train_fft = X_train.permute(0, 2, 1)
        fft_main_periods_wo_duplicates(X_train_fft, 5, dataset_name)
    #     # periods, amplitudes = FFT_for_Period(X_train_fft, k=5)
    #     # print("X_train FFT periods:", periods)
    #
    #     train_dataset = TensorDataset(X_train, y_train)
    #     train_loader = DataLoader(train_dataset, batch_size=5,
    #                               shuffle=True)
    #     test_dataset = TensorDataset(X_test, y_test)
    #     test_loader = DataLoader(test_dataset, batch_size=5,
    #                              shuffle=False)

        # i = 0
        # for sample in train_loader:
        #     i += 1
        #     x = sample[0]
        #     print(x.shape)
        #     x_fft = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        #     fft_main_periods(x_fft, 10)
        #     fft_find_amplitude(x_fft, 350)
        #     # w_periods, w_amplitudes = DWT_for_Period(x_fft, k=5)
        #     # print("DWT periods:", w_periods)
        #     if i == 3:
        #         break

    # import psutil
    #
    # # 查看CPU使用率
    # cpu_usage = psutil.cpu_percent(interval=1)
    # print(f"CPU使用率: {cpu_usage}%")
    #
    # # 查看内存使用情况
    # memory_info = psutil.virtual_memory()
    # print(f"总内存: {memory_info.total / (1024 ** 3):.2f} GB")
    # print(f"已使用内存: {memory_info.used / (1024 ** 3):.2f} GB")
    # print(f"可用内存: {memory_info.available / (1024 ** 3):.2f} GB")
    # print(f"内存使用率: {memory_info.percent}%")
    #
    # # 查看每个进程的内存和CPU使用情况
    # for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
    #     print(proc.info)

    # Plot a subset of the training data
    # plot_time_series(X_train, num_series=3, series_length=X_train.shape[2])

    # train_file_path = './dataset/General/JapaneseVowels/JapaneseVowels_TRAIN.ts'
    # # train_file_path = './dataset/NATOPS/NATOPS_TRAIN.ts'
    # test_file_path = './dataset/General/JapaneseVowels/JapaneseVowels_TEST.ts'
    # output_directory = './dataset/General/JapaneseVowels/output/'
    #
    # with open(train_file_path) as file:
    #     lines = file.readlines()
    #     i = 0
    #     for line in lines:
    #         print(line)
    #         i += 1
    #         if i == 100:
    #             break

    # # Generate sample time series
    # np.random.seed(0)
    # time = np.arange(0, 400)
    # trend = 0.01 * time
    # seasonal = 10 * np.sin(2 * np.pi * time / 50)
    # noise = np.random.normal(0, 2, time.shape)
    # time_series = trend + seasonal + noise
    #
    # # Compute FFT
    # fft_result = np.fft.fft(time_series)
    # freq = np.fft.fftfreq(time_series.size)
    #
    # # Consider only the positive frequencies
    # positive_freq_indices = np.where(freq > 0)
    # positive_freq = freq[positive_freq_indices]
    # positive_fft_result = fft_result[positive_freq_indices]
    #
    # # Extract the dominant period
    # dominant_frequency = positive_freq[np.argmax(np.abs(positive_fft_result))]
    # dominant_period = 1 / dominant_frequency
    #
    # # Extract the trend component
    # trend_component = np.fft.ifft(np.where(np.abs(freq) < 0.1, fft_result, 0)).real
    #
    # # Print the dominant period
    # print(f"Dominant period: {dominant_period:.2f} time units")
    #
    # # Visualize the results
    # plt.figure(figsize=(14, 7))
    #
    # # Original time series
    # plt.subplot(2, 2, 1)
    # plt.plot(time, time_series, label='Original time series')
    # plt.legend()
    #
    # # Frequency spectrum
    # plt.subplot(2, 2, 2)
    # plt.plot(positive_freq, np.abs(positive_fft_result), label='Frequency spectrum')
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    #
    # # Extracted trend
    # plt.subplot(2, 2, 3)
    # plt.plot(time, trend_component, label='Extracted trend', color='orange')
    # plt.legend()
    #
    # # Original time series and extracted trend
    # plt.subplot(2, 2, 4)
    # plt.plot(time, time_series, label='Original time series')
    # plt.plot(time, trend_component, label='Extracted trend', color='orange')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # Generate sample time series with noise
    # np.random.seed(0)
    # time = np.arange(0, 400)
    # trend = 0.01 * time
    # seasonal = 10 * np.sin(2 * np.pi * time / 50)
    # noise = np.random.normal(0, 2, time.shape)
    # time_series = trend + seasonal + noise
    #
    # # Compute FFT
    # fft_result = np.fft.fft(time_series)
    # freq = np.fft.fftfreq(time_series.size)
    #
    # # Only consider positive frequencies
    # positive_freq_indices = np.where(freq > 0)
    # positive_freq = freq[positive_freq_indices]
    # positive_fft_result = fft_result[positive_freq_indices]
    #
    # # Identify noise characteristics
    # noise_threshold = np.percentile(np.abs(positive_fft_result), 90)
    # noise_freq_indices = np.where(np.abs(positive_fft_result) > noise_threshold)
    #
    # # Visualize the results
    # plt.figure(figsize=(14, 7))
    #
    # # Original time series
    # plt.subplot(2, 2, 1)
    # plt.plot(time, time_series, label='Original time series')
    # plt.legend()
    #
    # # Frequency spectrum
    # plt.subplot(2, 2, 2)
    # plt.plot(positive_freq, np.abs(positive_fft_result), label='Frequency spectrum')
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    #
    # # Identified noise frequencies
    # plt.subplot(2, 2, 3)
    # plt.plot(positive_freq, np.abs(positive_fft_result), label='Frequency spectrum')
    # plt.scatter(positive_freq[noise_freq_indices], np.abs(positive_fft_result)[noise_freq_indices], color='red',
    #             label='Noise frequencies')
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # import statsmodels.api as sm
    # from statsmodels.tsa.stattools import adfuller, kpss
    # import pywt
    #
    # # 生成示例时间序列数据
    # np.random.seed(0)
    # time = np.arange(0, 400)
    # trend = 0.01 * time
    # seasonal = 10 * np.sin(2 * np.pi * time / 50)
    # noise = np.random.normal(0, 2, time.shape)
    # time_series = trend + seasonal + noise
    #
    # # 绘制时间序列图
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, time_series)
    # plt.title("Time Series")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.show()
    #
    # # ADF检验
    # adf_result = adfuller(time_series)
    # print("ADF Statistic:", adf_result[0])
    # print("p-value:", adf_result[1])
    #
    # # KPSS检验
    # kpss_result = kpss(time_series, regression='c')
    # print("KPSS Statistic:", kpss_result[0])
    # print("p-value:", kpss_result[1])
    #
    # # 自相关函数（ACF）和偏自相关函数（PACF）
    # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    # sm.graphics.tsa.plot_acf(time_series, lags=40, ax=ax[0])
    # sm.graphics.tsa.plot_pacf(time_series, lags=40, ax=ax[1])
    # plt.show()
    #
    # # 自适应选择FFT或小波变换
    # if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
    #     print("Signal is stationary. Using FFT.")
    #
    #     # 使用FFT计算主要周期
    #     freq_spectrum = np.fft.fft(time_series)
    #     freqs = np.fft.fftfreq(len(time_series))
    #     positive_freqs = freqs[np.where(freqs > 0)]
    #     positive_spectrum = np.abs(freq_spectrum[np.where(freqs > 0)])
    #
    #     dominant_freq_index = np.argmax(positive_spectrum)
    #     dominant_freq = positive_freqs[dominant_freq_index]
    #     dominant_period_fft = 1 / dominant_freq
    #
    #     print("Dominant Period using FFT:", dominant_period_fft)
    #
    # else:
    #     print("Signal is non-stationary. Using Wavelet Transform.")
    #
    #     # 使用Mexican Hat小波进行CWT计算主要周期
    #     widths = np.arange(1, 128)
    #     cwt_matrix, freqs = pywt.cwt(time_series, widths, 'mexh')
    #
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(cwt_matrix, extent=[0, 400, 1, 128], cmap='PRGn', aspect='auto',
    #                vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    #     plt.colorbar(label='Coefficient Value')
    #     plt.ylabel('Scale (width)')
    #     plt.xlabel('Time')
    #     plt.title('Continuous Wavelet Transform (Mexican Hat)')
    #     plt.show()
    #
    #     dominant_scale = widths[np.argmax(np.sum(np.abs(cwt_matrix), axis=1))]
    #     dominant_period_mexican_hat = dominant_scale
    #
    #     print("Dominant Period using Mexican Hat Wavelet:", dominant_period_mexican_hat)
