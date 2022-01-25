import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()

    data = [line.strip().split(',') for line in data]
    data = np.array(data).astype(np.float)
    return data


def plot_data(model_type):
    acc = get_data(f"./plot_data/{model_type}.csv")
    x = np.linspace(2, 10, 9)

    sns.set()
    fig = plt.figure(figsize=(10, 5))
    ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)

    ax1.plot(x, acc[0], 'r', label='GPT')
    ax1.plot(x, acc[1], 'b', label='RNN')
    ax1.plot(x, acc[2], 'g', label='LSTM')
    ax1.plot(x, acc[3], 'y', label='GRU')
    ax1.legend(loc="lower left")
    ax1.set_title("20 epochs")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("% accuracy")

    ax2.plot(x, acc[4], 'r', label='GPT')
    ax2.plot(x, acc[5], 'b', label='RNN')
    ax2.plot(x, acc[6], 'g', label='LSTM')
    ax2.plot(x, acc[7], 'y', label='GRU')
    ax2.legend(loc="lower left")
    ax2.set_title("50 epochs")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("% accuracy")

    plt.show()
