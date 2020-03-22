import matplotlib.pyplot as plt


def plot_data(data, title, label=None):
    plt.plot(data, label=label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_losses(train_hist, test_hist):
    plt.plot(train_hist, label='Training Loss')
    plt.plot(test_hist, label='Test Loss')
    plt.ylim((0, 5))
    plt.legend()
    plt.show()


def plot_predicted_data(data, train_data, true_cases, predicted_cases, scaler):
    plt.plot(data.index[:len(train_data)],
             scaler.inverse_transform(train_data).flatten(),
             label='Historical Daily Cases')

    plt.plot(data.index[len(train_data):len(train_data) + len(true_cases)],
             true_cases,
             label='Real Daily Cases')

    plt.plot(data.index[len(train_data):len(train_data) + len(true_cases)],
             predicted_cases,
             label='Predicted Daily Cases')
    plt.legend()
    plt.show()


def plot_real_predicted(data, predicted_data):
    plt.plot(data, label='Historical Daily Cases')
    plt.plot(predicted_data, label='Predicted Daily Cases')
    plt.legend()
    plt.show()
