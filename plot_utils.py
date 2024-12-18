import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accs, val_accs, num_epochs):
    epochs = range(1, num_epochs + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('Accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, train_accs, 'g--', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'm--', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Train and Validation Loss & Accuracy over Epochs')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.show()
