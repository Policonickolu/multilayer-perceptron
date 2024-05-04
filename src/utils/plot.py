
from matplotlib import pyplot as plt

def plot_loss_and_accuracy(losses, accuracies, val_losses, val_accuracies):
    epochs = range(1, len(losses) + 1)

    # Plot the loss learning curve
    plt.figure()
    plt.plot(epochs, losses, label='training loss')
    plt.plot(epochs, val_losses, '--', label='validation loss')
    plt.title('Loss Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the accuracy learning curve
    plt.figure()
    plt.plot(epochs, accuracies, label='training acc')
    plt.plot(epochs, val_accuracies, label='validation acc')
    plt.title('Accuracy Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()