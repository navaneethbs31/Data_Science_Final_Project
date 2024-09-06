import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Load and Preprocess the Datasets
def load_and_preprocess_data():
    # Load datasets
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
    (X_train_fmnist, y_train_fmnist), (X_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
    (X_train_cifar10, y_train_cifar10), (X_test_cifar10, y_test_cifar10) = cifar10.load_data()

    # Preprocess datasets (normalize and one-hot encode)
    def preprocess_data(X_train, X_test, y_train, y_test, channels):
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], channels).astype('float32') / 255.0
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], channels).astype('float32') / 255.0
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return X_train, X_test, y_train, y_test

    X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = preprocess_data(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist, 1)
    X_train_fmnist, X_test_fmnist, y_train_fmnist, y_test_fmnist = preprocess_data(X_train_fmnist, X_test_fmnist, y_train_fmnist, y_test_fmnist, 1)
    X_train_cifar10, X_test_cifar10, y_train_cifar10, y_test_cifar10 = preprocess_data(X_train_cifar10, X_test_cifar10, y_train_cifar10, y_test_cifar10, 3)

    return (X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist), \
           (X_train_fmnist, y_train_fmnist, X_test_fmnist, y_test_fmnist), \
           (X_train_cifar10, y_train_cifar10, X_test_cifar10, y_test_cifar10)

# 2. Implement CNN Architectures
def build_lenet5(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape),
        layers.AveragePooling2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Define other architectures (AlexNet, GoogLeNet, VGGNet, ResNet, Xception, SENet) similarly...

# 3. Train Each Model
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

# 4. Evaluate the Performance
def evaluate_model(model, X_test, y_test, num_classes):
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)
    report = classification_report(y_true, y_pred_classes, target_names=[str(i) for i in range(num_classes)])
    return report

# 5. Plot the Loss Curves and Performance Metrics
def plot_loss_curves(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main function to execute the entire process
def main():
    # Load and preprocess data
    (X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist), \
    (X_train_fmnist, y_train_fmnist, X_test_fmnist, y_test_fmnist), \
    (X_train_cifar10, y_train_cifar10, X_test_cifar10, y_test_cifar10) = load_and_preprocess_data()

    input_shapes = {'mnist': (28, 28, 1), 'fmnist': (28, 28, 1), 'cifar10': (32, 32, 3)}
    num_classes = 10

    # Example: Train LeNet-5 on MNIST
    model = build_lenet5(input_shapes['mnist'], num_classes)
    history = train_model(model, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist)

    # Evaluate the model
    report = evaluate_model(model, X_test_mnist, y_test_mnist, num_classes)
    print(report)

    # Plot the loss curves
    plot_loss_curves(history)

# Run the main function
if __name__ == "__main__":
    main()
