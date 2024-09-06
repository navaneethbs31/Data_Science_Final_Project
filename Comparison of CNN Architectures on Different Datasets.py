# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define global variables
DATASETS = ['MNIST', 'FMNIST', 'CIFAR10']
MODELS = ['LeNet5', 'AlexNet', 'GoogLeNet', 'VGG16', 'ResNet50', 'Xception', 'SENet']
BATCH_SIZE = 64
EPOCHS = 10  # Adjust as needed


# Function to load and preprocess datasets
def load_datasets(dataset_name):
    if dataset_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        num_classes = 10
        input_shape = (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    elif dataset_name == 'FMNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        num_classes = 10
        input_shape = (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    elif dataset_name == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
        input_shape = (32, 32, 3)
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError("Unsupported dataset")

    y_train = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, y_test_cat, input_shape, num_classes


# Function to build LeNet-5
def build_lenet5(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape, padding='same'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='tanh'))
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# Function to build AlexNet
def build_alexnet(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# Function to build GoogLeNet (InceptionV3)
def build_googlenet(input_shape, num_classes):
    base_model = applications.InceptionV3(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


# Function to build VGG16
def build_vgg16(input_shape, num_classes):
    base_model = applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


# Function to build ResNet50
def build_resnet50(input_shape, num_classes):
    base_model = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


# Function to build Xception
def build_xception(input_shape, num_classes):
    base_model = applications.Xception(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


# Function to build SENet (Simplified version)
def build_senet(input_shape, num_classes):
    # Squeeze and Excitation block
    def se_block(input_tensor, reduction=16):
        channels = input_tensor.shape[-1]
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Dense(channels // reduction, activation='relu')(se)
        se = layers.Dense(channels, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, channels))(se)
        return layers.multiply([input_tensor, se])

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


# Dictionary to map model names to their build functions
MODEL_BUILDERS = {
    'LeNet5': build_lenet5,
    'AlexNet': build_alexnet,
    'GoogLeNet': build_googlenet,
    'VGG16': build_vgg16,
    'ResNet50': build_resnet50,
    'Xception': build_xception,
    'SENet': build_senet
}


# Function to compile and train the model
def compile_and_train(model, x_train, y_train, x_val, y_val, dataset_name, model_name):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_val, y_val),
                        verbose=2)
    return history


# Function to evaluate the model
def evaluate_model(model, x_test, y_test_cat, y_test, dataset_name, model_name):
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    return accuracy, precision, recall, f1, report


# Function to plot training history
def plot_history(history, dataset_name, model_name):
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main execution loop
def main():
    # To store all results
    results = {}

    for dataset in DATASETS:
        print(f"\nLoading and preprocessing {dataset} dataset...")
        x_train, y_train, x_test, y_test, y_test_cat, input_shape, num_classes = load_datasets(dataset)

        # Split a validation set from training data
        validation_split = 0.1
        val_size = int(x_train.shape[0] * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train_new = x_train[val_size:]
        y_train_new = y_train[val_size:]

        results[dataset] = {}

        for model_name in MODELS:
            print(f"\nBuilding and training {model_name} on {dataset}...")
            builder = MODEL_BUILDERS.get(model_name)
            if not builder:
                print(f"Model {model_name} is not defined.")
                continue
            try:
                model = builder(input_shape, num_classes)
            except Exception as e:
                print(f"Error building {model_name}: {e}")
                continue

            # Adjust input shape if necessary
            # Some models require minimum image size, e.g., InceptionV3 requires at least 75x75
            # To handle this, you might need to resize images. Here, we'll resize all to 75x75 if needed.
            required_size = 75
            if input_shape[0] < required_size or input_shape[1] < required_size:
                print(f"Resizing images to {required_size}x{required_size} for {model_name}...")
                x_train_resized = tf.image.resize(x_train_new, [required_size, required_size]).numpy()
                x_val_resized = tf.image.resize(x_val, [required_size, required_size]).numpy()
                x_test_resized = tf.image.resize(x_test, [required_size, required_size]).numpy()
            else:
                x_train_resized = x_train_new
                x_val_resized = x_val
                x_test_resized = x_test

            # For grayscale images and models expecting 3 channels, replicate channels
            if input_shape[2] == 1 and model_name in ['GoogLeNet', 'VGG16', 'ResNet50', 'Xception', 'SENet', 'AlexNet']:
                x_train_resized = np.repeat(x_train_resized, 3, axis=3)
                x_val_resized = np.repeat(x_val_resized, 3, axis=3)
                x_test_resized = np.repeat(x_test_resized, 3, axis=3)
                input_shape_model = (x_train_resized.shape[1], x_train_resized.shape[2], 3)
            else:
                input_shape_model = (x_train_resized.shape[1], x_train_resized.shape[2], x_train_resized.shape[3])

            # Rebuild the model with new input shape if resized or channels changed
            if input_shape_model != input_shape:
                print(f"Rebuilding {model_name} with new input shape {input_shape_model}...")
                try:
                    model = builder(input_shape_model, num_classes)
                except Exception as e:
                    print(f"Error rebuilding {model_name} with new input shape: {e}")
                    continue

            # Compile and train
            history = compile_and_train(model, x_train_resized, y_train_new, x_val_resized, y_val, dataset, model_name)

            # Plot training history
            plot_history(history, dataset, model_name)

            # Evaluate the model
            accuracy, precision, recall, f1, report = evaluate_model(model, x_test_resized, y_test_cat, y_test, dataset,
                                                                     model_name)
            print(f"Evaluation on {dataset} with {model_name}:")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

            # Store the results
            results[dataset][model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': report,
                'history': history.history
            }

    # After all trainings, plot comparative metrics
    plot_comparative_metrics(results)

    # Save the results to a file if needed
    # You can implement saving to JSON or CSV as per requirements


# Function to plot comparative metrics
def plot_comparative_metrics(results):
    for dataset, models_results in results.items():
        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        model_names = list(models_results.keys())
        accuracies = [models_results[m]['accuracy'] for m in model_names]
        plt.bar(model_names, accuracies, color='skyblue')
        plt.ylim(0, 1)
        plt.title(f'Accuracy Comparison on {dataset}')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.show()

        # Plot Precision
        plt.figure(figsize=(10, 6))
        precisions = [models_results[m]['precision'] for m in model_names]
        plt.bar(model_names, precisions, color='lightgreen')
        plt.ylim(0, 1)
        plt.title(f'Precision Comparison on {dataset}')
        plt.xlabel('Models')
        plt.ylabel('Precision')
        plt.xticks(rotation=45)
        plt.show()

        # Plot Recall
        plt.figure(figsize=(10, 6))
        recalls = [models_results[m]['recall'] for m in model_names]
        plt.bar(model_names, recalls, color='salmon')
        plt.ylim(0, 1)
        plt.title(f'Recall Comparison on {dataset}')
        plt.xlabel('Models')
        plt.ylabel('Recall')
        plt.xticks(rotation=45)
        plt.show()

        # Plot F1-Score
        plt.figure(figsize=(10, 6))
        f1_scores = [models_results[m]['f1_score'] for m in model_names]
        plt.bar(model_names, f1_scores, color='violet')
        plt.ylim(0, 1)
        plt.title(f'F1-Score Comparison on {dataset}')
        plt.xlabel('Models')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    main()
