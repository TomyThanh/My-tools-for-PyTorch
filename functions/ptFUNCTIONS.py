import torch
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import random
from timeit import default_timer
import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
import pathlib
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchinfo import summary   
import lzma
import mmap
import pickle

#Plots the dataset and the prediction
def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    # Sicherstellen, dass alle Eingabedaten die richtige Form haben
    train_data = train_data.squeeze()  # Um 1D-Arrays zu bekommen
    train_labels = train_labels.squeeze()
    test_data = test_data.squeeze()
    test_labels = test_labels.squeeze()

    if predictions is not None:
        predictions = predictions.squeeze()

    # Überprüfen der Dimensionen
    assert train_data.shape == train_labels.shape, "train_data und train_labels haben unterschiedliche Dimensionen!"
    assert test_data.shape == test_labels.shape, "test_data und test_labels haben unterschiedliche Dimensionen!"
    if predictions is not None:
        assert test_data.shape == predictions.shape, "test_data und predictions haben unterschiedliche Dimensionen!"

    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend(prop={"size": 14})
    plt.show()


#TRAINING AND TESTING LOOP
def train_and_test_loop(model, X_train, y_train, X_test, y_test, loss_fn, optimizer, epochs):
    epoch_count = []
    loss_values = []
    testLossValues = []
    
    for epoch in range(epochs):
        # === Training Loop ===
        model.train()  # Model in Trainingsmodus setzen
        y_pred = model(X_train)  # Vorhersage auf Trainingsdaten
        loss = loss_fn(y_pred, y_train)  # Verlust berechnen
        
        optimizer.zero_grad()  # Gradienten auf Null setzen
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimieren

        # Loggen für Plot
        epoch_count.append(epoch)
        loss_values.append(loss.item())

        # === Testing Loop ===
        model.eval()  # Modell in Evaluierungsmodus setzen
        with torch.inference_mode():  # Deaktivierung des Gradienten-Tracking
            test_pred = model(X_test)  # Vorhersage auf Testdaten
            testLoss = loss_fn(test_pred, y_test)  # Testverlust berechnen

        testLossValues.append(testLoss.item())
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item()} | Test Loss: {testLoss.item()}")

    # Rückgabe der Loss-Werte für das Plotten
    return epoch_count, loss_values, testLossValues, test_pred, testLoss


#Accuracy_fn
def accuracy_fn(y_pred, y_true):
    # y_pred sind Logits → zuerst die Klasse mit der höchsten Wahrscheinlichkeit auswählen
    y_pred_classes = y_pred.argmax(dim=1)
    correct = torch.eq(y_true, y_pred_classes).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

#Timer_fn
def print_train_time(start: float, end:float, device: torch.device):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

##Visualizing model's predictions with a COMPLEX FUNCTION
def plot_decision_boundary(model, X, y):
    """Plottet die Entscheidungsgrenze eines Modells (funktioniert für Binary und Multiclass)"""
    
    # Falls Tensor -> Numpy konvertieren
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Erstelle Meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_in = np.c_[xx.ravel(), yy.ravel()]
    X_in_tensor = torch.tensor(X_in, dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        y_pred = model(X_in_tensor)

    # Überprüfe Output-Shape
    if y_pred.shape[1] > 1:
        # Multiclass: Softmax + Argmax
        y_pred = torch.softmax(y_pred, dim=1).argmax(dim=1)
    else:
        # Binary: Sigmoid + Round
        y_pred = torch.round(torch.sigmoid(y_pred))

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.6)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolor="k")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)
    plt.show()


#Plots the learning curve
def plot_learning_curve(train_losses, test_losses, test_accuracies, epochs):
    """
    Plots the learning curve showing loss and accuracy for both training and testing.
    
    Args:
        train_losses (list): Training losses for each epoch.
        test_losses (list): Testing losses for each epoch.
        test_accuracies (list): Testing accuracies for each epoch.
        epochs (int): Total number of epochs.
    """
    # Plotting loss
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Training Loss", color="blue")
    plt.plot(range(epochs), test_losses, label="Test Loss", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), test_accuracies, label="Test Accuracy", color="red")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Display the plots
    plt.show()


# PLOTS LOSS (LEARNING) CURVE FOR train_function()
def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.show()



#Konvertiert ein Tensor Dataset in NumPy-Arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    
    for image, label in dataset:
        images.append(image.numpy())  # image wird als Tensor behandelt, also .numpy() funktioniert hier
        labels.append(label)  # label ist bereits eine einfache Zahl, kein Tensor
        
    images = np.concatenate(images, axis=0)
    labels = np.array(labels)  # labels als NumPy-Array umwandeln
    
    return images, labels


#Plots random images to predict
def plot_random_image(model, data_loader, classes, device='cuda'):
    # Zufälliges Batch aus dem DataLoader auswählen
    images, labels = next(iter(data_loader))
    random_idx = random.randint(0, len(images) - 1)

    target_image = images[random_idx]
    target_label = labels[random_idx]

    # Vorhersage des Modells
    model.eval()
    with torch.inference_mode():  # Keine Gradientenberechnung notwendig
        target_image = target_image.unsqueeze(0).to(device)  # Batch-Größe 1
        outputs = model(target_image)  # Modellvorhersage
        pred_probs = torch.softmax(outputs, dim=1)  # Softmax anwenden
        pred_label = pred_probs.argmax(dim=1).item()  # Die Klasse mit der höchsten Wahrscheinlichkeit auswählen

    # Bild zurückskalieren: Um die Bildwerte von [0, 1] auf [0, 255] zu bringen
    # Wenn du Normalisierung angewendet hast, solltest du sie rückgängig machen.
    target_image = target_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Rückskalieren auf [0, 255] (Standard-PyTorch Normalisierungswerte sind [0.485, 0.456, 0.406] für den Mittelwert
    # und [0.229, 0.224, 0.225] für die Standardabweichung. Diese Werte müssen beim Umkehren der Normalisierung
    # berücksichtigt werden.)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Umkehrung der Normalisierung
    target_image = target_image * std + mean  # Rückskalierung

    # Die Werte sollten jetzt im Bereich [0, 1] liegen, also können wir sie multiplizieren, um sie auf [0, 255] zu bringen.
    target_image = np.clip(target_image * 255, 0, 255).astype(np.uint8)

    # Visualisierung des Bildes
    plt.imshow(target_image)
    plt.title(f"True: {classes[target_label]}, Pred: {classes[pred_label]}", color="green" if pred_label == target_label else "red")
    plt.axis('off')
    plt.show()


#Evaluates and makes predictions with a model created by a CLASS
def eval_model(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn: accuracy_fn,
              device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):

            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred, y)

    #Scale loss and acc to find the average loss/acc per batch
    loss /= len(data_loader)
    acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,  # only works when model was created with class
            "model_loss": loss.item(),
            "model_acc": round(acc)}


#Training Loop
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: accuracy_fn,
               device: torch.device):
    """Performs a training with model trying to learn on data_loader"""
    train_loss = 0
    model.train()
    for batch, (x_train, y_train) in enumerate(data_loader):

        x_train = x_train.to(device)
        y_train = y_train.to(device)

        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        train_loss = train_loss + loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 ==0:
            print(f"Looked at {batch*len(x_train)} samples")

    train_loss = train_loss / len(data_loader)
    return train_loss

    

#Testing Loop
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn: accuracy_fn,
              device: torch.device):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for x_test, y_test in data_loader:

            x_test = x_test.to(device)
            y_test = y_test.to(device)
            
            y_pred_test = model(x_test)
            loss = loss_fn(y_pred_test, y_test)
            acc = accuracy_fn(y_pred_test, y_test)
            
            test_loss = test_loss + loss.item()
            test_acc = test_acc + acc
        
    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader)
    return test_loss, test_acc
       



### CREATING TRAINING AND TESTING LOOP FUNCTIONS ###
def train_function(model: torch.nn.Module,
                   train_dataloader: torch.utils.data.DataLoader,
                   test_dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn.Module,
                   accuracy_fn: accuracy_fn,
                   epochs: int,
                   device):
    
    # 2. Create an empty results dictionary
    results = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs), desc="Training läuft"):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, accuracy_fn, device)

        print(f"Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.0f}%")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results




#COMPARING RESULTS AND MAKING PREDICTIONS
"""
# compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
# compare_resutls["training time"] = [total_train_time_model_0, total_train_time_model_1, total_train_time_model_2]

# 9. Make and evaluate random predictions with best model

def make_predictions(model: nn.Module,
                     data: list,
                     device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #Prepare the sample
            sample = torch.unsqueeze(sample, dim=0).to(device)

            #Forward pass 
            pred_logit = model(sample)

            #Get prediction probability (logit -> pred prob)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            #Get pred off the GPU for further visualizations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_prohs to turn list into a tensor
    return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

#Make predictions
proof_predictions = make_predictions(model_1, test_samples, device)

#Convert prediction propabilities into labels
pred_classes = proof_predictions.argmax(dim=1)

#Plot predictions
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # Find the prediction (in text form, e.g "Sandal")
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form)
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    titel_text = f"Pred: {pred_label} |  Truth: {truth_label}"

    # Check for equality between pred and turth and change color of title text
    if pred_label == truth_label:
        plt.title(titel_text, fontsize=10,c="g")
    else:
        plt.title(titel_text, fontsize=10,c="r")
"""



#======CONFUSION MATRIX==========CONFUSION MATRIX==========CONFUSION MATRIX==========CONFUSION MATRIX======#
"""
# Confusion matrix is really good at evaluating your classification models

# 1. Make predictions with trained model
y_preds = []
model2.eval()
with torch.inference_mode():
    for x_test, y_test in tqdm(test_dataLoader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        y_logit = model2(x_test)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

        y_preds.append(y_pred.cpu())

y_pred_tensor = torch.cat(y_preds)


# 2. Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor,
                         target = test_data.targets)


# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10,7))
"""


# Compares the transformed image with the original one
def plot_transformed_images(image_paths, transform, n=3):
    """Selects random images from a path of images and loads/transforms them then plots the
    original vs the transformed one"""

    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis("off")

            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize= 16)
    plt.show()



# An own version of class_names = train_data.classes AND class_to_idx
def find_classes(directory: str):
    """Finds the class folder names in a target directory."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Could not find any classes in {directory}... please checkt file structure")
    
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx




# Own ImageFolder (customizable)
class ImageFolderCustom(Dataset):

    def __init__(self, targ_dir: str, transform=None):
        super().__init__() 
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transforms = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int):
        """Opens an image via a path and returns it."""
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        """Returns the total number of samples"""
        return len(self.paths)
    
    def __getitem__(self, index: int):
        """Returns one sample of data, data and label (X,y)"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


#Use this in combination with CUSTOMIZABLE ImageFolder :) MONKEYSSSSS
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool=True):
    
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes n shouldn't be larger than 10.")
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16,8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image_adjust = targ_image.permute(1,2,0)

        plt.subplot(1,n, i+1)
        plt.imshow(targ_image_adjust, cmap="gray")
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            
        plt.title(title)
    
    plt.show()



# Insert your own image and let the model predict what it is :) MONKEY
def custom_predictions(class_names, model, paths: str, transformer: torchvision.transforms, device):

    """MAKES A PREDICTION ON YOUR OWN IMAGE"""
    custom_image = torchvision.io.read_image(paths)
    custom_image_transformers = transformer
    transformed_custom_image = custom_image_transformers(custom_image) / 255
    transformed_custom_image = transformed_custom_image.type(torch.float32)
    transformed_custom_image = transformed_custom_image.to(device)

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        custom_prediction = model(transformed_custom_image.unsqueeze(0))
    custom_pred_prob = torch.softmax(custom_prediction, dim=1)
    custom_pred_label = torch.argmax(custom_pred_prob, dim=1)

    plt.imshow(custom_image.permute(1,2,0).numpy())
    plt.title(f"Your Image is PROBABLY a {class_names[custom_pred_label]}")    
    plt.axis("off")
    plt.show()





"""===========================LARGE LANGUAGE MODEL==========================="""

# TOKENIZER --> wandelt Wörter in Tokens um (Zahlen)
class CharTokenizer:
    def __init__(self, chars=None, vocab_file=None):
        """
        Initialisiert den Tokenizer.
        Entweder mit einer Liste von chars ODER mit einem Pfad zu einer Vokabular-Datei.
        """
        if chars:
            self.chars = sorted(list(set(chars)))
        elif vocab_file:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.chars = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Entweder chars oder vocab_file muss angegeben werden.")
        
        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Wandelt einen String in eine Liste von Token-IDs um."""
        return [self.string_to_int[c] for c in text if c in self.string_to_int]

    def decode(self, tokens):
        """Wandelt eine Liste von Token-IDs zurück in einen String."""
        return "".join([self.int_to_string[i] for i in tokens])




def get_batch(data, block_size, batch_size, device):
    """
    Bereitet einen Batch von Trainingsdaten vor.

    Args:   
        data (torch.Tensor): Tensor mit Token-IDs (ganzer Text bereits tokenisiert).
        block_size (int): Länge der Eingabesequenz.
        batch_size (int): Anzahl der Sequenzen pro Batch.
        device (torch.device): Gerät, auf das Tensoren geladen werden sollen.

    Returns:
        x (torch.Tensor): Eingabesequenzen (batch_size x block_size).
        y (torch.Tensor): Zielsequenzen (batch_size x block_size), jeweils einen Token versetzt.
    """
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# Nimmt EIN Stück aus einer riesigen Textdatei und wandelt sie diese in Tokens um: ONE CHUNK PER EPOCH
def get_random_chunk(filename, tokenizer, block_size=512):
    """
    Holt zufälligen Chunk aus großer Datei mit mmap – ideal für Language Modeling.
    Gibt ein dict zurück: input_ids, attention_mask, labels
    """
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            # Stelle sicher, dass genug Platz für einen Block ist
            max_start = file_size - block_size
            if max_start <= 0:
                raise ValueError("Datei zu klein für gegebenen block_size")

            start_pos = random.randint(0, max_start)
            mm.seek(start_pos)
            block = mm.read(block_size)

            # Decode + bereinigen
            decoded = block.decode("utf-8", errors="ignore").replace("\r", "").replace("\n", " ")

            # Tokenisieren
            tokens = tokenizer(decoded, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            }



# Nimmt sich LOTS OF CHUNKS PER EPOCH
def get_random_chunks(filename, tokenizer, block_size=512, num_chunks=100):
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            for _ in range(num_chunks):
                start = random.randint(0, file_size - block_size)
                mm.seek(start)
                block = mm.read(block_size)
                text = block.decode("utf-8", errors="ignore").replace("\r", "").replace("\n", " ")

                # Tokenize hier:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=block_size,
                    return_tensors="pt"
                )

                input_ids = tokens["input_ids"].squeeze(0)         # (block_size,)
                attention_mask = tokens["attention_mask"].squeeze(0)

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone()
                }




# read_chunks: liest die Datei Stück für Stück linear, ideal bei kleineren Dateien.
def read_chunks(file_path, chunk_size=512):
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk



# Die Funktion bereitet einen Trainings- oder Test-Batch vor – also Eingabe- und Zielsequenzen MIT get_random_chunk()!!!!
def get_batch_with_random_chunk(split, block_size, batch_size, encoder, filename_train, filename_val, device):
    data = get_random_chunk(split, block_size, batch_size, encoder, filename_train, filename_val)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


# Starts at the last checkpoint of the LLM Model
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r"checkpoint-\d+", d)]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


# Generiert einen text anhand des prompts
def generate_text(model,prompt, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)