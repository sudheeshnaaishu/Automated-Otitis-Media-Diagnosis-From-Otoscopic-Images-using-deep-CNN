from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import END  

import tkinter as tk
from tkinter import Text, ttk


import os
import cv2
import pickle
import joblib
import hashlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk, END, Label, Text, Tk

from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler, normalize

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Deep Learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (
    Convolution2D, MaxPooling2D, Conv2D, Flatten, Dense, 
    Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121, Xception
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

# Others
from tinydb import TinyDB, Query

def setBackground():
    global bg_photo, bg_label
    image_path = r"background.png"
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.lower()


main = Tk()
main.geometry("1300x1200")

screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

title = Label(main, text="Automated Otitis Media Diagnosis from Otoscopic Images using Deep CNN Model",
              font=('times', 20, 'bold'),
              bg='lightblue', fg='black')
title.place(x=400, y=10)

text = Text(main, height=25, width=80, font=('times', 12, 'bold'))
text.place(x=300, y=200)

setBackground() 



global filename
global X, Y
global model
global accuracy
global accuracy, precision, recall, f1

# Initialize empty lists for features and labelsz
X = []
Y = []

base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
model_folder = "model"

def clear_text():
    text.delete('1.0', END)


def uploadDataset():
    clear_text()
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")

def DenseNet121_feature_extraction():
    clear_text()
    global X, Y, base_model,categories,filename
    text.delete('1.0', END)

    model_data_path = "model/X.npy"
    model_label_path_GI = "model/Y.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path_GI):
        X = np.load(model_data_path)
        Y = np.load(model_label_path_GI)
    else:
 
        X = []
        Y = []
        data_folder=filename
        for class_label, class_name in enumerate(categories):
            class_folder = os.path.join(data_folder, class_name)
            for img_file in os.listdir(class_folder):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(class_folder, img_file)
                    print(img_path)
                    img = image.load_img(img_path, target_size=(128,128))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = base_model.predict(x)
                    features = np.squeeze(features)  # Flatten the features
                    X.append(features)
                    Y.append(class_label)
        # Convert lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Save processed images and labels
        np.save(model_data_path, X)
        np.save(model_label_path_GI, Y)
            
    text.insert(END, "Image Preprocessing Completed\n")
    text.insert(END, "Xception Feature Extraction completed\n")
    text.insert(END, f"Feature Dimension: {X.shape}\n")


  
def Train_test_spliting():
    clear_text()
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    
    
    X_downsampled = X
    Y_downsampled = Y
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]  
    else:
        indices = np.arange(X_downsampled.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]
        
    
    X_train, X_test, y_train, y_test = train_test_split(X_downsampled, Y_downsampled, test_size=0.2, random_state=42)

    text.insert(END, f"Input Data Train  Size: {X_train.shape}\n")
    text.insert(END, f"Input Data Test  Size: {X_test.shape}\n")
    text.insert(END, f"Output  Train Size: {y_train.shape}\n")
    text.insert(END, f"Output  Test Size: {y_test.shape}\n")

    
def performance_evaluation(algorithm, model, X_test, y_test, categories, text):

    # -------- Clear previous output --------
    # (Optional if you already call clear_text() in button)
    # text.delete('1.0', END)

    # -------- Predictions --------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    text.insert(END, f"{algorithm} Accuracy  : {acc:.2f}\n")
    text.insert(END, f"{algorithm} Precision : {prec:.2f}\n")
    text.insert(END, f"{algorithm} Recall    : {rec:.2f}\n")
    text.insert(END, f"{algorithm} F1-Score  : {f1:.2f}\n\n")

    # -------- Confusion Matrix (ALL MODELS) --------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        cmap='cubehelix',
        xticklabels=categories,
        yticklabels=categories
    )
    plt.title(f"{algorithm} Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

    # -------- Classification Report --------
    text.insert(END, f"{algorithm} Classification Report\n")
    text.insert(END, classification_report(y_test, y_pred, target_names=categories))
    text.insert(END, "\n")

    # -------- ROC Curve (SKIP FOR NEAREST CENTROID) --------
    if algorithm != "Nearest Centroid" and hasattr(model, "predict_proba"):

        y_score = model.predict_proba(X_test)

        if len(categories) > 2:
            y_test_bin = label_binarize(y_test, classes=range(len(categories)))

            auc_score = roc_auc_score(
                y_test_bin,
                y_score,
                average="macro",
                multi_class="ovr"
            )

            text.insert(END, f"{algorithm} AUC Score: {auc_score*100:.2f}\n\n")

            plt.figure(figsize=(8, 6))
            for i, cls in enumerate(categories):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, label=cls)

        else:
            auc_score = roc_auc_score(y_test, y_score[:, 1])
            text.insert(END, f"{algorithm} AUC Score : {auc_score*100:.2f}\n\n")

            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{algorithm} ROC Curve")
        plt.legend()
        plt.show()




def Model_NearestCentroid():
    clear_text()
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    
    model_filename = os.path.join(model_folder, "NearestCentroid_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = NearestCentroid()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)
    
   # Y_pred = Model1.predict(X_test)
    performance_evaluation(algorithm="Nearest Centroid",
    model=Model1,
    X_test=X_test,
    y_test=y_test,
    categories=categories,
    text=text)

def Model_XGBoost():
    clear_text()
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "XGBoost_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        # XGBoost Classifier
        xgb = XGBClassifier(
            learning_rate=0.01,
        )

        Model1 = xgb
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    # Evaluation (same pattern as Extra Trees)
    performance_evaluation(
        algorithm="XGBoost",
        model=Model1,
        X_test=X_test,
        y_test=y_test,
        categories=categories,
        text=text
    )

    text.insert(END, "Shape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n")

def Model_KNN():
    clear_text()
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "KNN_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        knn = KNeighborsClassifier(n_neighbors=100, metric='chebyshev')
        
        Model1 = knn
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    Y_pred = Model1.predict(X_test)
    performance_evaluation(algorithm="KNN",
    model=Model1,
    X_test=X_test,
    y_test=y_test,
    categories=categories,
    text=text)
    text.insert(END, "Shape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")   
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n")


def cnnModel():
    clear_text()
    global model
    text.delete('1.0', END)

    y_train_cat = to_categorical(y_train, num_classes=len(categories))
    y_test_cat  = to_categorical(y_test,  num_classes=len(categories))

    Model_file = os.path.join(model_folder, "Dense_CNN.json")
    Model_weights = os.path.join(model_folder, "Dense_CNN_weights.h5")

    if os.path.exists(Model_file):
        with open(Model_file, "r") as f:
            model = model_from_json(f.read())
        model.load_weights(Model_weights)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        text.insert(END, "Dense CNN Model Loaded\n")

    else:
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(categories), activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        hist = model.fit(
            X_train, y_train_cat,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test_cat),
            verbose=2
        )

        model.save_weights(Model_weights)
        with open(Model_file, "w") as f:
            f.write(model.to_json())

        text.insert(END, "Dense CNN Model Trained & Saved\n")
        
    y_prob = model.predict(X_test)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred) * 100
    text.insert(END, f"Proposed Dense CNN Accuracy: {acc:.2f}%\n")
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec  = recall_score(y_test, y_pred, average='macro') * 100
    f1   = f1_score(y_test, y_pred, average='macro') * 100

    text.insert(END, "\nDense CNN Performance \n")
    text.insert(END, f"Accuracy  : {acc:.2f}\n")
    text.insert(END, f"Precision : {prec:.2f}\n")
    text.insert(END, f"Recall    : {rec:.2f}\n")
    text.insert(END, f"F1-Score  : {f1:.2f}\n\n")

    # ---------------- Classification Report ----------------
    text.insert(END, "Classification Report\n")
    text.insert(END, classification_report(y_test, y_pred, target_names=categories))
    text.insert(END, "\n")

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        cmap='cubehelix',
        xticklabels=categories,
        yticklabels=categories
    )
    plt.title("Dense CNN Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

    y_test_bin = label_binarize(y_test, classes=range(len(categories)))

    auc_score = roc_auc_score(
        y_test_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    text.insert(END, f"Dense CNN AUC Score (Macro Avg): {auc_score*100:.2f}\n\n")

    plt.figure(figsize=(8, 6))
    for i in range(len(categories)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=categories[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Dense CNN ROC Curve")
    plt.legend()
    plt.show()



def predict():
    clear_text()
    global base_model, categories, model

    # FIXED category order (must match training)
    categories = [
        'Acute Otitis Media',
        'Cerumen Impaction',
        'Chronic Otitis Media',
        'Myringosclerosis',
        'Normal'
    ]

    # ---------------- Load CNN Model ----------------
    Model_file    = os.path.join(model_folder, "Dense_CNN.json")
    Model_weights = os.path.join(model_folder, "Dense_CNN_weights.h5")

    if not (os.path.exists(Model_file) and os.path.exists(Model_weights)):
        messagebox.showerror("Error", "CNN model not found. Train CNN first.")
        return

    with open(Model_file, "r") as f:
        model = model_from_json(f.read())

    model.load_weights(Model_weights)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ---------------- Select Image ----------------
    filename = filedialog.askopenfilename(initialdir="testImages")
    if not filename:
        return

    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = densenet_preprocess(x)
    x = np.expand_dims(x, axis=0)

    features = base_model.predict(x)
    features = features.reshape(1, -1)

    # ---------------- CNN Prediction ----------------
    probs = model.predict(features)
    preds = np.argmax(probs, axis=1)[0]

    class_label = categories[preds]

    # ---------------- Display Result ----------------
    img_cv = cv2.imread(filename)
    img_cv = cv2.resize(img_cv, (700, 400))

    text_to_display = f'Output Classified as: {class_label}'
    cv2.putText(
        img_cv,
        text_to_display,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(f'Output Classified as: {class_label}', img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def close():
    main.destroy()
    #text.delete('1.0', END)



db = TinyDB("users_db.json")
users_table = db.table("users")

def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            User = Query()
            if users_table.search((User.username == username) & (User.role == role)):
                messagebox.showerror("Error", f"{role} with this username already exists!")
                return

            users_table.insert({
                "username": username,
                "password": hashed_password,
                "role": role
            })

            messagebox.showinfo("Success", f"{role} Signup Successful!")
            signup_window.destroy()
            show_login_screen()
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)


def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            User = Query()
            result = users_table.search(
                (User.username == username) &
                (User.password == hashed_password) &
                (User.role == role)
            )

            if result:
                messagebox.showinfo("Success", f"{role} Login Successful!")
                login_window.destroy()
                clear_buttons()
                if role == "Admin":
                    show_main_buttons()
                elif role == "User":
                    show_user_buttons()
            else:
                messagebox.showerror("Error", "Invalid Credentials!")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)


def clear_buttons():
    for widget in main.winfo_children():
        if widget not in [title, text, bg_label]:
            widget.destroy()
    bg_label.lower()
    title.lift()
    text.lift()


import tkinter as tk

def show_main_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()  

    tk.Button(main, text="Upload Dataset",
              command=uploadDataset,
              font=font1).place(x=20, y=100)

    tk.Button(main, text="Image Feature Extraction",
              command=DenseNet121_feature_extraction,
              font=font1).place(x=20, y=150)

    tk.Button(main, text="Train Test Splitting",
              command=Train_test_spliting,
              font=font1).place(x=20, y=200)

    tk.Button(main, text="Train Nearest Centroid",
              command=Model_NearestCentroid,
              font=font1).place(x=20, y=250)

    tk.Button(main, text="Train XGBoost Model",
              command=Model_XGBoost,
              font=font1).place(x=20, y=300)
    
    tk.Button(main, text="Train KNN Model",
              command=Model_KNN,
              font=font1).place(x=20, y=350)

    tk.Button(main, text="Train proposed CNN Model",
              command=cnnModel,
              font=font1).place(x=20, y=400)


    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=20, y=450)

def show_user_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    tk.Button(main, text="Prediction",
              command=predict,
              font=font1).place(x=20, y=200)

    tk.Button(main, text="Exit", command=close, font=font1).place(x=20, y=250)

    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=20, y=300)

def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

    tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='red').place(x=100, y=100)
    tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='red').place(x=400, y=100)
    tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=700, y=100)
    tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=1000, y=100)

def close():
    main.destroy()

show_login_screen()
main.mainloop()

