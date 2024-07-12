"""
import tensorflow as tf

from tensorflow.keras import models, layers
# from tensorflow.keras.preprocessing.image import Data,load_img
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Sequential

accuracy = 0

for i in range(5):
    print(f"Iteration {i+1}")
    images_path = r"C:\Users\Anirudh\Documents\GitHub\FractureDetection\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification\train"  # add path here
    img_labels = ['Fractured', 'Not Fractured']

    def get_data(input_string):
        data = []
        path = os.path.join(images_path, input_string)
        label = img_labels.index(input_string)

        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                if img_arr is None:
                    print(f"Failed to read {os.path.join(path, img)}")
                    continue
                resized = cv2.resize(img_arr, (224, 224))
                data.append([resized, label])
            except Exception as e:
                print(f"Error reading {os.path.join(path, img)}: {e}")

        return data

    X_Frac = get_data('Fractured')
    X_notFrac = get_data('Not Fractured')
    X = np.array([x[0] for x in X_Frac + X_notFrac])
    y = np.array([x[1] for x in X_Frac + X_notFrac])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = Sequential([
        Conv2D(256, (3, 3), activation='relu', input_shape=(145, 145, 3)),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc}")

    if test_acc > accuracy:
        model.save()
