import mne
import matplotlib
import transformers
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2 as cv
import os
import argparse
import numpy as np
import math
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
#import CSP  # replace 'your_module' with the actual module where CSP is defined
# Import 'Xdawn' from the appropriate library
from mne.preprocessing import Xdawn
# Import 'ERPCovariances' from the appropriate library
from pyriemann.estimation import ERPCovariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
import joblib


#from cv_bridge import CvBridge
#Vectorizer

# CLASIFICADOR
def clasifiers():
    clfs = {
    'LR': (
        make_pipeline(TfidfVectorizer(), LogisticRegression()),
        {'logisticregression__C': np.exp(np.linspace(-4, 4, 5))},
    ),
    #'LDA': (
    #    make_pipeline(TfidfVectorizer(), LDA(shrinkage='auto', solver='eigen')),
    #    {},
    #),
    'SVM': (
        make_pipeline(TfidfVectorizer(), SVC()),
        {'svc__C': np.exp(np.linspace(-4, 4, 5)), 'svc__kernel': ('linear', 'rbf')},
    ),
    #'CSP LDA': (
    #    make_pipeline(CSP(), LDA(shrinkage='auto', solver='eigen')),
    #    {'csp__n_components': (6, 9, 13), 'csp__cov_est': ('concat', 'epoch')},
    #),
    #'Xdawn LDA': (
    #    #make_pipeline(Xdawn(2, classes=[1]), TfidfVectorizer(), LDA(shrinkage='auto', solver='eigen')),
    #    #{},
    #    make_pipeline(Xdawn(2), TfidfVectorizer(), LDA(shrinkage='auto', solver='eigen')),
    #    {},
    #),
    #'ERPCov TS LR': (
    #    make_pipeline(ERPCovariances(estimator='lwf'), TangentSpace(), LogisticRegression()),
    #    {'erpcovariances__estimator': ('lwf', 'oas')},
    #),
    #'ERPCov MDM': (
    #    make_pipeline(ERPCovariances(estimator='lwf'), MDM()),
    #    {'erpcovariances__estimator': ('lwf', 'oas')},
    #),
    }
    return clfs

# entrenamiento
def clasificar_datos(data, labels):
    # configuramos los clasificadores
    conf_clasif = clasifiers()
    best_classifier =None 
    best_accuracy = 0
    
    # Assuming 'data' is a list of NumPy arrays
    data_flattened = np.array([' '.join(str(item) for item in text) for text in data])

    # para entrenar 
    x_train, x_test, y_train, y_test = train_test_split(data_flattened, labels, test_size=0.2, random_state=42)

    # iteramos en los clasificadores
    #for clf_name, (clf, clf_params) in conf_clasif.items():
    for clf_name, (clf, param_grid) in conf_clasif.items():
        clf.fit(x_train, y_train) # entrenamiento del clasificador
        y_pred = clf.predict(x_test) # clasificacion en el conjunto de prueba
        accuracy = accuracy_score(y_test, y_pred) # calculo de la precision
        print(f"Clasificador: {clf_name}, Precisión: {accuracy:.2f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf
            best_classifier_name = clf_name
            best_y_pred = y_pred

    print("y_pred: ", y_pred)
    
    # SE GUARDA EL CLASIFICADOR EN UN ARCHIVO
    joblib.dump(best_classifier, 'clasificador.pkl')

def leer_imagenes():
     # Creamos un diccionario donde guardaremos los valores calculados
    labels = []
    data = []
    filename = 'train.txt'

    with open(filename, 'r') as f:
        for line in f.readlines():
                fields = line.split()
                #if not args.test:
                print(fields[0], fields[1])
                
                # leemos la imagen y obtenemos su etiqueta
                img = cv.imread(fields[0])
                label = fields[1]

                data.append(img)
                labels.append(label)
    
    return (data, labels)



if __name__ == "__main__":
    #data = '/home/lba35/pepe/src/turtlebot_tracking/src/yolov3.cfg'
    data, labels = leer_imagenes()
    # procesar
    # no se si hay q procesar las imagenes si va a venir mejor

    # clasificar
    clasificar_datos(data, labels)