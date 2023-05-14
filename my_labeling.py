__authors__ = ['1636290, 1631153, 1636589']
__group__ = ['DJ.10']

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import numpy as np #para usar numpy hay que poner np.<lo que queramos usar de la libreria>
import Kmeans
import math
import time

#Funcions  d'analisi qualitatiu
def Retrieval_by_color(imagenes, etiquetes, pregunta):
    imag = []
    
    for i, clase in zip(imagenes, etiquetes):
        if pregunta in clase: #ponemos 'in' porque si se trata de una cadena un == falla
            imag.append(i)
            
    return np.array(imag)
    
def Retrieval_by_shape(imagenes, etiqueta, pregunta):
    imag = []
    
    for i, clase in zip(imagenes, etiqueta):
        if pregunta == clase:
            imag.append(i)
            
    return np.array(imag)

def Retrieval_combined(imagenes, etiqueta_forma, etiqueta_color, pregunta_forma, pregunta_color):
    imag = []
    
    for i, forma, color in zip(imagenes, etiqueta_forma, etiqueta_color):
        if pregunta_forma in forma and pregunta_color == color:
            imag.append(i)
            
    return np.array(imag)

    
#test per a les Funcions  d'analisi qualitatiu
def test_Retrieval_by_color():

    #Llamamos a Retrieval_by_color:
    results_red = Retrieval_by_color(test_imgs, test_color_labels, "Red")
    results_green = Retrieval_by_color(test_imgs, test_color_labels, "Green")
    results_blue = Retrieval_by_color(test_imgs, test_color_labels, "Blue")

    #Visualizamos las imagenes que se han recogido
    visualize_retrieval(results_red, 5, title="Ropa de color rojo")
    visualize_retrieval(results_green, 5, title="Ropa de color verde")
    visualize_retrieval(results_blue, 5, title="Ropa de color azul")


def test_Retrieval_by_shape():

    #Llamamos a Retrieval_by_shape:
    results = Retrieval_by_shape(test_imgs, test_class_labels, "Heels")
    visualize_retrieval(results, 5, title="Forma: Heels")

    results = Retrieval_by_shape(test_imgs, test_class_labels, "Dresses")
    visualize_retrieval(results, 5, title="Forma: Dresses")
    
def test_Retrieval_combined():

    results = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, "Shirts", "Blue")
    print(results)
    visualize_retrieval(results, 5, title="Retrieval T-Shirt and Blue clothes")

    results = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, "Sandals", "Red")
    print(results)
    visualize_retrieval(results, 5, title="Retrieval Skirt and Red clothes")
    
    results = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, "Dresses", "White")
    print(results)
    visualize_retrieval(results, 5, title="Retrieval Sneaker and White clothes")











if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    #test_Retrieval_by_color() #Problema: No pasa el test en menos de 3s
    #test_Retrieval_by_shape() #Problema: No pasa el test en menos de 3s
    # You can start coding your functions here
    #test_Retrieval_combined()
