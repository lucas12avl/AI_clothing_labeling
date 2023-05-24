__authors__ = ['1636290, 1631153, 1636589']
__group__ = ['DJ.10']

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import numpy as np #para usar numpy hay que poner np.<lo que queramos usar de la libreria>
import Kmeans as km
import KNN as knn
import time as t

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
        if pregunta_forma in forma:
            if pregunta_color in color: #falla si se pone en ==
                imag.append(i)
            
    return np.array(imag)

def my_colors():
    etiquetes = []
    
    for test in test_imgs:
        Km = km.KMeans(test, K = 2) #dependiendo de la K dará unos resultados u otros, pero funciona
        Km.fit()
        etiquetes.append(km.get_colors(Km.centroids))
    
    Knn = knn.KNN(train_imgs, train_class_labels) 
    class_label = Knn.predict(test_imgs, 3)
    
    return np.array(etiquetes), class_label
        


    
#test per a les Funcions  d'analisi qualitatiu
def test_Retrieval_by_color():
        
    #Llamamos a Retrieval_by_color:
    rojo = Retrieval_by_color(test_imgs, my_test_color_labels, "Red")
    verde = Retrieval_by_color(test_imgs, my_test_color_labels, "Green")
    azul = Retrieval_by_color(test_imgs, my_test_color_labels, "Blue")

    #Visualizamos las imagenes que se han recogido
    visualize_retrieval(rojo, 5, title="Ropa de color rojo")
    visualize_retrieval(verde, 5, title="Ropa de color verde")
    visualize_retrieval(azul, 5, title="Ropa de color azul")


def test_Retrieval_by_shape():

    #Llamamos a Retrieval_by_shape:
    shape = Retrieval_by_shape(test_imgs, my_class_label, "Heels")
    visualize_retrieval(shape, 5, title="Forma: Heels")

    shape = Retrieval_by_shape(test_imgs, my_class_label, "Dresses")
    visualize_retrieval(shape, 5, title="Forma: Dresses")
    
def test_Retrieval_combined():

    combined = Retrieval_combined(test_imgs, my_class_label, my_test_color_labels, "Shirts", "Blue")
    visualize_retrieval(combined, 5, title="Camisetas azules")

    combined = Retrieval_combined(test_imgs, my_class_label, my_test_color_labels, "Sandals", "Red")
    visualize_retrieval(combined, 5, title="Sandalias rojas")
    
    combined = Retrieval_combined(test_imgs, my_class_label, my_test_color_labels, "Dresses", "White")
    visualize_retrieval(combined, 5, title="Vestidos blancos")

#                   #
#   TEST QUANTITAU  #
#                   #

def test_Kmean_statistics():
  ka = km.KMeans(train_imgs)
  WCD_r, i, temps = Kmean_statistics(ka, 3)
  print("\n WithinClassDistance:", WCD_r, "\n clusters:",i, "\n time:",temps)   
    
def test_get_shape_accuracy(gt_labels):
    total = len(my_class_label)
    correct = np.sum(my_class_label == gt_labels)
    result = correct / total * 100
    
    return result
    
def Kmean_statistics(kmeans, kmax):
    i = []
    WCD_r = []
    temps = []
    
    for k in range(2, kmax+1):
        kmeans.K = k #si se cambia a 'k' minuscula el tiempo baja MUCHO
        start = t.time()
        kmeans.fit()
        end = t.time()
        WCD = kmeans.withinClassDistance()
        
        i.append(k)
        temps.append((end - start))
        WCD_r.append(WCD)
        print("hecho")
    
    return WCD_r, i, temps


def get_shape_accuracy(gt):
    total = len(my_class_label)
    et_corr = np.count_nonzero(my_class_label == gt) #con for va un poco más lento
    percentatge = (et_corr / total) * 100
    
    print("Shape accuracy is: %s%%"% percentatge)

    return percentatge









if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    my_test_color_labels, my_class_label = my_colors()
    #test per a les Funcions  d'analisi qualitatiu
    #test_Retrieval_by_color()
    #test_Retrieval_by_shape()
    #test_Retrieval_combined()
    #Posible problema, que tarda casi 3s en pasar.
    #test_Kmean_statistics()
    get_shape_accuracy(test_class_labels)
