__authors__ = ['1636290, 1631153, 1636589']
__group__ = ['DJ.10']

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import numpy as np #para usar numpy hay que poner np.<lo que queramos usar de la libreria>
import Kmeans as km
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



    
#test per a les Funcions  d'analisi qualitatiu
def test_Retrieval_by_color():

    #Llamamos a Retrieval_by_color:
    rojo = Retrieval_by_color(test_imgs, test_color_labels, "Red")
    verde = Retrieval_by_color(test_imgs, test_color_labels, "Green")
    azul = Retrieval_by_color(test_imgs, test_color_labels, "Blue")

    #Visualizamos las imagenes que se han recogido
    visualize_retrieval(rojo, 5, title="Ropa de color rojo")
    visualize_retrieval(verde, 5, title="Ropa de color verde")
    visualize_retrieval(azul, 5, title="Ropa de color azul")


def test_Retrieval_by_shape():

    #Llamamos a Retrieval_by_shape:
    shape = Retrieval_by_shape(test_imgs, test_class_labels, "Heels")
    visualize_retrieval(shape, 5, title="Forma: Heels")

    shape = Retrieval_by_shape(test_imgs, test_class_labels, "Dresses")
    visualize_retrieval(shape, 5, title="Forma: Dresses")
    
def test_Retrieval_combined():

    combined = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, "Shirts", "Blue")
    visualize_retrieval(combined, 5, title="Camisetas azules")

    combined = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, "Sandals", "Red")
    visualize_retrieval(combined, 5, title="Sandalias rojas")
    
    combined = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, "Dresses", "White")
    visualize_retrieval(combined, 5, title="Vestidos blancos")

# test QUANTITAU
def Kmean_statistics(kmeans, imatges, kmax):
    i = []
    WCD_r = []
    temps = []
    for k in range(2, kmax+1):
        kmeans.k = k
        start = t.time()
        kmeans.fit()
        end = t.time()
        WCD = kmeans.withinClassDistance()
        
        i.append(k)
        temps.append((end - start))
        WCD_r.append(WCD)
    
    return WCD_r, i, temps




def get_shape_accuracy(etiquetes, gt):
    total = len(etiquetes)
    et_corr = sum(etiquetes == gt) #con for va un poco más lento
    percentatge = (et_corr / total)*100
    
    return percentatge


def get_color_accuracy(etiquetes, gt):
    total = len(etiquetes)

    correctes = 0
    
    for i in range(total):
        for j in range()
        
    
    
    return 1






if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    #test per a les Funcions  d'analisi qualitatiu
    test_Retrieval_by_color()
    test_Retrieval_by_shape()
    test_Retrieval_combined()
    #Posible problema, que tarda casi 3s en pasar.
