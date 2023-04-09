__authors__ = '1636290, 1631153, 1636589'
__group__ = 'DJ.10'

import numpy as np #para usar numpy hay que poner np.<lo que queramos usar de la libreria>
import utils


class KMeans:

    def __init__(self, X, K=1, options=None): #de momento k=1 mas adelante  
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()


    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X): # esta bien hecho pero da error por el init_centroids
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        # te pasan una matriz,comruebas qye s una lista. si la matriz es 3d, la pasas a 2d
       
        X = X.astype(float) #Convierte los valores a tipo float

          # Si la matriz X tiene solo 1 o 2 dimensiones, no es necesario hacer nada
        if len(X.shape) > 2: # Si la matriz X tiene más de 2 dimensiones
                  
           if X.shape[-1] == 3: # Si la matriz X es una imagen de dimensiones F x C x 3, se aplanará a una matriz de dimensiones N x 3.
               X = X.reshape(-1, 3)
           
           else:# sino, aplanamos a una matriz de dimensiones N x D (D = última dimensión --> columnas)
               X = X.reshape(-1, X.shape[-1])
               
        self.X = X # Guarda la matriz X en la variable de instancia self.X
       

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
       
     """Initialization of centroids """

    
     # Inicializar centroides y centroides antiguos
     self.centroids = np.zeros((self.K, self.X.shape[1]))  # se crea una matriz de K filas y número de columnas igual al número de características de los datos de entrada
     self.old_centroids = np.zeros((self.K, self.X.shape[1]))  # se crea una matriz de K filas y número de columnas igual al número de características de los datos de entrada para almacenar los centroides antiguos

     # Asignar valores a los centroides en función de la opción de inicialización
     if self.options['km_init'].lower() == 'first':  # si la opción de inicialización es 'first'
        # Opción 'first': asignar los primeros K puntos de la imagen a los centroides
        unique_points = set()  # se crea un conjunto vacío para almacenar los puntos únicos
        for i in range(self.K):  # para cada uno de los K centroides
            # Encontrar el primer punto único y asignarlo al i-ésimo centroide
            j = 0  # se inicializa un índice para recorrer los puntos de entrada
            while tuple(self.X[j]) in unique_points:  # mientras el punto ya haya sido asignado a otro centroide
                j += 1  # se pasa al siguiente punto
            self.centroids[i] = self.X[j]  # se asigna el primer punto único encontrado al i-ésimo centroide
            unique_points.add(tuple(self.X[j]))  # se agrega el punto encontrado al conjunto de puntos únicos

     elif self.options['km_init'].lower() == 'random':  # si la opción de inicialización es 'random'
        # Opción 'random': elegir K puntos aleatorios de la imagen como centroides
        self.centroids = self.X[np.random.choice(self.X.shape[0], self.K, replace=False)]  # se seleccionan K puntos aleatorios sin reemplazo de los datos de entrada para asignarlos como centroides

     elif self.options['km_init'].lower() == 'custom':  # si la opción de inicialización es 'custom'
        # Opción 'custom': utilizar una estrategia de inicialización definida por el usuario
        # En este caso, inicializamos los centroides distribuyendo uniformemente los puntos a lo largo de cada dimensión de característica
        for i in range(self.K):  # para cada uno de los K centroides
            self.centroids[i, :] = np.linspace(np.min(self.X[:, i]), np.max(self.X[:, i]), self.K)  # se distribuyen uniformemente los puntos a lo largo de cada dimensión de característica para asignarlos como centroides

     # Inicializar centroides antiguos con los mismos valores que los centroides
     self.old_centroids = self.centroids.copy()  # se copian los centroides asignados como nuevos en la variable de los centroides antiguos



    def get_labels(self): 
        """       
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # self.labels = np.random.randint(self.K, size=self.X.shape[0])
        dist = distance(self.X, self.centroids) #claculamos las distancias a los clusters
        self.labels = np.argmin(dist, axis=1)  # pedimos que de cada fila (axis=1) de la matriz dist, nos guarde la columna donde se encuentra el valor mas pequeño
        # axis = 1 --> de cada fila 


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids.copy()
        for i in range(self.K):
            centre = self.X[self.labels == i]
            if len(centre) > 0:
                self.centroids[i] = np.sum(centre, axis=0) / len(centre)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        for k in range(self.K):
            if not np.array_equal(self.centroids[k], self.old_centroids[k]): #compara si son diferentes los arrays
                return False #es diferente

        return True


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        condicio = False
        while not condicio and (self.num_iter < self.options['max_iter']):
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            condicio = self.converges()


    def withinClassDistance(self):
        #Puede fallar
        WCD = 0.0
        N = self.X.shape[0]
        
        for k in range(self.K):
            x = self.X[self.labels == k]
            Cx = self.centroids[k]
            distancia = distance(x, Cx.reshape(1,-1))
            WCD = WCD + np.sum(np.square(distancia))
        
        return (WCD / N)

        """           
        returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################


    def find_bestK(self, max_K):
        ant_WCD = 0
        bestK = None
        llindar = 20
        
        for k in range(2, max_K):
            self.K = k
            self.fit()
            WCD = self.withinClassDistance()
            
            if ant_WCD != 0:
                dec = 100 * (WCD / ant_WCD)
                if (100 - dec) < llindar:
                    bestK = k
                    break  
              
            ant_WCD = WCD

        if bestK is None:
            self.K = max_K
        else:
            self.K = bestK
        
        return bestK
        
        
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################



def distance(X, C):
    dist = np.sqrt(np.sum(np.square(X[:, np.newaxis] - C), axis=2))
    return dist

    # https://www.w3docs.com/snippets/python/python-numpy-valueerror-operands-could-not-be-broadcast-together-with-shapes.html

    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)
    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)
    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    labels = [] #crea la lista
    for i in np.argmax(utils.get_color_prob(centroids), axis=1): #por cada fila se guarda la columna con el valor mas grande
        labels.append(utils.colors[i]) #se añade un color a la lista
    return labels

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

