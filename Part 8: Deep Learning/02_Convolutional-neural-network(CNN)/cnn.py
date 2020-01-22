# Convolutional Neural Network


# Part 1 - Building libraries for the CNN

# Substep 1: Importing the Keras libraries and packages
from keras.models import Sequential      # To init. NN through seq. of layers
from keras.layers import Convolution2D   # To add convo;utional layers
from keras.layers import MaxPooling2D    # To add pooling layers
from keras.layers import Flatten         # Converging maxpooling to flattening
from keras.layers import Dense           # To add fully connected layers

# Substep 2: Initialising the CNN
classifier = Sequential()

# Step 2: Making CNN layesrs
# Substep 1 - Convolution: Create "feature maps" to obtain 1st convolutional
            # layer by applying feature detector.
            # e.g;  I/P image ⨂ feature detector = feature map
            # Here, ⨂ > N-Ary Circled Times Operator.
classifier.add(
        Convolution2D(32,    # Total filteres/kernels/feature detectors 
                      3, 3,  # Total row & col dimention for filteres/kernel/feature detector
                      input_shape = (64, 64, 3),   # Convert all images into same size/format 
                      activation = 'relu'))        # relu > rectifier function to resist from linearity


# Substep 2 - Pooling > Reducing size of feature map to get pooled feature map
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Total row & col dimention for pooled feature map

# Repeating substep 1 & 2. 
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Substep 3 - Flattening: Converging pixels into nodes
classifier.add(Flatten())

# Substep 4 - Full connection(hidden layer):  Adding hidden & output layer
classifier.add(Dense(output_dim = 128,     # No. of nodes hidden layer
                     activation = 'relu'))  # relu > rectifier function to resist from linearity
                                            # for hidden layer
classifier.add(Dense(output_dim = 1,      # One node for outcome for one class; either cat/dog
                     activation = 'sigmoid'))  # sigmoid function for 1 DV 
                                                # Softmax function: Actually sigmoid function but used for more than 1 DV.

                    
# Step 2(a): Compiling the CNN
classifier.compile(optimizer = 'adam',            # Type of Stochastic gradient algo to organizing of back propagation in most efficient way
                   metrics = ['accuracy'],        # For accuracy of result
                   loss = 'binary_crossentropy')  # O/P is 1 variable (binary)->  binary_crossentropy
                                                    # more than 2 O/P variables called -> catagorical_crossentropy 
                   

# Part 3: Fitting the CNN to the images
# Substep 1: Image Augmenteation > avoid overfitting, enrich dataset, gives good result in small dataset 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,  # fixing size
                                   shear_range = 0.2, # Geometrical transformation to fix pixel
                                   zoom_range = 0.2,  #  Zoom image
                                   horizontal_flip = True) # flip image horizontally

# Substep 2: Again rescale
test_datagen = ImageDataGenerator(rescale = 1./255)

# Substep 3: Creating training set 
training_set = train_datagen.flow_from_directory('dataset/training_set',   # location
                                                 target_size = (64, 64),   # Dimension size for row & col
                                                 batch_size = 32,          # Splitting dataset into 32
                                                 class_mode = 'binary')    # binary > 2 classes: cat\dog

# Substep 4: Creating test set
test_set = test_datagen.flow_from_directory('dataset/test_set',            # location
                                            target_size = (64, 64),        # Dimension size for row & col
                                            batch_size = 32,               # Splitting dataset into 32
                                            class_mode = 'binary')         # binary > 2 classes: cat\dog

# Substep 5: Training set 
classifier.fit_generator(training_set,               # training set
                         samples_per_epoch = 8000,   # no. of images in training set   
                         nb_epoch = 25,              # Process of propagation to loss function is called epoch
                         validation_data = test_set, # To evaluate performance on test dataset
                         nb_val_samples = 2000)      # No. of images in test dataset

