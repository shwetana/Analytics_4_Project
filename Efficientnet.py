import numpy as np
import tensorflow as tf
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
warnings.filterwarnings("ignore")

dataset='casting product image data for quality inspection/casting_data'

dataset_path = os.listdir(dataset)

print (dataset_path)  #what kinds of classes are in this dataset

print("Types of classes labels found: ", len(dataset_path))

###================ building class labels ===========================

class_labels = []

for item in dataset_path:
 # Get all the file names
 all_classes = os.listdir(dataset + '/' +item)
 #print(all_classes)

 # Add them to the list
 for room in all_classes:
    class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))
    # print(class_labels[:5])


#=========Building a dataframe with labels and path to image

# Build a dataframe
df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
# print(df.head())
# print(df.tail())

#=========== resizing image==================================

# from cv2 import cv2
from PIL import Image

path = 'casting product image data for quality inspection/casting_data/'
dataset_path = os.listdir(dataset)

im_size = 224 #300-efficientnetb3

images = []
labels = []

for i in dataset_path :
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames :
        # img = cv2.imread(data_path + '/' + f)
        # img = cv2.resize(img , (im_size , im_size))
        img = Image.open(data_path + '/' + f)
        img = img.resize((im_size,im_size))
        images.append(np.array(img))
        labels.append(i)


#This model takes input images of shape (224, 224, 3), and the input data should range [0, 255].

images = np.array(images)
print(images.shape)
# print(type(images))
images = images.astype('float32') / 255.0
# print(images.shape) # (7204, 224, 224, 3)

#================== encoder ========================

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
y=df['Labels'].values
print("=====================================================")
print(y)

y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print("=====================================================")
print (y)

y=y.reshape(-1,1)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
print(Y[:5])
print(Y[35:])


## =================== split train test dataset ==============

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


images, Y = shuffle(images, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

#inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


##==== efficient net implementation ====

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50 , EfficientNetB7, EfficientNetB3

NUM_CLASSES = 2
IMG_SIZE = 224 #300 - efficientnetb3
size = (IMG_SIZE, IMG_SIZE)


inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))


## Using model without transfer learning

outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
# outputs = EfficientNetB3(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
# outputs = ResNet50(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )

model.summary()



hist = model.fit(train_x, train_y, epochs=30, verbose=2)



###====== transfer learning from pretrained weights ================

# def build_model(num_classes):
#     inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
#     #x = img_augmentation(inputs)
#     x = inputs
#     # model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
#     # model = ResNet50(include_top=False, input_tensor=x, weights="imagenet")
#     # model = EfficientNetB7(include_top=False, input_tensor=x, weights="imagenet")
#     model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
#
#     # Freeze the pretrained weights
#     model.trainable = False
#
#     # Rebuild top
#     x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
#     x = layers.BatchNormalization()(x)
#
#     top_dropout_rate = 0.2
#     x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
#     outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
#
#     # Compile
#     model = tf.keras.Model(inputs, outputs, name="EfficientNet")
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
#     model.compile(
#         optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
#     )
#     return model
#
# model = build_model(num_classes=NUM_CLASSES)

#
# # Cuda and cudnn is installed for this tensorflow version. So we can see GPU is enabled
tf.config.experimental.list_physical_devices()
with tf.device('/GPU:0'):
    gpu_performance =model.fit(train_x, train_y, epochs=30,callbacks=ModelCheckpoint(
        'EffNet7_Casting_defects.hdf5',
        save_best_only=True,
        monitor='val_loss'), verbose=2)
    print(gpu_performance)
    model.save('D:\\Analytics_4_Project_Image_Classification\\models\\EffNetb0.h5')

from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error
from sklearn.metrics import classification_report, plot_confusion_matrix,confusion_matrix,f1_score


def evaluate_model(model, model_name,y_pred):
    # y_pred = model.predict(test_x)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_y, y_pred)
    model_acc = model.score(train_x, train_y)
    recall = recall_score(y_true=test_y , y_pred=y_pred)
    precision = precision_score(y_true=test_y , y_pred=y_pred)
    mse = mean_squared_error(y_true=test_y , y_pred=y_pred)
    # f1=f1_score(test_y,y_pred,)
    print(f'Model: {model_name}')
    print('-' * 50)
    print(f'Model Accuracy: {model_acc:.4f}, Testing Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}, Precision: {precision:.4f}, MSE: {mse:.4f}')
    print(f"accuracy of {model_name} is {accuracy * 100:0.2f}%")
    print('-' * 50)

# Load saved model
# best_model = load_model('./EffNet_Casting_Inspection.hdf5')
#
# preds = best_model.evaluate(test_x, test_y)
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))

# best_model=load_model('./ResNetfinal.h5')
best_model=load_model('D:\\Analytics_4_Project_Image_Classification\\models\\EffNetb0.h5')


preds = best_model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

evaluate_model(best_model,'EfficientNetB3',preds)