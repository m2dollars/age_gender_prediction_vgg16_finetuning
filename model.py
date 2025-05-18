# import tensorflow as tf
# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.layers import Dropout, Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.models import Model

# def build_model(input_shape=(128, 128, 1)):
#     inputs = Input(input_shape)
#     x = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
#     x = BatchNormalization(axis=3)(x)
#     x = MaxPooling2D((3, 3))(x)
    
#     x = Conv2D(128, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2), strides = (2, 2))(x)
    
#     x = Conv2D(256, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
    
#     x = Flatten()(x)
#     dense_1 = Dense(256, activation='relu')(x)
#     dense_2 = Dense(256, activation='relu')(x)
#     dense_3 = Dense(128, activation='relu')(dense_2) 
    
#     dropout_1 = Dropout(0.4)(dense_1)
#     dropout_2 = Dropout(0.4)(dense_3)
    
#     output_gender = Dense(1, activation= 'sigmoid', name='gender_output')(dropout_1)
#     output_age = Dense(1, activation='linear', name='age_output')(dropout_2)
    
#     model = Model(inputs=[inputs], outputs=[output_gender, output_age])
#     return model

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(128, 128, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    age_output = Dense(1, activation='relu', name='age_output')(x)
    
    model = Model(inputs=base_model.input, outputs=[gender_output, age_output])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss={'gender_output': 'binary_crossentropy', 'age_output': 'mae'},
                  metrics={'gender_output': 'accuracy', 'age_output': 'mae'})

    return model


    