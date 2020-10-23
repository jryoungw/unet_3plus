from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

input_size = (None,None,1)
inputs = Input(input_size)

def aggregate(l1, l2, l3, l4, l5):
    
    out = concatenate([l1, l2, l3, l4, l5], axis = -1)
    out =  Conv2D(320, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    
    return out

base_channel = 32

def unet3plus(inputs, conv_num = base_channel):
    
    XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE1)
    XE1_pool = MaxPooling2D(pool_size=(2, 2))(XE1)
    
    XE2 = Conv2D(conv_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE1_pool)
    XE2 = Conv2D(conv_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE2)
    XE2_pool = MaxPooling2D(pool_size=(2, 2))(XE2)
    
    XE3 = Conv2D(conv_num*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE2_pool)
    XE3 = Conv2D(conv_num*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE3)
    XE3_pool = MaxPooling2D(pool_size=(2, 2))(XE3)
    
    XE4 = Conv2D(conv_num*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE3_pool)
    XE4 = Conv2D(conv_num*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE4)
    XE4 = Dropout(0.5)(XE4)
    XE4_pool = MaxPooling2D(pool_size=(2, 2))(XE4)

    XE5 = Conv2D(conv_num*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE4_pool)
    XE5 = Conv2D(conv_num*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE5)
    XE5 = Dropout(0.5)(XE5)
    
    XD4_from_XE5 = UpSampling2D(size=(2,2), interpolation='bilinear')(XE5)
    XD4_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD4_from_XE5)
    XD4_from_XE4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE4)
    XD4_from_XE3 = MaxPooling2D(pool_size=(2,2))(XE3)
    XD4_from_XE3 = Conv2D(conv_num, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(XD4_from_XE3)
    XD4_from_XE2 = MaxPooling2D(pool_size=(4,4))(XE2)
    XD4_from_XE2 = Conv2D(conv_num, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(XD4_from_XE2)
    XD4_from_XE1 = MaxPooling2D(pool_size=(8,8))(XE1)
    XD4_from_XE1 = Conv2D(conv_num, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(XD4_from_XE1)
    XD4 = aggregate(XD4_from_XE5, XD4_from_XE4, XD4_from_XE3, XD4_from_XE2, XD4_from_XE1)
    
    XD3_from_XE5 = UpSampling2D(size=(4,4), interpolation='bilinear')(XE5)
    XD3_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XE5)
    XD3_from_XD4 = UpSampling2D(size=(2,2), interpolation='bilinear')(XD4)
    XD3_from_XD4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XD4)
    XD3_from_XE3 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE3)
    XD3_from_XE2 = MaxPooling2D(pool_size=(2,2))(XE2)
    XD3_from_XE2 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XE2)
    XD3_from_XE1 = MaxPooling2D(pool_size=(4,4))(XE1)
    XD3_from_XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XE1)
    XD3 = aggregate(XD3_from_XE5, XD3_from_XD4, XD3_from_XE3, XD3_from_XE2, XD3_from_XE1)
    
    XD2_from_XE5 = UpSampling2D(size=(8,8), interpolation='bilinear')(XE5)
    XD2_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XE5)
    XD2_from_XE4 = UpSampling2D(size=(4,4), interpolation='bilinear')(XE4)
    XD2_from_XE4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XE4)
    XD2_from_XD3 = UpSampling2D(size=(2,2), interpolation='bilinear')(XD3)
    XD2_from_XD3 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XD3)
    XD2_from_XE2 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE2)
    XD2_from_XE1 = MaxPooling2D(pool_size=(2,2))(XE1)
    XD2_from_XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XE1)
    XD2 = aggregate(XD2_from_XE5, XD2_from_XE4, XD2_from_XD3, XD2_from_XE2, XD2_from_XE1)
    
    XD1_from_XE5 = UpSampling2D(size=(16,16), interpolation='bilinear')(XE5)
    XD1_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XE5)
    XD1_from_XE4 = UpSampling2D(size=(8,8), interpolation='bilinear')(XE4)
    XD1_from_XE4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XE4)
    XD1_from_XE3 = UpSampling2D(size=(4,4), interpolation='bilinear')(XE3)
    XD1_from_XE3 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XE3)
    XD1_from_XD2 = UpSampling2D(size=(2,2), interpolation='bilinear')(XD2)
    XD1_from_XD2 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XD2)
    XD1_from_XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE1)
    XD1 = aggregate(XD1_from_XE5, XD1_from_XE4, XD1_from_XE3, XD1_from_XD2, XD1_from_XE1)
    
    out = Conv2D(conv_num*5, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(XD1)
    out = Conv2D(1, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(out)
    
    return out