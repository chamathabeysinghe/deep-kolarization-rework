import numpy as np
from keras.layers import Input
from keras.optimizers import Adam

from model.flowchroma_network import FlowChroma

enc_input = Input(batch_shape=(None, None, None, 1), name='encoder_input') # What is height what is width
incep_out = Input(batch_shape=(None, 1536), name='inception_input')
model = FlowChroma([enc_input, incep_out]).build()
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

X = [np.random.random((10,320,240,1)),np.random.random((10,1536))]
y= [np.random.random((10,320,240,2))]
model.fit(x=X, y=y, batch_size=5, epochs=10)




