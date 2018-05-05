## Achieving 99.8% accuracy on MNIST with BatchNormalization 

Model 2 performed best, as seen in mnist_ensemble2.ipynb. 

```
m2 = Sequential([
    BatchNormalization(input_shape=(28,28)),
    Convolution1D(32, (3), activation='relu', padding='same'),
    Dropout(0.1),
    MaxPooling1D(2),
    BatchNormalization(),
    Convolution1D(64, (3), activation='relu', padding='same'),
    Dropout(0.1),
    MaxPooling1D(2),
    Flatten(),
    BatchNormalization(),
    Dense(20, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')   
])

m2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
``` 


