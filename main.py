from layers import Dense

X = [[1.5, 2, 3.7],
     [4.3, 5, 2.1],
     [8, 3.2, 5.6]]

dense1 = Dense(3, 3, activation='relu')
dense1.forward(X)
print(dense1.outputs)

