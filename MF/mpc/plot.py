import matplotlib.pyplot as plt

epochs = []
rmse = []
mae = []

with open('log.txt', 'r') as f:
    for line in f.readlines():
        splited = line.split(',')
        epochs.append(float(splited[0].split(':')[1].strip()))
        rmse.append(float(splited[1].split(':')[1].strip()))
        mae.append(float(splited[2].split(':')[1].strip()))


plt.plot(epochs, rmse, label='rmse loss')
plt.plot(epochs, mae, label='mae loss')
plt.legend(loc='best')
plt.grid()
plt.show()