import matplotlib.pyplot as plt

x_epoch = []
y_loss = []

with open ('log.txt') as f:
    for line in f.readlines():
        splited = line.split(',')
        if len(splited) > 1:
            epoch, loss = splited[0].split(':')[1].strip(), splited[1].split(':')[1].strip()
            x_epoch.append(int(epoch))
            y_loss.append(float(loss))


plt.plot(x_epoch, y_loss)
plt.xlabel('epoch')
plt.ylabel('rmse')
plt.grid()
plt.show()
