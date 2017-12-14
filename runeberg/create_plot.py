import matplotlib.pyplot as plt
import sys
import re
import numpy as np

def read_data(filename):
    acc=[]
    loss=[]
    val_acc=[]
    val_loss=[]
    with open(filename) as f:
        for line in f.readlines():
            if 'loss' in line:
                tmp=[float(i) for i in re.findall('(?<=\:) [0-9\.]+', line)]
                loss.append(tmp[0])
                acc.append(tmp[1])
                val_loss.append(tmp[2])
                val_acc.append(tmp[3])

    acc=np.array(acc)
    loss=np.array(loss)
    val_acc=np.array(val_acc)
    val_loss=np.array(val_loss)
    return acc, loss, val_acc, val_loss

nobn_acc, nobn_loss, nobn_val_acc, nobn_val_loss=read_data(sys.argv[1])
bn_acc, bn_loss, bn_val_acc, bn_val_loss=read_data(sys.argv[2])

plt.figure(1)
plt.title('Accuracy with and without batch normalization')

# Training acc
plt.plot(range(len(nobn_acc)), nobn_acc, 'g')
# Validation acc
plt.plot(range(len(nobn_val_acc)), nobn_val_acc, 'b')
# Training acc
plt.plot(range(len(bn_acc)), bn_acc, 'r')
# Validation acc
plt.plot(range(len(bn_val_acc)), bn_val_acc, 'm')
plt.legend(['Training acc (No BN)', 'Validation acc (No BN)','Training acc (BN)', 'Validation acc (BN)'], loc=4)
# loss plots
plt.figure(2)
plt.title('Loss with and without batch normalization')
# Training loss
print(nobn_loss)
plt.plot(range(len(nobn_loss)), nobn_loss, 'g')
# Validation loss
plt.plot(range(len(nobn_val_loss)), nobn_val_loss, 'b')
# Training loss
plt.plot(range(len(bn_loss)), bn_loss, 'r')
# Validation loss
plt.plot(range(len(bn_val_loss)), bn_val_loss, 'm')
plt.legend(['Training loss (No BN)', 'Validation loss (No BN)','Training loss (BN)', 'Validation loss (BN)'])
plt.show()


