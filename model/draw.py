from os import access
import matplotlib
matplotlib.use("PDF")
import numpy as np
import matplotlib.pyplot as plt

def draw_line(data,label,path=None):
    x = [i for i in range(len(data[0]))]
    
    fit = plt.figure()
    print(label,data[0][299],data[1][299],data[2][299])
    plt.plot(x,data[0],label='gru')
    plt.plot(x,data[1],label='lstm')
    plt.plot(x,data[2],label='rnn')

    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig('./{}.png'.format(label))

    plt.close('all')


def draw(feature_index):
    loss = []
    f1 = []
    acc = []
    for label in ['gru','lstm','rnn']:
        loss.append(np.load('./result/{}_{}_loss.npy'.format(feature_index,label)))
        f1.append(np.load('./result/{}_{}_f1.npy'.format(feature_index,label)))
        acc.append(np.load('./result/{}_{}_acc.npy'.format(feature_index,label)))
    draw_line(f1,'feature{}_f1'.format(feature_index))
    draw_line(loss,'feature{}_loss'.format(feature_index))
    draw_line(acc,'feature{}_acc'.format(feature_index))
  
if __name__=='__main__':
    feature_num = [1,2,3]
    algo = ['rnn','gru','lstm']
    for f_n in feature_num:
        draw(feature_index=f_n)