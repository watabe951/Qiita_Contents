import csv
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

length = 10000
alpha_2 = 0.25
sigma_2 = 0.2



x = []
y = []
x.append(rd.normal(0, np.sqrt(alpha_2)))
y.append(x[0] + rd.normal(0, np.sqrt(sigma_2)))
for t in range(length):
    # 1階差分トレンドを適用
    #Predictor Step

    x.append(x[t] + rd.normal(0, np.sqrt(alpha_2)))
    y.append(x[t+1] + rd.normal(0, np.sqrt(sigma_2)))
    # x[t+1, i] = x_resampled[t, i] + v # システムノイズの付加
    # sigma_2 += float(t) * 0.5/ float(length) 
x = np.array(x)
y = np.array(y)

print(x)
print(y)

def draw_graph():
    # グラフ描画
    T = len(y)
    
    plt.figure(figsize=(16,8))
    plt.plot(range(T), y,"r")
    plt.plot(range(T), x, "g")
    plt.show()
    # for t in range(T):
    #     plt.scatter(np.ones(self.n_particle)*t, self.x[t][:,0], color="r", s=2, alpha=0.1)
    
    # plt.title("sigma^2={0}, alpha^2={1}, log likelihood={2:.3f}".format(self.sigma_2, 
    #                                                                             self.alpha_2, 
    #                                                                             self.log_likelihood))
np.savetxt("noisy_obs.txt", y)
np.savetxt("noisy_truth.txt", x)

draw_graph()
    