import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def visulise_scatter(params):
  sns.scatterplot(params[0],params[1])
  plt.show()