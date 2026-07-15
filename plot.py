import numpy as np
import matplotlib.pyplot as plt

fpr = np.load("saved_models/fake_news_fpr_sgd.npy")
tpr = np.load("saved_models/fake_news_tpr_sgd.npy")

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (sgd)")
plt.show()