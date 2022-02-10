# %%
import mytrain as mt
from importlib import reload
import numpy as np

# %%
reload(mt)
# read data
X_train, y_train, X_val, y_val, X_test, y_test = mt.get_CIFAR10_data()


# %%
reload(mt)
# extract features(by pca, hog or,,)
X_train, X_val, X_test = mt.extract_features(X_train, X_val, X_test, method='pca')


# %%
reload(mt)
# train and validate
best_net, best_parameter, parameter_list = mt.train_and_validate([X_train, y_train, X_val, y_val, X_test, y_test])
print('best_parameter: \n ', best_parameter)
print('best_parameter: %.6f\n ' % (best_parameter['val_acc']*100))
# Print your test accuracy: this should be above 48%
test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: %.6f' % (test_acc*100))

best_net.predict(X_val).sum()/len(X_val)
best_net.predict(X_val)

# %%
val_sum = np.sum(best_net.predict(X_val) == y_val)
test_sum = np.sum(best_net.predict(X_test) == y_test)

b = best_net.predict(X_val) == best_net.predict(X_test)