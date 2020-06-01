import numpy as np
import matplotlib.pyplot as plt

n = 10 # data points

# The following simple Python instructions define our x and y values (with 100 data points)
x = np.random.rand(n,1)
y = 5*x*x+np.random.randn(n,1) # y = 5x^2 + random noise


X = np.c_[np.ones((n,1)), x, x*x] # column wise array concatenation
print(X)

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print("y = " + str(beta[2,0]) + "*x^2 + " + str(beta[1,0]) + "*x + " + str(beta[0,0]))

nfit = 100
xplot = np.linspace(0.0,1.0, num=nfit)

Xplot = np.c_[np.ones((nfit,1)), xplot, xplot**2] # concatenate columns (as above)

ypredict = Xplot.dot(beta)

ytrue = 5*xplot*xplot

plt.plot(x, y ,'ro')
plt.plot(xplot, ytrue, label="$y_{\mathrm{Quadratic}}$")
plt.plot(xplot, ypredict, label="$y_{\mathrm{predict}}$")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Quadratic Regression')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression

clf2 = LinearRegression()
clf2.fit(X, y)
ysklearn = clf2.predict(Xplot)

print("ypredict = " + str(clf2.coef_[0, 2]) + "*x^2 + " + str(clf2.coef_[0, 1]) + "*x + " + str(clf2.coef_[0, 0]))
print("ysklearn = " + str(beta[2,0]) + "*x^2 + " + str(beta[1,0]) + "*x + " + str(beta[0,0]))
# note that the indices are reversed in the scikit-learn approach compared to what we did before:
# the shape is (1, n) instead of (n, 1)

plt.plot(x, y ,'ro')
plt.plot(xplot, ytrue, label="$y_{\mathrm{true}}$")
plt.plot(xplot, ypredict, label="$y_{\mathrm{predict}}$")
plt.plot(xplot, ysklearn, label="$y_{\mathrm{sklearn}}$")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Quadratic Regression')
plt.legend()
plt.show()

err_predict = abs(ypredict[:, 0] - ytrue)/abs(ytrue) # the predicted y's have shape (n, 1)
err_sklearn = abs(ysklearn[:, 0] - ytrue)/abs(ytrue)

plt.plot(xplot, err_predict, label="$\epsilon_{\mathrm{predict}}$")
plt.plot(xplot, err_sklearn, label="$\epsilon_{\mathrm{sklearn}}$")
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon_{\mathrm{rel}}$')
plt.axis([0, 1, 0, 2])
plt.title(r'Absolute relative error')
plt.legend()
plt.show()

plt.plot(xplot, abs(err_predict), label="$\epsilon_{\mathrm{predict}}$")
plt.plot(xplot, abs(err_sklearn), label="$\epsilon_{\mathrm{sklearn}}$")
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon_{\mathrm{rel}}$')
plt.axis([0, 1.0, 0, 0.02])
plt.title(r'Absolute relative error')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error


ypredict2 = X.dot(beta)
ysklearn2 = clf2.predict(X)

print("Mean squared error (ypredict):", mean_squared_error(y, ypredict2))
print("Mean squared error (ysklearn):", mean_squared_error(y, ysklearn2))



from sklearn.metrics import r2_score

print("R^2 score (ypredict):", r2_score(y, ypredict2))
print("R^2 score (ysklearn):", r2_score(y, ysklearn2))


