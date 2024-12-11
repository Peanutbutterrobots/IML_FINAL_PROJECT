import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from sklearn.decomposition import PCA
import addcopyfighandler

data = pd.read_csv("all_brands_combined_wo_img_2.csv")
data = data[data["Price"] != "Not Priced"].reset_index(drop=True)
data.dropna(subset="Mileage", inplace=True)

def get_onehot_brands(data):
	ohe = OneHotEncoder(handle_unknown="error")
	out = ohe.fit_transform(data["Brand"].to_numpy().reshape(-1, 1)).toarray().astype(int)
	return out

def get_onehot_models(data):
	ohe = OneHotEncoder(handle_unknown="error")
	out = ohe.fit_transform(data["Model"].to_numpy().reshape(-1, 1)).toarray().astype(int)
	return out

def get_years(data):
	return data["Year"].to_numpy().reshape(-1, 1).astype(int)

def get_ages(data):
	return (2024 - data["Year"]).to_numpy().reshape(-1, 1).astype(int)

def get_miles(data):
	return data["Mileage"].to_numpy().reshape(-1, 1).astype(int)

def get_onehot_colors(data):
	ohe = OneHotEncoder(handle_unknown="error")
	out = ohe.fit_transform(data["Exterior Color"].to_numpy().reshape(-1, 1)).toarray().astype(int)
	return out

def get_onehot_fuel(data):
	ohe = OneHotEncoder(handle_unknown="error")
	out = ohe.fit_transform(data["Fuel Type"].to_numpy().reshape(-1, 1)).toarray().astype(int)
	return out

def get_onehot_drivetrain(data):
	ohe = OneHotEncoder(handle_unknown="error")
	out = ohe.fit_transform(data["Drivetrain"].to_numpy().reshape(-1, 1)).toarray().astype(int)
	return out

def get_onehot_body_style(data):
	ohe = OneHotEncoder(handle_unknown="error")
	out = ohe.fit_transform(data["Bodystyle"].to_numpy().reshape(-1, 1)).toarray().astype(int)
	return out

def get_prices(data):
	return data["Price"].to_numpy().reshape(-1, 1).astype(int)


A = np.ones((data.shape[0], 9), dtype=int)
A[:,:7] = get_onehot_brands(data)
A[:,7] = get_years(data).flatten()
A[:,8] = get_miles(data).flatten()

A = np.concatenate((get_onehot_brands(data), 
                    get_onehot_models(data), 
                    get_ages(data), 
                    get_miles(data), 
                    np.power(get_years(data), 2), 
                    np.power(get_miles(data), 2), 
                    np.power(get_years(data), 3), 
                    np.power(get_miles(data), 3), 
                    get_onehot_body_style(data), 
                    get_onehot_drivetrain(data),
                    get_onehot_colors(data), 
                    get_onehot_fuel(data)), axis = 1)

print(A.shape[1], "total DOF\n")

b = get_prices(data).astype(float)

class LinearRegressor(nn.Module):
	def __init__(self, num_components, reg_coeff = 0):
		super().__init__()
		self.num_components = num_components
		self.reg_coeff = reg_coeff
		self.pca = None
		self.transform = lambda x: np.concatenate((self.pca.transform(x), np.ones((x.shape[0], 1))), axis = 1)
	
	def train_on(self, A, b):
		self.pca = PCA(n_components=self.num_components, svd_solver="full")

		A = self.pca.fit_transform(A)
		A = np.concatenate((A, np.ones((A.shape[0], 1))), axis = 1)

		self.w = np.linalg.inv(A.T@A + self.reg_coeff * np.eye(A.shape[1]))@(A.T)@b

	def forward(self, x):
		return (self.transform(x)@self.w).astype(float)

mse = lambda y, b: np.sum(np.power(y-b, 2))/(y.shape[0])
me = lambda y, b: np.sum(np.abs(y-b))/(y.shape[0])

def graphData(y, b):
	maxval = min([np.max(y), np.max(b)])
	minval = max([np.min(y), np.min(b)])
	plt.plot(b, y, ".")
	plt.xlabel("True car values")
	plt.ylabel("Predicted car values")
	plt.plot([minval, maxval], [minval, maxval], "k-")
	plt.show()

def graphAll(split, model):
	A_train, A_test, b_train, b_test, A, b = split
	y_train = model(A_train)
	y_test = model(A_test)
	y_all = model(A)
	fig, (train_plot, test_plot, all_plot) = plt.subplots(1, 3)
	fig.set_figwidth(14)
	fig.subplots_adjust(left=0.064, bottom=0.11, right=0.971, top=0.88, wspace=0.2, hspace=0.2)
	fig.suptitle("Model Prediction Quality")
	train_plot.plot(b_train, y_train, "b.", label="avg error: %.2f"%(me(y_train, b_train)))
	test_plot.plot(b_test, y_test, "b.", label="avg error: %.2f"%(me(y_test, b_test)))
	all_plot.plot(b, y_all, "b.", label="avg error: %.2f"%(me(y_all, b)))
	train_max = min([np.max(y_train), np.max(b_train)])
	train_min = max([np.min(y_train), np.min(b_train)])
	test_max = min([np.max(y_test), np.max(b_test)])
	test_min = max([np.min(y_test), np.min(b_test)])
	all_max = min([np.max(y_all), np.max(b)])
	all_min = max([np.min(y_all), np.min(b)])
	train_plot.plot([train_min, train_max], [train_min, train_max], "k-")
	test_plot.plot([test_min, test_max], [test_min, test_max], "k-")
	all_plot.plot([all_min, all_max], [all_min, all_max], "k-")
	train_plot.set_ylabel("Predicted car values")
	train_plot.set_xlabel("Real car values")
	test_plot.set_title("Testing Data")
	train_plot.set_title("Training Data")
	all_plot.set_title("All Data")
	test_plot.legend()
	train_plot.legend()
	all_plot.legend()
	plt.show()

def graphError(split, model):
	A_train, A_test, b_train, b_test, A, b = split
	y_train = model(A_train)
	y_test = model(A_test)
	y_all = model(A)
	fig, (train_plot, test_plot, all_plot) = plt.subplots(1, 3)
	fig.set_figwidth(14)
	fig.subplots_adjust(left=0.064, bottom=0.11, right=0.971, top=0.88, wspace=0.2, hspace=0.2)
	fig.suptitle("Error Graphs")
	bins = 25
	train_plot.hist(b_train - y_train, bins = bins, label="Var: %.2f"%(np.var(b_train - y_train)))
	test_plot.hist(b_test - y_test, bins = bins, label="Var: %.2f"%(np.var(b_test - y_test)))
	all_plot.hist(b - y_all, bins = bins, label="Var: %.2f"%(np.var(b - y_all)))
	train_plot.set_ylabel("Predicted car values")
	train_plot.set_xlabel("Real car values")
	test_plot.set_title("Testing Data")
	train_plot.set_title("Training Data")
	all_plot.set_title("All Data")
	test_plot.legend()
	train_plot.legend()
	all_plot.legend()
	plt.show()


from sklearn.model_selection import train_test_split

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)
split = (A_train, A_test, b_train, b_test, A, b)

############################################
#                                          #
############################################

############################################
#                EVALUATOR                 # 
############################################
model = LinearRegressor(206, 0.492)

model.train_on(A_train, b_train)

y_test = model(A_test)
y_train = model(A_train)
y_all = model(A)

print("Testing")
print("MSE: %.3f"%(mse(y_test, b_test)))
print("Mean Error: %.3f"%(me(y_test, b_test)))

print("\nTraining")
print("MSE: %.3f"%(mse(y_train, b_train)))
print("Mean Error: %.3f"%(me(y_train, b_train)))

print("\nAll")
print("MSE: %.3f"%(mse(y_all, b)))
print("Mean Error: %.3f"%(me(y_all, b)))

graphAll(split, model)
#graphError(split, model)
############################################


# ############################################
# #     BEST N_COMPONENTS TESTING GROUND     #
# ############################################
# reg_coeff = 0
# x = np.array(list(range(1, 238 + 1)))
# train_MSE = []
# test_MSE = []
# all_MSE = []
# train_MAE = []
# test_MAE = []
# all_MAE = []
# print("Reg coeff: %0.3f"%(reg_coeff))
# for i in x:
# 	model = LinearRegressor(i, 0.332)
# 	model.train_on(A_train, b_train)
# 	train_MSE.append(mse(model(A_train), b_train))
# 	test_MSE.append(mse(model(A_test), b_test))
# 	all_MSE.append(mse(model(A), b))
# 	train_MAE.append(me(model(A_train), b_train))
# 	test_MAE.append(me(model(A_test), b_test))
# 	all_MAE.append(me(model(A), b))

# print("MSE:")
# print("Best N_components (training data): %i (%i)"%(x[np.argmin(train_MSE)], np.min(train_MSE)))
# print("Best N_components (testing data): %i (%i)"%(x[np.argmin(test_MSE)], np.min(test_MSE)))
# print("Best N_components (all data): %i (%i)"%(x[np.argmin(all_MSE)], np.min(all_MSE)))

# print("\nMAE:")
# print("Best N_components (training data): %i (%i)"%(x[np.argmin(train_MAE)], np.min(train_MAE)))
# print("Best N_components (testing data): %i (%i)"%(x[np.argmin(test_MAE)], np.min(test_MAE)))
# print("Best N_components (all data): %i (%i)"%(x[np.argmin(all_MAE)], np.min(all_MAE)))


# fig, (MSE_plot, MAE_plot) = plt.subplots(1, 2)
# fig.suptitle("Model Prediction Quality")
# MSE_plot.plot(x, np.array(train_MSE), "b", label="Training Data")
# MSE_plot.plot(x, np.array(test_MSE), "g", label="Testing Data")
# MSE_plot.plot(x, np.array(all_MSE), "r", label="All Data")
# MAE_plot.plot(x, np.array(train_MAE), "b", label="Training Data")
# MAE_plot.plot(x, np.array(test_MAE), "g", label="Testing Data")
# MAE_plot.plot(x, np.array(all_MAE), "r", label="All Data")
# MSE_plot.set_ylabel("MSE")
# MAE_plot.set_ylabel("MAE")
# MSE_plot.set_xlabel("N Components")
# MAE_plot.set_xlabel("N Components")
# MAE_plot.set_title("MAE / N Components")
# MSE_plot.set_title("MSE / N Components")
# MAE_plot.legend()
# MSE_plot.legend()
# MSE_plot.set_yscale("log")
# MAE_plot.set_yscale("log")
# plt.show()
# ############################################


# ############################################
# #    BEST REGULARIZATION TESTING GROUND    #
# ############################################
# n_components = 238
# x = np.linspace(0.01, 1, 100)
# train = []
# test = []
# all = []
# print("N_components: %i"%(n_components))
# for i in x:
# 	model = LinearRegressor(n_components, i)
# 	model.train_on(A_train, b_train)
# 	train.append(mse(model(A_train), b_train))
# 	test.append(mse(model(A_test), b_test))
# 	all.append(mse(model(A), b))
# train = np.array(train)
# test = np.array(test)
# all = np.array(all)

# print("Best Regularization constant (testing data): %.3f"%(x[np.argmin(test)]))
# print("Best Regularization constant (all data): %.3f"%(x[np.argmin(all)]))

# plt.plot(x, test, "g", label="Testing Data")
# plt.plot(x, all, "r", label="All Data")
# plt.plot(x, train, "b", label="Training Data")
# plt.plot(x[np.argmin(test)], np.min(test), "g.")
# plt.plot(x[np.argmin(all)], np.min(all), "r.")
# plt.title("MSE Error / Regularization Constant")
# plt.xlabel("Regularization Constant (10^x)")
# plt.ylabel("MSE Error (log)")
# plt.yscale("log")
# plt.legend()
# plt.show()
# ############################################