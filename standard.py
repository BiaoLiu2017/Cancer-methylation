from sklearn.preprocessing import StandardScaler
import numpy as np
data = np.loadtxt("final_top12_marker_450K_train_8620.txt", delimiter="\t")
data_val = np.loadtxt("final_top12_marker_450K_validation_2158.txt", delimiter="\t")
data_test = np.loadtxt("final_top12_marker_450K_test_3055.txt", delimiter="\t")
scaler = StandardScaler()
scaler.fit(data)
x = scaler.transform(data)
val = scaler.transform(data_val)
test = scaler.transform(data_test)
mean = scaler.mean_
var = scaler.var_
np.savetxt("final_top12_marker_450K_train_8620_scaler.txt", x, fmt='%.3f')
np.savetxt("final_top12_marker_450K_validation_2158_scaler.txt", val, fmt='%.3f')
np.savetxt("final_top12_marker_450K_test_3055_scaler.txt", test, fmt='%.3f')
np.savetxt("mean_of_train_450K_8620.txt", mean)
np.savetxt("var_of_train_450K_8620.txt", var)
