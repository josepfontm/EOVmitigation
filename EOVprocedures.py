import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

def implicit_pca(train_dsf: np.ndarray,
                 test_dsf: np.ndarray,
                 damage_existence: np.ndarray,
                 discarded_pcs: np.ndarray,
                 print_results: bool = True,
                 save_results: bool = True):

        """
            EOV Mitigation using Implicit PCA.
            This method discards a subset of Principal Components from the overall. 
            The main reason is that EOPs are more significantly present in the first PCs.
            Parameters
            ----------
            train_dsf : np.array
                Train data (Damage-sensitive features) with EOV Influence
            test_dsf : np.array
                Test data (Damage-sensitive features) with EOV Influence
            damage_existence : 1D array (rows should correspond to different observations)
                0 equals undamaged conditions and 1 equals damaged conditions.
            discarded_pcs : list
                List of PCs discarded
            print_results : bool
                Option to print results from EOV Mitigation Procedure on terminal.
            save_results : bool
                Option to save results from EOV Mitigation Procedure in a .csv file

            Returns
            ----------
            damage_index: list
                List of damage indexes for each PSD.
        """

        #Discard PCs which have high correlation with EOVs.

        print(np.shape(train_dsf))
        print(np.shape(test_dsf))

        train_dsf_corrected = np.delete(train_dsf, discarded_pcs, 1)
        test_dsf_corrected = np.delete(test_dsf, discarded_pcs, 1)

        print(np.shape(train_dsf_corrected))
        print(np.shape(test_dsf_corrected))

        dataset_dsf_corrected = np.vstack((train_dsf_corrected, test_dsf_corrected))

        #Mahalanobis Distance
        sigma = np.linalg.inv(np.cov(train_dsf_corrected.T))

        print(np.shape(sigma))

        di_array = []

        for observation in range(np.shape(dataset_dsf_corrected)[0]):
            d = dataset_dsf_corrected[observation].reshape(1,-1)@sigma@dataset_dsf_corrected[observation].reshape(-1,1)

            if di_array == []:
                di_array = d
            else:
                di_array = np.append(di_array,d)

        if save_results == True:
            np.savetxt("di_implicit_pca.csv",di_array,delimiter=",")

        #Calculate F1 Score
        di_train = di_array[0:len(train_dsf_corrected)]

        threshold = np.percentile(di_train,q=95)

        print(threshold)

        print(np.shape(di_array))

        y_pred = di_array > threshold
        y_pred = y_pred.astype(int)

        print(np.shape(y_pred))

        u = 0
        fa = 0
        ud = 0
        d = 0

        print(np.shape(y_pred))
        print(np.shape(damage_existence))

        for i in range(np.shape(dataset_dsf_corrected)[0]):
            if y_pred[i] == 1 and damage_existence[i] == 1:
                u = u + 1
            elif y_pred[i] == 1 and damage_existence[i] == 0:
                fa = fa + 1
            elif y_pred[i] == 0 and damage_existence[i] == 1:
                ud = ud + 1
            elif y_pred[i] == 0 and damage_existence[i] == 0:
                d = d + 1

        print('Results') 
        print(u)
        print(fa)
        print(ud)
        print(d)
        print(u+fa+ud+d)

        print(np.shape(damage_existence))
        print(y_pred)

        np.savetxt("y_pred.csv",y_pred,delimiter=",")
        np.savetxt("di_array.csv",di_array,delimiter=",")

        f1 = f1_score(damage_existence, y_pred)

        if print_results == True:
            print('IMPLICIT PCA')
            print('---DATA---')
            print('How many PCs are used? ' + str(np.shape(train_dsf)[1]))
            print('How many PCs have been discarded? ' + str(len(discarded_pcs)))

            print('---PREDICTIONS---')
            print('Undamaged:' + str(u))
            print('False Alarm:' + str(fa))
            print('Unnoticed Damagae:' + str(ud))
            print('Damage:' + str(d))

            print('---PERFORMANCE---')
            print('F1 Score: ' + str(f1))

        return di_array,f1,threshold

def explicit_pca_reg(data: np.ndarray,
                        damage_existence: np.ndarray,
                        eovs: np.ndarray,
                        train_proportion: float,
                        n_pcs: int,
                        discarded_pcs: np.ndarray,
                        max_order: int,
                        print_results: bool = False,
                        save_results: bool = True,
                        plot_results: bool = False):

    """
        EOV Mitigation using Explicit PCA.
        This method uses EOVs measured to generate polynomial regression models.
        Parameters
        ----------
        psd : 3D array
            Collection of Power Spectral Density arrays.
        damage_existence : 1D array (rows should correspond to different observations)
            0 equals undamaged conditions and 1 equals damaged conditions.
        eovs : 1D or 2D array (rows should correspond to different observations)
            Information regarding the environmental and/or operational conditions of a given observation.
        train_proportion : float
            Value between 0 and 1, which splits data between training and test.
        n_pcs : int
            Number of Principal Components extracted from the Power Spectral Density array.
        discarded_pcs : list
            List of PCs discarded
        max_order : int
            Highest order for the Polynomial Regression model to try.
        print_results : bool
            Option to print results from EOV Mitigation Procedure on terminal.
        save_results : bool
            Option to save results from EOV Mitigation Procedure in a .csv file
        plot_results : bool
            Plots helpful to explain the results

        Returns
        ----------
        damage_index: list
            List of damage indexes for each PSD.
    """

    data = self.value
    labels = np.hstack((eovs,damage_existence))

    #Flatten Power Spectral Density matrices if more than one accelerometer is used.
    data_flatten = data.reshape(-1,np.shape(data)[2])
    data_flatten = data_flatten.T

    data_u = []
    data_d = []
    labels_u = []
    labels_d = []

    #Split dataset between undamaged and damaged observations
    print('--------------FOR------------')
    for i in range(np.shape(labels)[0]):
        if damage_existence[i] == 1: #Undamaged
            if data_u == []:
                data_u = data_flatten[i,:]
                labels_u = labels[i]
            else:
                data_u = np.vstack((data_u, data_flatten[i,:]))
                labels_u = np.vstack((labels_u, labels[i,:]))
        elif damage_existence[i] == 0: #Damaged
            if data_d == []:
                data_d = data_flatten[i,:]
                labels_d = labels[i]
            else:
                data_d = np.vstack((data_d, data_flatten[i,:]))
                labels_d = np.vstack((labels_d, labels[i,:]))

    data_train, data_u_test, labels_train, labels_u_test = train_test_split(data_u,
                                                                            labels_u, 
                                                                            train_size = train_proportion, 
                                                                            random_state = 42,
                                                                            stratify = labels_u[:,:-1])

    data_test = np.vstack((data_u_test, data_d))
    labels_test = np.vstack((labels_u_test, labels_d))

    dataset = np.vstack((data_train, data_test))
    labels_dataset = np.vstack((labels_train, labels_test))

    #Use training data to establish normalization
    scaler = StandardScaler()
    scaler.fit(data_train)

    #Normalize data
    data_train_scaled = scaler.transform(data_train)
    data_test_scaled = scaler.transform(data_test)
    data_validation_scaled = scaler.transform(data_u_test)

    #Apply Principal Components Analysis (PCA)
    pca = PCA(n_components=n_pcs)
    #Use training data to fit PCA, then apply to transform both datasets
    #Training and Testing
    pca.fit(data_train_scaled)

    train_pca = pca.transform(data_train_scaled)
    test_pca = pca.transform(data_test_scaled)
    validation_pca = pca.transform(data_validation_scaled) #Used only to validate results from regression model
    dataset_pca = np.vstack((train_pca, test_pca))

    #Discard PCs which have high correlation with EOVs.
    train_pca = np.delete(train_pca, discarded_pcs, 1)
    test_pca = np.delete(test_pca, discarded_pcs, 1)
    validation_pca = np.delete(validation_pca, discarded_pcs, 1)
    dataset_pca = np.delete(dataset_pca, discarded_pcs, 1)

    data_train  = []
    dataset = []

    for eov in range(np.shape(eovs)[1]): #Do not iterate through discarded PCs
        for pca in range(np.shape(train_pca)[1]):

            mse_results = []
            order_regression = []

            X_train = labels_train[:,eov].reshape(-1,1)
            Y_train = train_pca[:,pca].reshape(-1,1)

            X_validation = labels_u_test[:,eov].reshape(-1,1)
            Y_validation = validation_pca[:,pca].reshape(-1,1)

            X_dataset = labels_dataset[:,eov].reshape(-1,1)
            Y_dataset = dataset_pca[:,pca].reshape(-1,1)

            for order in range(max_order):
                order = order + 1

                poly_features = PolynomialFeatures(degree=order, include_bias=False)
                X_train_poly = poly_features.fit_transform(X_train)
                X_validation_poly = poly_features.transform(X_validation)

                reg = LinearRegression()

                reg = reg.fit(X_train_poly, Y_train)

                X_vals = np.linspace(np.min(X_train), np.max(X_train),100).reshape(-1,1)
                X_vals_poly = poly_features.transform(X_vals)
                Y_vals = reg.predict(X_vals_poly)

                Y_train_pred = reg.predict(X_train_poly)
                Y_validation_pred = reg.predict(X_validation_poly)

                mse = mean_squared_error(Y_train, Y_train_pred)

                mse_results.append(mse)
                order_regression.append(order)

            best_order = min(mse_results)
            index = mse_results.index(best_order)

            poly_features = PolynomialFeatures(degree=order_regression[index], include_bias = False)
            X_train_poly = poly_features.fit_transform(X_train)

            reg = LinearRegression()
            reg = reg.fit(X_train_poly, Y_train)

            X_dataset_poly = poly_features.transform(X_dataset)
            Y_dataset_pred = reg.predict(X_dataset_poly)

            corrected_train = Y_train - Y_train_pred
            corrected_dataset = Y_dataset - Y_dataset_pred

            if data_train == []:
                data_train = corrected_train
                dataset = corrected_dataset
            else:
                data_train = np.hstack((data_train, corrected_train))
                dataset = np.hstack((dataset, corrected_dataset))

    #Mahalanobis Distance
    sigma = np.linalg.inv(np.cov(data_train.T))
    di_array = []

    for observation in range(np.shape(dataset_pca)[0]):
        d = dataset[observation].reshape(1,-1) @ sigma @ dataset[observation].reshape(-1,1)

        if di_array == []:
            di_array = d
        else:
            di_array = np.append(di_array,d)

    if save_results == True:
        np.savetxt("di_explicit_pca_reg_npcs_" + str(n_pcs) + ".csv",di_array,delimiter=",")

    #Calculate F1 Score
    di_train = di_array[0:len(train_pca)]

    threshold = np.percentile(di_train,q=95)

    y_pred = di_array < threshold
    y_pred = y_pred.astype(int)

    u = 0
    fa = 0
    ud = 0
    d = 0

    damage_existence = labels_dataset[:,-1]

    for i in range(np.shape(dataset_pca)[0]):
        if y_pred[i] == 1 and damage_existence[i] == 1:
            u = u + 1
        elif y_pred[i] == 1 and damage_existence[i] == 0:
            ud = ud + 1
        elif y_pred[i] == 0 and damage_existence[i] == 1:
            fa = fa + 1
        elif y_pred[i] == 0 and damage_existence[i] == 0:
            d = d + 1

    f1 = f1_score(damage_existence, y_pred)

    if print_results == True:
        print('EXPLICIT PCA REGRESSION')
        print('---DATA---')
        print('How many PCs are used? ' + str(n_pcs))
        print('How many PCs have been discarded? ' + str(len(discarded_pcs)))
        print('Maximum order used in Polynomial Regression Models? ' + str(max_order))

        print('---PREDICTIONS---')
        print('Undamaged:' + str(u))
        print('False Alarm:' + str(fa))
        print('Unnoticed Damagae:' + str(ud))
        print('Damage:' + str(d))

        print('---PERFORMANCE---')
        print('F1 Score: ' + str(f1))

    if plot_results ==True:
        pymodal.plot_control_chart(di_array,di_train,threshold,colors,"Explicit PCA Regression",n_pcs)

    return di_array,f1,threshold,y_pred
    
def pc_informed_reg(data: np.ndarray,
                    damage_existence: np.ndarray,
                    eovs: np.ndarray,
                    train_proportion: float,
                    n_pcs: int,
                    eov_sensitive_pcs: np.ndarray,
                    max_order: int,
                    print_results: bool = False,
                    save_results: bool = True,
                    plot_results: bool = False):

    """
        EOV Mitigation using Principal Component Informed Regression.
        Method proposed in ECCOMAS SMART 2023.
        J.Font-Moré, L.D. Avendano-Valencia, D. Garcia-Cava, M.A. Pérez, "Interpreting
        environmental variability from damage sensitive features"
        X ECCOMAS Thematic Conference on Smart Structures and Materials (SMART 2023)

        In this publication, we proposed a method that uses the so-called EOV-Sensitive PCs 
        as a surrogate of the Environmental and Operational variables driving 
        the non-stanionary behaviour in the DSFs. Hence, a regression model 
        using EOV-Sensitive PCs as predictors and remaining PCs as explained variables.

        Parameters
        ----------
        psd : 3D array
            Collection of Power Spectral Density arrays.
        damage_existence : 1D array (rows should correspond to different observations)
            0 equals undamaged conditions and 1 equals damaged conditions.
        eovs : 1D or 2D array (rows should correspond to different observations)
            Information regarding the environmental and/or operational conditions of a given observation.
        train_proportion : float
            Value between 0 and 1, which splits data between training and test.
        n_pcs : int
            Number of Principal Components extracted from the Power Spectral Density array.
        discarded_pcs : list
            List of PCs discarded
        max_order : int
            Highest order for the Polynomial Regression model to try.
        print_results : bool
            Option to print results from EOV Mitigation Procedure on terminal.
        save_results : bool
            Option to save results from EOV Mitigation Procedure in a .csv file
        plot_results : bool
            Plots helpful to explain the results

        Returns
        ----------
        damage_index: list
            List of damage indexes for each PSD.
    """

    data = self.value
    labels = np.hstack((eovs,damage_existence))

    #Flatten Power Spectral Density matrices if more than one accelerometer is used.
    data_flatten = data.reshape(-1,np.shape(data)[2])
    data_flatten = data_flatten.T

    data_u = []
    data_d = []
    labels_u = []
    labels_d = []

    #Split dataset between undamaged and damaged observations
    print('--------------FOR------------')
    for i in range(np.shape(labels)[0]):
        if damage_existence[i] == 1: #Undamaged
            if data_u == []:
                data_u = data_flatten[i,:]
                labels_u = labels[i]
            else:
                data_u = np.vstack((data_u, data_flatten[i,:]))
                labels_u = np.vstack((labels_u, labels[i,:]))
        elif damage_existence[i] == 0: #Damaged
            if data_d == []:
                data_d = data_flatten[i,:]
                labels_d = labels[i]
            else:
                data_d = np.vstack((data_d, data_flatten[i,:]))
                labels_d = np.vstack((labels_d, labels[i,:]))

    data_train, data_u_test, labels_train, labels_u_test = train_test_split(data_u,
                                                                            labels_u, 
                                                                            train_size = train_proportion, 
                                                                            random_state = 42,
                                                                            stratify = labels_u[:,:-1])

    data_test = np.vstack((data_u_test, data_d))
    labels_test = np.vstack((labels_u_test, labels_d))

    dataset = np.vstack((data_train, data_test))
    labels_dataset = np.vstack((labels_train, labels_test))

    #Use training data to establish normalization
    scaler = StandardScaler() 
    scaler.fit(data_train)

    #Normalize data
    data_train_scaled = scaler.transform(data_train)
    data_test_scaled = scaler.transform(data_test)
    data_validation_scaled = scaler.transform(data_u_test)

    #Apply Principal Components Analysis (PCA)
    pca = PCA(n_components=n_pcs)
    #Use training data to fit PCA, then apply to transform both datasets
    #Training and Testing
    pca.fit(data_train_scaled)

    train_pca = pca.transform(data_train_scaled)
    test_pca = pca.transform(data_test_scaled)
    validation_pca = pca.transform(data_validation_scaled) #Used only to validate results from regression model
    dataset_pca = np.vstack((train_pca, test_pca))

    data_train  = []
    dataset = []

    for eov in eov_sensitive_pcs: 
        for pca in range(len(eov_sensitive_pcs),np.shape(train_pca)[1]):
            mse_results = []
            order_regression = []

            X_train = train_pca[:,eov].reshape(-1,1)
            Y_train = train_pca[:,pca].reshape(-1,1)

            X_validation = validation_pca[:,eov].reshape(-1,1)
            Y_validation = validation_pca[:,pca].reshape(-1,1)

            X_dataset = dataset_pca[:,eov].reshape(-1,1)
            Y_dataset = dataset_pca[:,pca].reshape(-1,1)

            for order in range(max_order):
                order = order + 1

                poly_features = PolynomialFeatures(degree=order, include_bias=False)
                X_train_poly = poly_features.fit_transform(X_train)
                X_validation_poly = poly_features.transform(X_validation)

                reg = LinearRegression()

                reg = reg.fit(X_train_poly, Y_train)

                X_vals = np.linspace(np.min(X_train), np.max(X_train),100).reshape(-1,1)
                X_vals_poly = poly_features.transform(X_vals)
                Y_vals = reg.predict(X_vals_poly)

                Y_train_pred = reg.predict(X_train_poly)
                Y_validation_pred = reg.predict(X_validation_poly)

                mse = mean_squared_error(Y_train, Y_train_pred)

                mse_results.append(mse)
                order_regression.append(order)

            best_order = min(mse_results)
            index = mse_results.index(best_order)

            poly_features = PolynomialFeatures(degree=order_regression[index], include_bias = False)
            X_train_poly = poly_features.fit_transform(X_train)

            reg = LinearRegression()
            reg = reg.fit(X_train_poly, Y_train)

            X_dataset_poly = poly_features.transform(X_dataset)
            Y_dataset_pred = reg.predict(X_dataset_poly)

            corrected_train = Y_train - Y_train_pred
            corrected_dataset = Y_dataset - Y_dataset_pred

            if data_train == []:
                data_train = corrected_train
                dataset = corrected_dataset
            else:
                data_train = np.hstack((data_train, corrected_train))
                dataset = np.hstack((dataset, corrected_dataset))

    #Mahalanobis Distance
    sigma = np.linalg.inv(np.cov(data_train.T))
    di_array = []

    for observation in range(np.shape(dataset_pca)[0]):
        d = dataset[observation].reshape(1,-1) @ sigma @ dataset[observation].reshape(-1,1)

        if di_array == []:
            di_array = d
        else:
            di_array = np.append(di_array,d)

    if save_results == True:
        np.savetxt("di_explicit_pca_reg_npcs_" + str(n_pcs) + ".csv",di_array,delimiter=",")

    #Calculate F1 Score
    di_train = di_array[0:len(train_pca)]

    threshold = np.percentile(di_train,q=95)

    y_pred = di_array < threshold
    y_pred = y_pred.astype(int)

    u = 0
    ud = 0
    fa = 0
    d = 0

    damage_existence = labels_dataset[:,-1]

    for i in range(np.shape(dataset_pca)[0]):
        if y_pred[i] == 1 and damage_existence[i] == 1:
            u = u + 1
        elif y_pred[i] == 1 and damage_existence[i] == 0:
            ud = ud + 1
        elif y_pred[i] == 0 and damage_existence[i] == 1:
            fa = fa + 1
        elif y_pred[i] == 0 and damage_existence[i] == 0:
            d = d + 1

    f1 = f1_score(damage_existence, y_pred)

    if print_results == True:
        print('PC-INFORMED REGRESSION')
        print('---DATA---')
        print('How many PCs are used? ' + str(n_pcs))
        print('How many PCs have been used as surrogate variable? ' + str(len(eov_sensitive_pcs)))
        print('Maximum order used in Polynomial Regression Models? ' + str(max_order))

        print('---PREDICTIONS---')
        print('Undamaged:' + str(u))
        print('False Alarm:' + str(fa))
        print('Unnoticed Damagae:' + str(ud))
        print('Damage:' + str(d))

        print('---PERFORMANCE---')
        print('F1 Score: ' + str(f1))

    if plot_results ==True:
        pymodal.plot_control_chart(di_array,di_train,threshold,colors,"PC-Informed Regression",n_pcs)

    return di_array,f1,threshold,y_pred
