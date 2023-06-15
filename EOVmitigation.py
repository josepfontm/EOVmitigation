import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def implicit_pca(self,
                     damage_existence: np.ndarray,
                     eovs: np.ndarray,
                     train_proportion: float,
                     n_pcs: int,
                     discarded_pcs: np.ndarray,
                     print_results: bool = True,
                     save_results: bool = True):

        """
            EOV Mitigation using Implicit PCA.
            This method discards a subset of Principal Components from the overall. 
            The main reason is that EOPs are more significantly present in the first PCs.
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
            print_results : bool
                Option to print results from EOV Mitigation Procedure on terminal.
            save_results : bool
                Option to save results from EOV Mitigation Procedure in a .csv file

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

        np.savetxt("data_u.csv",data_u,delimiter=",")

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

        #Apply Principal Components Analysis (PCA)
        pca = PCA(n_components=n_pcs)
        #Use training data to fit PCA, then apply to transform both datasets
        #Training and Testing
        pca.fit(data_train_scaled)

        train_pca = pca.transform(data_train_scaled)
        test_pca = pca.transform(data_test_scaled)
        dataset_pca = np.vstack((train_pca, test_pca))

        #Discard PCs which have high correlation with EOVs.
        train_pca = np.delete(train_pca, discarded_pcs, 1)
        test_pca = np.delete(test_pca, discarded_pcs, 1)
        dataset_pca = np.delete(dataset_pca, discarded_pcs, 1)

        np.savetxt("dataset_pca.csv",dataset_pca,delimiter=",")

        #Mahalanobis Distance
        sigma = np.linalg.inv(np.cov(train_pca.T))

        di_array = []

        for observation in range(np.shape(dataset_pca)[0]):
            d = dataset_pca[observation].reshape(1,-1)@sigma@dataset_pca[observation].reshape(-1,1)

            if di_array == []:
                di_array = d
            else:
                di_array = np.append(di_array,d)

        if save_results == True:
            np.savetxt("di_implicit_pca_npcs_"+str(n_pcs)+".csv",di_array,delimiter=",")

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

        print(damage_existence)
        print(y_pred)

        np.savetxt("y_pred.csv",y_pred,delimiter=",")
        np.savetxt("di_array.csv",di_array,delimiter=",")
        np.savetxt("damage_existence.csv",damage_existence,delimiter=",")

        f1 = f1_score(damage_existence, y_pred)

        if print_results == True:
            print('IMPLICIT PCA')
            print('---DATA---')
            print('How many PCs are used? ' + str(n_pcs))
            print('How many PCs have been discarded? ' + str(len(discarded_pcs)))

            print('---PREDICTIONS---')
            print('Undamaged:' + str(u))
            print('False Alarm:' + str(fa))
            print('Unnoticed Damagae:' + str(ud))
            print('Damage:' + str(d))

            print('---PERFORMANCE---')
            print('F1 Score: ' + str(f1))

        return di_array,f1,threshold
