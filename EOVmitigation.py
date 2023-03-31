import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def implicit_pca(self,
                     damage_existence: np.ndarray,
                     eovs: np.ndarray,
                     train_proportion: float,
                     n_pcs: int,
                     discarded_pcs: np.ndarray,
                     save_results: bool):

        """
            EOV Mitigation using Implicit PCA.
            This method discards a subset of Principal Components from the overall. 
            The main reason is that EOPs are more significantly present in the first PCs.
            Parameters
            ----------
            psd : 3D array
                Collection of Power Spectral Density arrays.
            damage_existence : list
                0 equals undamaged conditions and 1 equals damaged conditions.
            n_pcs : int
                Number of Principal Components extracted from the Power Spectral Density array.
            discarded_pcs : list
                List of PCs discarded

            Returns
            ----------
            damage_index: list
                List of damage indexes for each PSD.
        """

        data = self.value

        eovs = eovs.T

        #Flatten Power Spectral Density matrices if more than one accelerometer is used.
        data_flatten = data.reshape(-1,np.shape(data)[2])
        data_flatten = data_flatten.T

        #Create Pandas Data Frame from Power Spectral Density arrays
        freq_vector = self.freq_vector
        lines = self.lines

        names = []
        
        for line in range(lines):
            for freq in freq_vector:
                names.append('Line:'+str(line)+'_Freq:'+str(freq))

        #Split dataset between undamaged and damaged observations
        data_u = data_flatten[(damage_existence == 0),:]
        data_d = data_flatten[(damage_existence == 1),:]
        eovs_u = eovs[(damage_existence == 0)]
        eovs_d = eovs[(damage_existence == 1)]

        data_train, data_u_test, eovs_train, eovs_u_test = train_test_split(data_u,
                                                                            eovs_u, 
                                                                            train_size = train_proportion, 
                                                                            random_state = 42,
                                                                            stratify = eovs_u)

        data_test = np.vstack((data_u_test, data_d))
        # eovs_test = np.vstack((eovs_u_test, eovs_d))

        dataset = np.vstack((data_train, data_test))

        #Use training data to establish normalization
        scaler = MinMaxScaler()
        scaler.fit(data_train)

        #Normalize data
        data_train_scaled = scaler.transform(data_train)
        data_test_scaled = scaler.transform(data_test)
        dataset_scaled = scaler.transform(dataset)

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