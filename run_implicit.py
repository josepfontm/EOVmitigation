#Import libraries
import pymodal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import EOVprocedures

def run_implicit(n_pcs):

    labels = pd.read_csv('data/labels.csv', header = None, names = ['measure']) #Labels
    temperature = pd.read_csv('data/temperature.csv', header = None, names = ['temperature']) #Temperature variable, in this case, there are no more EOVs
    damage_existence = pd.read_csv('data/damage_existence.csv', header = None, names = ['state']) #Undamage = 1, Damaged = 0
    damage_existence = damage_existence.to_numpy()

    #Load FRF object
    print('Load FRF')
    psd = pymodal.load_FRF('data/PSD_file.zip') #Load PSDs using PyModal library, other methods can be used.
    psd = psd.change_lines([0,1,2]) #Select which accelerometers are used. In this case, all 8 [0,1,2,3,4,5,6,7]
    psd_array = psd.value

    psd_array = psd_array.reshape(-1,np.shape(psd_array)[2]) #Reshape to line all PSDs values on a signle column (513,8) -> column-vector
    psd_array = psd_array.T #Transpose to have each measurement as a row

    freq_vector = psd.freq_vector #Use to name each and every one of the frequency lines 
    lines = psd.lines

    names = []

    #Build list of headers for variables FRF
    for line in range(lines):
        for freq in freq_vector:
            names.append('Line:'+str(line)+'_Freq:'+str(freq))

    df_X = pd.DataFrame(psd_array,columns=names) #Dataframe with dynamical data

    df_Y = pd.concat([labels,temperature], axis=1) #Dataframe with EOVs and labels

    df_Y['measure'] = df_Y['measure'].str.slice(0,6) #Labels are sliced 

    # --- TRAIN-TEST SPLIT ---
    print('TRAIN-TEST SPLIT')
    X_train = df_X[df_Y['measure']=='Case_R'] #Extract X from undamaged observations, which have Case_R label
    Y_train = df_Y[df_Y['measure']=='Case_R'] #Extract EOVs from undamaged observations, which have Case_R label

    X_test = df_X[df_Y['measure']!='Case_R'] #Extract X from damaged observations, which have label different from Case_R
    Y_test = df_Y[df_Y['measure']!='Case_R'] #Extract EOVs from damaged observations, which have label different from Case_R

    X_train,uX_test,Y_train, uY_test = train_test_split(X_train, 
                                                        Y_train, 
                                                        train_size = 0.8, 
                                                        random_state = 42,
                                                        stratify=Y_train['temperature'])

    # Training: 80% undamage observations
    # Testing: 20% remaining undamage data + 100% damage observations 

    X_test = pd.concat([uX_test,X_test],axis=0)
    Y_test = pd.concat([uY_test,Y_test],axis=0)

    X_train=X_train.to_numpy()
    Y_train=Y_train.to_numpy()

    X_test=X_test.to_numpy()
    Y_test=Y_test.to_numpy()

    #Normalize data before applying PCA
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- PRINCIPAL COMPONENT ANALYSIS ---
    pca = PCA(n_components=n_pcs)
    pca.fit(X_train_scaled)

    train_dsf = pca.transform(X_train_scaled)
    test_dsf = pca.transform(X_test_scaled)

    EOVprocedures.implicit_pca(train_dsf, 
                            test_dsf, 
                            damage_existence,
                            discarded_pcs=[0],
                            print_results=True,
                            save_results=True)