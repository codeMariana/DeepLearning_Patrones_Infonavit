## FILE TO STORE FUNCTIONS USED FOR DATA CLEANING


"------------------------------------------------------------------------------"
#############
## Imports ##
#############

## Python libraries
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)


import re
import unicodedata
import plotly.express as px
import numpy as np
import seaborn as sns
import probscale
from scipy import stats
import sys, os
from pandas_profiling import ProfileReport
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, OneHotEncoder,KBinsDiscretizer
from sklearn.impute import SimpleImputer




"------------------------------------------------------------------------------"
###########################
## Function's parameters ##
###########################

#cat_vars=['estrato_nuevo', 'region','region_txt', 'empresas', 'p1_2', 'p1_4', 'p1_5','p2_7', 'p2_8','p2_9', 'p2_10',
 #'p2_11', 'p2_12', 'p2_13', 'p2_14','p2_15','p2_16', 'p2_17','p2_18','p2_19_a', 'p2_19_b','p2_19_c','p2_19_d','p2_19_e',
 #'p2_19_f', 'p2_19_g','p2_19_h','p2_19_i','p2_19_j','p2_19_k','p2_19_l', 'p2_19_m','p2_19_n','p2_19_o','p2_20_a','p2_20_b',
 #'p2_20_c', 'p2_20_d', 'p2_25', 'p2_27_a','p2_27_b', 'p2_27_c','p2_28', 'p2_29','p2_31','p2_32','p2_33','p2_34_a',
 #'p2_34_b','p2_34_c','p2_34_d','p2_34_e','p2_35_a', 'p2_35_b', 'p2_35_c', 'p2_35_d', 'p2_35_e', 'p2_35_f','p2_36',
 #'p2_39','p2_40','p2_41_a','p2_41_b', 'p2_41_c','p2_41_d', 'p2_41_e','p2_41_f','p2_41_g','p2_42_a','p2_42_b','p2_42_c',
 #'p2_42_d','p2_42_e','p2_42_f','p2_42_g', 'p2_44_a','p2_44_b','p2_44_c','p2_44_d','p2_44_e', 'p2_44_f', 'p2_45_a','p2_45_b',
 #'p2_45_c','p2_45_d','p2_45_e','p2_45_f', 'p2_46', 'p2_47_a','p2_47_b','p2_47_c','p2_47_d', 'p3_1', 'p3_3','p3_5',
 #'p3_8','p3_9','p3_10', 'p3_11','p3_12', 'p3_13', 'p4_1','p4_2','p4_3', 'p5_1','p5_2', 'p5_3', 'p5_4','p5_5','p6_1',
 #'p6_3','levantamiento']

cat_vars=['estrato_nuevo', 'region', 'empresas', 'p1_2', 'p1_4', 'p1_5','p2_7', 'p2_8','p2_9', 'p2_10','p2_11', 'p2_12',
          'p2_13', 'p2_14','p2_15','p2_16', 'p2_17','p2_18','p2_19', 'p2_20', 'inversion_total', '2_22_autorizados',
          'p2_25', 'p2_27_bis', '2_27','p2_28', 'p2_29','p2_31','p2_32','p2_33','p2_34_a','p2_34_b','p2_34_c','p2_34_d',
          'p2_34_e','p2_35_a', 'p2_35_b', 'p2_35_c', 'p2_35_d', 'p2_35_e', 'p2_35_f','p2_36','p2_39','p2_40',
          'p2_41_imp','p2_41_contr','p2_41_none','p2_41_imp2','p2_41_contr2','p2_41_contr98','p2_41_imp98', 'p2_41_contr99',
          'p2_41_imp99','p2_41_both','2_41_g', 'p2_42_imp', 'p2_42_contr','p2_42_all', 'p2_42_none','p2_42_contr98',
          'p2_42_imp98','p2_42_contr99','p2_42_imp99','2_42_g2', 'p2_44_imp', 'p2_44_contr', 'p2_44_none','p2_44_both',
          'p2_44_contr98', 'p2_44_imp98', 'p2_44_imp99','p2_44_contr99', 'p2_45_imp','p2_45_contr','p2_45_none','p2_45_both',
          'p2_45_contr98', 'p2_45_imp98', 'p2_45_imp99', 'p2_46', 'p2_47_a','p2_47_b','p2_47_c','p2_47_d', 'p3_1',
          'p3_3','p3_5', 'p3_8','p3_9','p3_10', 'p3_11','p3_12', 'p3_13', 'p4_1','p4_2','p4_3', 'p5_1','p5_2', 'p5_3',
          'p5_4','p5_5','p6_1', 'p6_3','levantamiento', 'rfclimpio']

vars_to_cat=['p1_2', 'p1_4', 'p1_5','p2_7', 'p2_8','p2_9', 'p2_10','p2_11', 'p2_12',
          'p2_13', 'p2_14','p2_15','p2_16', 'p2_17','p2_18','p2_19', 'p2_20', 'inversion_total', '2_22_autorizados',
          'p2_25', 'p2_27_bis', '2_27','p2_28', 'p2_29','p2_31','p2_32','p2_33','p2_34_a','p2_34_b','p2_34_c','p2_34_d',
          'p2_34_e','p2_35_a', 'p2_35_b', 'p2_35_c', 'p2_35_d', 'p2_35_e', 'p2_35_f','p2_36','p2_39','p2_40',
          'p2_41_imp','p2_41_contr','p2_41_none','p2_41_imp2','p2_41_contr2','p2_41_contr98','p2_41_imp98', 'p2_41_contr99',
          'p2_41_imp99','p2_41_both','2_41_g', 'p2_42_imp', 'p2_42_contr','p2_42_all', 'p2_42_none','p2_42_contr98',
          'p2_42_imp98','p2_42_contr99','p2_42_imp99','2_42_g2', 'p2_44_imp', 'p2_44_contr', 'p2_44_none','p2_44_both',
          'p2_44_contr98', 'p2_44_imp98', 'p2_44_imp99','p2_44_contr99', 'p2_45_imp','p2_45_contr','p2_45_none','p2_45_both',
          'p2_45_contr98', 'p2_45_imp98', 'p2_45_imp99','p2_45_contr99' , 'p2_46', 'p2_47_a','p2_47_b','p2_47_c','p2_47_d', 'p3_1',
          'p3_3','p3_5', 'p3_8','p3_9','p3_10', 'p3_11','p3_12', 'p3_13', 'p4_1','p4_2','p4_3', 'p5_1','p5_2', 'p5_3',
          'p5_4','p5_5','p6_1', 'p6_3']

"------------------------------------------------------------------------------"
###############
## Functions ##
###############

def count_unique_obs(data):
    """
    Counting number of unique observations for all variables
        args:
        data (dataframe): data that is being analyzed
        returns:
        (series): number of unique observations for all variables
    """
    return data.nunique()



def proporcion(listaVar,n):
    """
    Calculate the data proportion of categorical variables.
        args:
            listaVar (Serie): Serie with unique values of categorical variables
                               to get use value_counts() into a Serie
            n (int): value of total observation of data set.
        returns:
           newList(list): List with name, count and proportion of each category.
    """
    newList = []
    for lis in listaVar.iteritems():
        newList.append([lis[0],lis[1],"{}%".format(round(100*(lis[1]/n),1))])
    return newList



def data_profiling_categ(data, cat_vars):
    """
    Create the data profiling of categorical variables.
        args:
            data (Data Frame): data set into Dataframe.
            cat_vars (list): list with categorical variables names.
        returns:
           display(): display the Dataframes with info.
    """

    for val in cat_vars:
        print("*********************************")
        print("Variable Categorica {}".format(val))
        print("*********************************")

        catego  = data[val].value_counts()
        totalOb = len(data[val])
        can_Cat = len(catego)
        moda    = data[val].mode().values[0]
        valFal  = data[val].isnull().sum()
        top1    = [catego[0:1].index[0],catego[0:1].values[0]] if can_Cat >= 1 else 0
        top2    = [catego[1:2].index[0],catego[1:2].values[0]] if can_Cat >= 2 else 0
        top3    = [catego[2:3].index[0],catego[2:3].values[0]] if can_Cat >= 3 else 0

        elemVarCat = {
            "Info":val,
            "Num_Registros":[totalOb],
            "Num_de_categorias":[can_Cat],
            "Moda":[moda],
            "Valores_faltantes":[valFal],
            "Top1":[top1],
            "Top2":[top2],
            "Top3":[top3]
            }

        #primerdataframe
        df_catVar = pd.DataFrame(elemVarCat).set_index("Info").T

        #mostrar primer data frame
        print(display(df_catVar))

        print("Valores de las categorias y sus proporciones")
        #segundodataframe donde se muestra los valores de las categorias su cantidad y su proporción.
        pro = proporcion(catego,totalOb)
        dfProp = pd.DataFrame(pro,columns=['Categoría', 'Observaciones', 'proporción']).set_index("Categoría")
        #mostrar primer data frame
        print(display(dfProp))
        print("\n\n".format())
    return




## Data profiling for numeric variables
def data_profiling_numeric(data, num_vars):
    """
    Data profiling for numeric variables
        Args:
            data(dataframe): dataframe that will be analyzed.
        num_vars (list): list of variables' names in the dataframe that will be analyzed.
        Retruns:
            Dataframe with the data profiling (type, number of observations, mean, sd, quartiles, max, min, unique observations, top 5 repeated observations, number of null variables)
            of the choosen numeric variables.
    """

    ## Copy of initial dataframe to select only numerical columns
    dfx = data.loc[:, num_vars]


    ## Pipeline to create dataframe with general data description
    print("*********************************")
    print("** General description of data **")
    print("*********************************")

    #### List where the resulting dataframes will be stored for further concatenation
    res_dfs = []

    #### Type of numeric variables
    dfx_dtype = dfx.dtypes.to_frame().T
    dfx_dtype.index = ["dtype"]
    res_dfs.append(dfx_dtype)

    #### Counting unique variables
    dfx_uniqvars = dfx.nunique().to_frame().T
    dfx_uniqvars.index = ["count_unique"]
    res_dfs.append(dfx_uniqvars)

    #### Counting missing values
    dfx_missing = dfx.isnull().sum().to_frame().T
    dfx_missing.index = ["missing_v"]
    res_dfs.append(dfx_missing)

    #### General description of the data and addition of min values
    dfx_desc = dfx.describe()
    dfx_desc.loc["min", :] = dfx.min(axis=0)
    res_dfs.append(dfx_desc)

    #### Concatenating resulting dataframes into one final result
    print(display(pd.concat(res_dfs, axis=0)))
    print("-"*75)
    print("-"*75)
    print("\n\n".format())


    ## Pipeline to obtain top repeated variables per column
    print("****************************")
    print("** Top repeated variables **")
    print("****************************")

    #### Initial variables
    tops = 5 #### Number of tops that will be selected
    i = 0 #### Counter to start joining dataframes

    #### Loop through all variables that will be processed
    for col_sel in dfx:

        #### Creating dataframe with top entries and count
        dfxx = dfx[col_sel].value_counts().iloc[:tops].to_frame()
        dfxx.reset_index(drop=False, inplace=True)
        dfxx["part"] = round(dfxx[col_sel]/dfx[col_sel].count()*100, 2)
        dfxx.columns = pd.MultiIndex.from_tuples([(col_sel, tag) for tag in ["value", "count", "part_notnull"]])

        #### Joining all the variables in one final dataframe
        if i == 0:
            df_tops = dfxx
            i += 1
        else:
            df_tops = df_tops.join(dfxx)

    ## Fill empty spaces of resulting dataframe and renaming index entries
    df_tops.fillna("-", inplace=True)
    df_tops.index = ["top_" + str(i) for i in range(1, df_tops.shape[0] + 1)]
    print(display(df_tops))
    print("-"*75)
    print("-"*75)
    print()
    return



def convert_to_num(df, vars_to_cat):
    """
    Transform columns' names to standard format (lowercase, no spaces, no points)
        args:
            dataframe (dataframe): df whose columns will be formatted.
        returns:
            dataframe (dataframe): df with transformed columns .
    """

    for i in vars_to_cat:
        df[i] = df[i].fillna(0).astype(int)
        #df[i] = df[i].astype(int)
        #df[i] = df[i].astype('Int64')
        #df.loc[df[i].notnull(), i] = df.loc[df[i].notnull(), i].apply(int)
        #sub2['income'].fillna((-50000, inplace=True)
        #sub2['income'].fillna((sub2['income'].mean()), inplace=True)
        #sub2['income'].fillna((-50000, inplace=True)
        # df[i] = df[i].apply(np.int64)
        # df[i] = np.floor(pd.to_numeric(df[i], errors='coerce')).astype('Int64')
        # df[i] = df[i].astype(float).astype('Int64')
        # df[i] = df[i].astype(pd.Int64Dtype())
        # df[i] = pd.to_numeric(df[i],errors='coerce').astype(pd.Int64Dtype())
        # df[df[i].notnull()] = df[df[i].notnull()].astype(int)

        return df



def convert_object(dataframe, vars_to_cat):
    """
    Transform columns' names to standard format (lowercase, no spaces, no points)
        args:
            dataframe (dataframe): df whose columns will be formatted.
        returns:
            dataframe (dataframe): df with transformed columns .
    """

    for i in vars_to_cat:
        #data["type"] = data["type"].astype('category').cat.codes


        dataframe[i] = dataframe[i].astype(object)
        #dataframe[i] = dataframe[i].astype('category', copy=False)
        return dataframe




