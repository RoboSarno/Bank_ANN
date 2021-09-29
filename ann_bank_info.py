import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
       
class ANN:
    def __init__(self):
        self.X = 0
        self.y = 0
        self.dataset = 0
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0
        self.ann = None
        self.le = None
        self.sc = None
        
    # Getters 
    def print_Dataset(self):
        st.write(self.dataset)
        
    def test_Individual(self, individual):
        print(individual)
        ans = self.ann.predict(self.sc.transform([individual])) > 0.5
        print(ans)
        
    # Setters
    def read_CSV(self):
        self.dataset = pd.read_csv('Churn_Modelling.csv')
        self.X = self.dataset.iloc[:, 3:-1].values
        self.y = self.dataset.iloc[:, -1].values
        
    def fix_Data(self):
        
        self.le = LabelEncoder()
        self.X[:, 2] = self.le.fit_transform(self.X[:, 2])
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        self.X = np.array(ct.fit_transform(self.X))
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
            
    def make_ANN(self):
        self.ann = tf.keras.models.Sequential()
        
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        self.ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        self.ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        self.ann.fit(self.X_train, self.y_train, batch_size = 32, epochs = 100)
        

st.write("""
        # Leave or Stay
        """)
st.write("""
        **Information:** Fake randomly generated bank information from some random bank in Europe.\n
        **Problem:** Regarding the information below, based on a unquie customer or an example customer, predict weather this individual will leave or stay.
        """)
st.write("""  
        ##### Data Layout:   
        **Geography:** str() | **Credit Score:** int()| **Gender:** int(0, 0, 0) | **Age:** int() | **Tenure:** int() | **Balance:** float() | **Number of Products:** int() | **Does this customer have a credit card:** bool() | **Is this customer an Active Member:** bool() | **Estimated Salary:** int()
        """)

nn = ANN()
nn.read_CSV()
nn.print_Dataset()
nn.fix_Data()


st.sidebar.write("""
            #### What is the unique customer that you want to predict will leave or stay?
            """)

geog_Choices = {1: "France", 2: "Germany", 3: "Spain"}
gender_Choices = {1: "Male", 2: "Female", 3: "Other"}
yes_no_Choices = {1: "Yes", 2: "No"}

def geog_Choices_Func(option):
    return geog_Choices[option]

def gender_Choices_Func(option):
    return gender_Choices[option]

def yes_no_Choices_Func(option):
    return yes_no_Choices[option]
    

geog_sel = st.sidebar.selectbox("Geography option", options=list(geog_Choices.keys()), format_func=geog_Choices_Func)
cred_score_sel = st.sidebar.slider("Credit Score:", 300, 850)
gender_sel = st.sidebar.selectbox("Gender option", options=list(gender_Choices.keys()), format_func=gender_Choices_Func)
age_sel = st.sidebar.text_input("Age:")
tenure_sel = st.sidebar.slider("Tenure:", 0, 10)
balance_sel = st.sidebar.text_input("Balance (ex. 100.0):")
num_prod_sel = st.sidebar.slider("Number of Products:", 0, 4)
hav_cc_sel = st.sidebar.selectbox("Does this customer have a credit card?", options=list(yes_no_Choices.keys()), format_func=yes_no_Choices_Func)
is_active_sel = st.sidebar.selectbox("Is this customer an Active Member?", options=list(yes_no_Choices.keys()), format_func=yes_no_Choices_Func)
est_sal_sel = st.sidebar.text_input("Estimated Salary:")

if geog_sel == 'France':
    geog_sel = [1,0,0]
    
elif geog_sel == 'Germany':
    geog_sel = [0,1,0]
    
else:
    geog_sel = [0,0,1]

if gender_sel >= 2 or hav_cc_sel == 2 or is_active_sel == 2:
    gender = 0
    have_cc = 0
    is_active = 0

geog_sel = np.array(geog_sel)
test_individual = np.array([cred_score_sel, gender_sel, age_sel, tenure_sel, balance_sel, num_prod_sel, hav_cc_sel, is_active_sel, est_sal_sel])
test_individual = np.hstack([geog_sel, test_individual])



st.write("""
         ##### The individual you specified will: 
         """)
predict = st.sidebar.button("Predict")

if predict:
    nn.make_ANN()
    stay_or_leave = nn.test_Individual(test_individual)
    if stay_or_leave:
        st.write("""
            ### Stay
            """)
    else:
            st.write("""
            ### Leave
            """)
    
    
    




        



    
    





