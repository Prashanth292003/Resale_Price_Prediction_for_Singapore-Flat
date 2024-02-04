import streamlit as st
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd



st.set_page_config(page_title="ML", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: blue;'>SINGAPORE FLAT RESALE PRICE PREDICTION</h1>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["***HOME***&nbsp;&nbsp;&nbsp;&nbsp;",'***ABOUT-MODEL***&nbsp;&nbsp;&nbsp;&nbsp;','***PREDICTION***&nbsp;&nbsp;&nbsp;&nbsp;',"***ACCURACY***&nbsp;&nbsp;&nbsp;&nbsp;","***PREPROCESSING***&nbsp;&nbsp;&nbsp;&nbsp;","***DATA***&nbsp;&nbsp;&nbsp;&nbsp;","***CONTACT US***&nbsp;&nbsp;&nbsp;&nbsp;"])
  
path1 = 'C:/Users/Senthil/Desktop/DS/Projects Photo/singapore_map.jpg'
A = Image.open(path1)
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)
with tab1: 
  col1, col2 = st.columns(2)
  col1.markdown("""""""  """"""")
  col1.write("The project at hand seeks to address the complexities of the highly competitive resale flat market in Singapore by developing a robust solution for predicting resale prices. The motivation behind this initiative stems from the challenging task of accurately estimating the resale value of a flat, considering factors such as location, flat type, floor area, and lease duration. The ultimate goal is to provide users, including both potential buyers and sellers, with a reliable estimate through the integration of a machine learning model and a user-friendly web application.The project's scope encompasses several key tasks. Firstly, there is a need to collect and preprocess a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB), covering the years 1990 to the present. This involves cleaning and structuring the data for optimal use in machine learning. Subsequently, relevant features such as town, flat type, storey range, floor area, flat model, and lease commence date will be extracted. Additionally, the project involves creating supplementary features that may enhance prediction accuracy.The model selection process focuses on choosing an appropriate machine learning regression model, such as linear regression, decision trees, or random forests. The selected model will be trained on historical data, utilizing a portion of the dataset for this purpose. To assess the model's performance, evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score will be employed.The subsequent phase involves the development of a user-friendly web application using Streamlit. This application will enable users to input details of a flat, including town, flat type, storey range, etc. The machine learning model, trained on historical data, will then be leveraged to predict the resale price based on user inputs. The deployment of this Streamlit application on the Render platform will make it accessible to users over the internet.Thorough testing and validation are integral components of the project to ensure the correct functioning of the deployed application and the provision of accurate predictions. In essence, this comprehensive project amalgamates machine learning, web development, and cloud deployment to offer a valuable tool for navigating the intricacies of the competitive resale flat market in Singapore. The machine learning model's ability to capture the nuances of various factors influencing resale prices aims to empower users in making well-informed decisions. Looking forward, potential future enhancements may include continuous model retraining, real-time data updates, and the expansion of the feature set to encompass more granular details affecting resale prices.")
  col2.markdown("""""""  """"""")
  col2.markdown("""""""  """"""")
  col2.image(A,width = 700)
  
  
with open("S_Model_R.pkl","rb") as file:
    Prediction = pickle.load(file) 
    
P= "C:/Users/Senthil/Desktop/DS/Code/Singapore_flat_P/Final_data.csv"
data = pd.read_csv(P)
X = data[["year","town","floor_area_sqm","lease_commence_year"]]
Y = data[["resale_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
Ans = Prediction.predict(X_test)
Final = r2_score(y_test,Ans)

feature_importances = pd.Series(Prediction.feature_importances_,index = X_train.columns).sort_values(ascending=False)

with tab3:
    A = st.number_input('ENTER YEAR',min_value = 1,value = None,step = 1)
    B = st.number_input('ENTER TOWN',min_value = 0,value = None,step = 1)
    # C = st.number_input('ENTER ROOM_TYPE',min_value = 1,value = None,step = 1)
    D = st.number_input('ENTER FLOOR_AREA_sqm',min_value = 1,value = None)
    E = st.number_input('ENTER LEASE_COMMENCE_YEAR',value = None,step = 1)
    if A is not None and B is not None and D is not None and E is not None:
       if st.button("CLICK HERE TO PREDICT"):
         W = Prediction.predict([[A,B,D,E]]) 
        #  st.write(W)
         for i, v in enumerate(W):
           st.markdown(f'<h1 style="text-align: center; color:red;">PREDICTED RESALE PRICE: {v:.5f}</h1>', unsafe_allow_html=True)
    
    
path2 =  'C:/Users/Senthil/Desktop/DS/Projects Photo/ml1.jpg'
L = Image.open(path2)
with tab2:
  st.image(L,width=1300)
  st.markdown("[CLICK HERE TO KNOW PROBLEM STATEMENT OF THE PROJECT](https://docs.google.com/document/d/1mPb68zw8G-iFNcFr4hSAp7yIXc3-0JVlFVKBPf-0Hxo/edit)")
  st.subheader('How I built this model?')
  st.write("Embarking on the journey of constructing a predictive model for estimating resale prices of flats in Singapore, I commenced with the crucial task of data collection. Leveraging historical resale flat transactions from the Singapore Housing and Development Board (HDB) spanning from 1990 to the present, I meticulously gathered a comprehensive dataset to serve as the foundation for the predictive model.The initial phase of data preprocessing unfolded, where I undertook the intricate task of cleaning and structuring the dataset to prepare it for machine learning. This encompassed addressing missing values, handling outliers, and ensuring data integrity, laying the groundwork for subsequent feature extraction and engineering.Feature engineering became pivotal in capturing the nuances of resale price determinants. Key features such as town, flat type, storey range, floor area, flat model, and lease commence date were extracted from the dataset. Additionally, I introduced novel features to enhance prediction accuracy, recognizing the multifaceted nature of factors influencing resale prices.With the preprocessed and enriched dataset in hand, the model selection process ensued. Deliberate consideration led me to opt for the Random Forest Regressor, a versatile and powerful machine learning algorithm well-suited for regression tasks. This choice was driven by its ability to handle complex relationships in the data and mitigate overfitting concerns.The training phase involved exposing the chosen model to the historical data, allowing it to discern patterns and relationships. A portion of the dataset was reserved for training, ensuring the model's ability to generalize to unseen data, a crucial aspect in real-world predictions.As the model took shape, meticulous evaluation became imperative. Leveraging regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score, I rigorously assessed the predictive performance, aiming for a finely-tuned model capable of accurate resale price estimations.Transitioning from model development to real-world applicability, I delved into the creation of a user-friendly web application using Streamlit. This interface allowed users to input details of a flat, including town, flat type, storey range, etc. Seamlessly integrating the trained Random Forest Regressor, the application enabled users to obtain instant predictions on the resale price based on their inputs.To make this application accessible to a wider audience, I deployed it on the Render platform, ensuring internet accessibility and ease of use for potential buyers and sellers in the Singapore resale flat market. Rigorous testing and validation procedures were implemented to guarantee the functionality and accuracy of the deployed application, thereby providing a reliable tool for navigating the competitive and dynamic landscape of resale flat transactions in Singapore.")
  st.subheader('ABOUT Random Forest Regressor')
  st.write("Random Forest Regressor, a versatile ensemble learning algorithm, is celebrated for its effectiveness in both classification and regression tasks. Operating on the principles of bagging, it constructs an ensemble of decision trees during training, mitigating overfitting risks by leveraging multiple weaker learners. The algorithm introduces randomness by selecting subsets of both training data and features, fostering robustness and adaptability. With the ability to handle numerical and categorical features seamlessly, Random Forest excels in diverse datasets and exhibits resilience to noise and outliers.One of its strengths lies in feature importance evaluation, assigning scores to features based on their contribution to overall predictive performance. This feature is invaluable for guiding feature selection and offering insights into underlying data patterns. The algorithm's predictive power extends to capturing complex non-linear relationships, making it a powerful tool in various domains, from finance and healthcare to image recognition.Random Forest's ensemble nature, combining the predictions of multiple trees, leads to stable and generalized results. Furthermore, it allows for parallelization, enhancing computational efficiency, especially with large datasets. The algorithm's straightforward implementation and minimal hyperparameter tuning requirements contribute to its popularity in real-world applications. Leveraging out-of-bag samples for unbiased performance estimation and visualization of decision boundaries, Random Forest enhances model interpretability.In practice, Random Forest has demonstrated effectiveness in time-series data forecasting, ecological modeling, and applications like predicting stock prices and disease outcomes. Its scalability and robustness make it a suitable candidate for big data scenarios and tasks where interpretability is crucial. With applications ranging from sentiment analysis and customer churn prediction to climate modeling and predictive maintenance, Random Forest Regressor stands as a reliable and versatile tool in the machine learning landscape.")
  st.subheader("NOTE:Essential Essential information to be aware of before entering")
  st.write("""Before proceeding with the predictive analysis, it's imperative to familiarize yourself with the four input boxes integral to the model's functionality.In the first input box, you'll specify the target year for the resale prediction, indicating the year in which you plan to make the purchase.The second input box introduces the concept of town selection, and I will provide a list of towns shortly. To designate your preferred town, input '0' for the first town, '1' for the second town, and increment the number accordingly for subsequent choices.The third input box pertains to the floor area in square meters, where you'll input the specific floor area of the flat under consideration.Lastly, the fourth input box is dedicated to the lease commencement year, enabling you to input the year in which the lease commenced for the flat you are evaluating.This information is pivotal in the overall estimation process. In summary, these interactive input boxes offer a granular and personalized approach to resale price prediction, factoring in critical elements such as target year, town selection, floor area, and lease commencement year.(TOWN:'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN')""")
  if st.button('CLICK HERE TO KNOW ABOUT FEATURE IMPORTANCE'):
      M = pd.DataFrame(feature_importances,columns = ['COLUMN_IMPORTANCE'])
      M.index.name = 'COLUMNS_NAME'
      st.write(M)
  
with tab4:
  path3 = 'C:/Users/Senthil/Desktop/DS/Projects Photo/accuracy.png'
  O = Image.open(path3)
  width, height = O.size
  st.image(O, width=1300)
  st.subheader("For any regressor or classification model, what is accuracy, and why is it so important?")
  st.write("Accuracy in the context of regression or classification models is a performance metric that measures the proportion of correct predictions made by the model out of the total instances. It is a fundamental evaluation criterion, reflecting the model's ability to provide accurate and reliable results.Accuracy is crucial because it offers a clear and easily interpretable indication of how well the model performs. In classification tasks, it shows the percentage of correctly classified instances among the total predictions. For regression tasks, accuracy is often measured using metrics like R-squared or Mean Squared Error, providing insight into the precision of the model's predictions.High accuracy signifies that the model is effectively capturing the underlying patterns and relationships in the data, making it a valuable tool for decision-making and prediction. It instills confidence in the model's predictive capabilities, making it an essential metric for assessing and comparing different models. However, it's important to note that accuracy alone may not be sufficient in all scenarios, and other metrics such as precision, recall, F1 score, or specific regression metrics may be considered depending on the nature of the problem.")
  if st.button("CLICK HERE TO CHECK ACCURACY OF THE MODEL"):
      st.markdown(f'<h1 style="text-align: center; color:blue;">Accuracy Score: {Final:.5%}</h1>', unsafe_allow_html=True)
  

with tab5:
  path4 = 'C:/Users/Senthil/Desktop/DS/Projects Photo/preprocessing.jpg'
  U = Image.open(path4)
  # width, height = U.size
  # resized_image = U.copy()
  # resized_image.thumbnail((1000,2000))
  st.image(U,width = 1300)
  st.subheader("What is all the preprocessing work in machine learning and why?") 
  st.write("Machine learning preprocessing involves a series of indispensable tasks designed to refine raw data and ready it for effective model training and evaluation. Starting with data collection from diverse sources, this phase ensures a comprehensive and relevant dataset. Data cleaning follows suit, addressing missing values, outliers, and inconsistencies to enhance the overall quality and robustness of the dataset. Exploratory data analysis provides a deeper understanding of the data's characteristics, guiding subsequent preprocessing decisions. Feature engineering allows for the creation or modification of features, optimizing the model's ability to capture relevant patterns.The process includes data splitting, differentiating the dataset into training, validation, and test sets to facilitate thorough model assessment. Normalization and scaling of numerical features ensure consistent scales, crucial for certain algorithms' convergence. Categorical data undergoes encoding to convert it into a numerical format compatible with machine learning models. Addressing class imbalances is essential to prevent biased models, and dimensionality reduction techniques help streamline models by reducing the number of features. In image-based tasks, data augmentation techniques generate additional training samples, improving model generalization.Text preprocessing, particularly in natural language processing (NLP), involves tokenization, stop word removal, and stemming or lemmatization. Constructing a processing pipeline streamlines and automates these preprocessing steps, ensuring consistency and efficiency. Time series data considerations, outlier handling, and strategies for managing missing data contribute to the creation of a robust dataset. Transformations for skewed data and encoding ordinal variables preserve information integrity. Finally, data scaling in neural networks aids in optimizing convergence during training. Each of these preprocessing steps collectively prepares the data, enabling machine learning models to learn patterns effectively, generalize well, and make accurate predictions on new, unseen data.")
  st.markdown("[CLICK HERE TO SEE PREPROCESSING WORK OF THIS MODEL](https://colab.research.google.com/drive/1c2zYApjTSGyjsq6lt07yDy3LGagLBI8w#scrollTo=v27R_e61F8AA)")
  
with tab6:
  st.subheader("What is DATA in machine learning")
  st.write('In the realm of machine learning, "data" is the fundamental building block that fuels the learning process of algorithms. It encompasses various types, each serving a distinct purpose in the model development cycle. The "training data" forms the bedrock of model learning, consisting of input features and corresponding output labels or target values. Through exposure to this dataset, the model discerns patterns and relationships, honing its predictive capabilities. As the model undergoes refinement, a portion of the data is reserved for "validation," aiding in the optimization of hyperparameters. Following training and validation, the model encounters "testing data," a set separate from the training process, crucial for assessing the model\'s performance on new, unseen instances. Data manifests in the form of "input features," representing the variables or attributes used for predictions, and "output labels" or "target values" in supervised learning, signifying the values the model aims to predict. "Unlabeled data" is prevalent in unsupervised learning scenarios, where the model identifies patterns without explicit target values. In mathematical terms, input features are often structured as a "features matrix," denoted as (X), and output labels or target values as a "target vector," represented by (y). The quality, quantity, and relevance of data are pivotal factors influencing a model\'s efficacy. Preprocessing and cleaning steps are undertaken to refine the dataset, ensuring optimal learning conditions. The successful navigation of machine learning tasks requires a nuanced understanding of the data\'s nature, guiding the selection of suitable algorithms and techniques. Ultimately, data in machine learning is the catalyst for model intelligence, empowering algorithms to make informed predictions and decisions based on learned patterns.')
  path8 = "C:/Users/Senthil/Desktop/DS/Code/Singapore_flat_P/merged_data.csv"
  with open(path8,"rb") as data:
    Y = data.read()
    st.download_button('DOWNLOAD DATASET BEFORE PREPROCESSING AND FEATURE SELECTION', Y, key='file_download', file_name='dataset.csv')
  
  path9 = "C:/Users/Senthil/Desktop/DS/Code/Singapore_flat_P/Final_data.csv"
  with open(path9,"rb") as data:
    Z = data.read()
    st.download_button('DOWNLOAD DATASET AFTER PREPROCESSING AND FEATURE SELECTION', Z,file_name='final_data.csv')


with tab7:
  with st.spinner("PLEASE WAIT..."):
    col1, col2 = st.columns(2)
    path10 = 'C:/Users/Senthil/Desktop/DS/Projects Photo/contact.jpg'
    H = Image.open(path10)
    col1.image(H,width = 650)
    col2.markdown("[CLICK HERE TO KNOW LINKEDIN PROFILE](https://www.linkedin.com/in/prashanth-k-290569202/)")
    col2.markdown("[CLICK HERE TO KNOW GITHUB PROFILE](https://github.com/Prashanth292003)")
    if col2.button("CLICK HERE TO KNOW EMAIL"):
      col2.write('prashanth.krishnan03@gmail.com')