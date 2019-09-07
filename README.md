<h4>census_data_model using machine learning algorithm as well as deep learning algorithm</h4>
<h3>Project Description</h3>
<p>In this project, I used sklearn and supervised learning techniques on data collected for the U.S. census to help a fictitious charity organization identify people most likely to donate to their cause.</p> 
<p>Here, I first investigate the factors that affect the likelihood of charity donations being made. Then, I use a training and predicting pipeline to evaluate the accuracy and efficiency/speed of six supervised machine learning algorithms ('logistic_regression','support_vector_machine','RandomForestClassifier','knearest_neigbour_classification','AdaBoostClassifier','GradientBoostingClassifier'). I then proceed to fine tune the parameters of the algorithm that provides the highest donation yield (while reducing mailing efforts/costs). Finally, I also explore the impact of reducing number of features in data.</p>
<p>After carefully studying given scores and details about the model which gave max prediction around 87%, I shifted my attension of using deep learning model. Which clearly gave me an accuracy score of 97% after some tuning.</p>
<p>The main.py is used to predict and get an accuracy score for both machine learning and deep learning model dynamically. The trained deep learning model is saved in model folder and also note that graph for learning curve with train vs validation curve saved in img folder which is also dynamically saved according to the choice of the user from config.json which is in config folder.</p> 
<p>For detailed anlysis and usage of machine learning algorithm(complete project) please look into project.ipynb but for deep learning model please do look into src folder and main.py.</p>

<h3>Install</h3>
<p>This project requires **Python 3.7** and the following Python libraries installed:</p>
<ul><li>NumPy</li>
<li>Pandas</li>
<li>matplotlib</li>
<li>scikit-learn</li>
<li>keras</li></ul>

<h2>Code</h2>
<p>The main code for this project is located in the `project.ipynb` notebook file. The other main codes is intertwined together along with main.py such as the folders(config,img,data,log,model,score,src) and files(census.csv,config.json,census.py,jsonstore.py,model_graph.py) are all working together to provide output in terminal and as well as file format such as accuracy_val.json,info.log,my_model.h5 as well as the graph produced in the img folder.</P>


