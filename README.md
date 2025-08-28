# Deep-Learning-Skills
This Repository details my Skills in Deep Learning

### Now, Keras is an API integrated with Tensorflow; Earlier there was a Problem of writing too much code using Tensforflow; Keras made the integration and made it easier

### Mlflow also Supports Tensorflow

## log the model using TensorFlow - mlflow.tensorflow.log_model(model,"model",signature=signature)

### We used a Tensor Model and Keras is a wrap on Top of it

space={
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-1)),
    "momentum":hp.uniform("momentum",0.0,1.0)

}

### From HyperOpt we used hp; It is like Hyperparameter Tuning

A) Deep Learning ANN Model with MLFlow

B) End to End ANN Project



### A) Deep Learning ANN Model with MLFlow

### 1. Used HyperOpt Library - Used to do Hyperparameter Tuning in ANN

There was previous version of Numpy that was incompatible; So we uninstalled it

## from hyperopt import STATUS_OK,Trials,fmin,hp,tpe

import mlflow
from mlflow.models import infer_signature

## 2. Load the dataset
data=pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

### This is a Classification Project, to predict Wine Quality

## 3. Split the data into training,validation and test sets

train,test=train_test_split(data,test_size=0.25,random_state=42)

train_x=train.drop(['quality'],axis=1).values

train_y=train[['quality']].values.ravel()

#### Usage of values.ravel in "train_y=train[['quality']].values.ravel()"

## train[['quality']].values - It is a 2D Array; 

## train_y=train[['quality']].values.ravel() - Gives a 1-D array and that is the reason we used it; Or else we need to reshape it using NumPy Array

## test dataset
test_x=test.drop(['quality'],axis=1).values
test_y=test[['quality']].values.ravel()

#### 4. We created further used Training Data into validation data from Train Data; We use Train and Valid data to check performance of model; We use test data as a New data and do predictions

## splitting this train data into train and validation

train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,test_size=0.20,random_state=42)

signature=infer_signature(train_x,train_y)

### 4. Creating an ANN Model

def train_model(params,epochs,train_x,train_y,valid_x,valid_y,test_x,test_y):
    ## Define model architecture
    mean=np.mean(train_x,axis=0)
    var=np.var(train_x,axis=0)
    model=keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),
            keras.layers.Normalization(mean=mean,variance=var),
            keras.layers.Dense(64,activation='relu'),
            keras.layers.Dense(1)
        ]
    )

mean=np.mean(train_x,axis=0) - For row wise it is going to find mean of all features

### This is required as we have to Perform Normalization when we are training our ANN, our Neural Network

### Next, we found out for Variance for row wise; Both we will use for our Layer Normalization

## Next we took Keras Sequential Model ; keras.Input([train_x.shape[1]] - We took all 11 Input Features and this Becomes our Input Layer

### Then we do Layer Normalization - keras.layers.Normalization(mean=mean,variance=var)

### Creating an Hidden Layer - keras.layers.Dense(64,activation='relu')

### Creating OutPut Layer - keras.layers.Dense(1)

We created an Simple ANN, with not much of Hidden Layers and all

### 5. Compiling the Model:

model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=params["lr"],momentum=params["momentum"]
    ),
    loss="mean_squared_error",
    metrics=[keras.metrics.RootMeanSquaredError()]
    )

### For SGD, we need to give both Learning Rate and also Momentum

### learning_rate=params["lr"] - Params - By what parameter you want to try different different experiments; We can try with different different Hidden Layers; But it Takes Time; So we will try with different Learning Rates; Model will be trained with different different Learning Rates;

### Momentum is also another Parameter which we want to try with different values for better outcome;

### Other reason is also, we want to Log some experiments; That is why we use HyperOpt too, it will try with each and every Parameters we have given over there

### 6. Train the ANN model with lr and momentum params wwith MLFLOW tracking
    
with mlflow.start_run(nested=True):
        model.fit(train_x,train_y,validation_data=(valid_x,valid_y),
                  epochs=epochs,
                  batch_size=64)
        ## Evaluate the model
        eval_result=model.evaluate(valid_x,valid_y,batch_size=64)
        eval_rmse=eval_result[1]
        ## Log the parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse",eval_rmse)
        ## log the model
        mlflow.tensorflow.log_model(model,"model",signature=signature)
        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

### It should also be in the Same function - Train the ANN model with lr and momentum params wwith MLFLOW tracking

### mlflow.start_run(nested=True) - Since we want to try with different different parameters, we have a option caled nested=True to do that; It will go inside and inside and follow a Nested Structure

### Training -    model.fit(train_x,train_y,validation_data=(valid_x,valid_y), epochs=epochs, batch_size=64)

### Evaluate the model - eval_result=model.evaluate(valid_x,valid_y,batch_size=64);

### eval_rmse=eval_result[1] - It will be in the form of a list, so we take first index to get RMSE; We want to track this and also Learning Rate, Momentum, etc.. 

## Log the parameters and results - mlflow.log_params(params); mlflow.log_metric("eval_rmse",eval_rmse)

## log the model using TensorFlow - mlflow.tensorflow.log_model(model,"model",signature=signature)

### Mlflow also Supports Tensorflow

### We used a Tensor Model and Keras is a wrap on Top of it; Signature defines the Schema of Input and Output

## It knows that this entire model is trained on Keras and Tensorflow and knows what are all the artifacts that needs to be create for this particular model

## Finally we returned everything - return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

### We did everything step by step - Created ANN, added Normalization, compiled it, trained model, and while training and compiling we set different parameters for learning rate and momentum; And later Logger all Parameters and Metrics

### 6. Based on HyperOpt created an Objective Function:

def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
    )
    return result

**We gave our Training Mode, parameters, epochs, train,test data and finally returning the result**

### 7. Setting all Parameters:

space={
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-1)),
    "momentum":hp.uniform("momentum",0.0,1.0)

}

### From HyperOpt we used hp; It has for loguniform and uniform; It is like Hyperparameter Tuning; Learning Rate - It will be ranging from 10^-5 to 10^-1

### Momentum - "momentum":hp.uniform("momentum",0.0,1.0); 0 to 1

### 8. In Training we wrote start_run; Now we will write one more to run on top of it

mlflow.set_experiment("wine-quality")
with mlflow.start_run():
    # Conduct the hyperparameter search using Hyperopt
    trials=Trials()
    best=fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=4,
        trials=trials
    )
    # Fetch the details of the best run
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    # Log the best parameters, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature=signature)
    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")

## Start Run - To start the Experiment

### trials=Trials() - This performs HyperParameter Tuning

### best=fmin(fn=objective,space=space,algo=tpe.suggest, max_evals=4,trials=trials) - Function name - Objective; space - Hyperparameters we want to try with;  type.suggest - It will be using internally all kind different kind of algorithms itself and maximum evaluation; And we gave maximum evaluation = 4;  Trials - We gave our trials; We defined these for Hyperparameter Search; 

### best_run = sorted(trials.results, key=lambda x: x["loss"])[0] - Then we found the best details using this; We sort it using Loss function and find out which loss is less; We will be taking it as best run

### Trying to run that best (We defined) - mlflow.log_params(best)

### Once we call Objective, it will call Train Model, Params is assigned to Space, That entire result we get for Logs and we are logging

## We created MLRuns inside the DL Folder; So it is good to go inside the folder and run for MLFlow; 

## Krish did "cd MLDLFlow" and went into the folder and used "mlflow ui" and it created mlruns and we can see experiments with id's for multiple epochs it ran for different learning rate and momentum values

### It gives learning rate, momentum, params, evaluation_metic for each metric

### In VS Code we got best learning rate, momentum, RMSE value

### In MLFLow, we can see expierments and runs; In one run, we were able to see, 5 different experiments, those were sub-experiments for different learning rate and momentums; We can also see for metrics too

### We can compare different runs too

### Once we found out our best model, we can go and register it 

### Once we go to artifacts, we can see model.keras, ML Model, conda.yaml,etc...; We can regigster the model from here

### We can go inside Models and see this, we can give some tags, alias

### We can run for any experiments and compare it; It is the power of MLFlow

#### We can load the model in VS Code too; Go to Artifact, Copy the code in "Validate model before deployment" and do inferencing, with doing some predictions

### We can also do using PyFunc Load_Model

### We can also register model using Code: mlflow.register_model(model_uri, "name_we_want_to_give"); If we go to models, it will automatically be registered, as we wrote code




### B) End to End ANN Project

### 1. Problem Statement and VS Code Setup:

In this series, the goal is to build an end-to-end deep learning project where we train an Artificial Neural Network (ANN). The two main libraries we’ll use are TensorFlow and Keras. TensorFlow is an open-source library that allows us to create deep learning projects from start to finish, and it comes with powerful features to build different types of neural networks such as ANN, RNN, LSTM, GRU, and even Transformers.

A common question is whether it’s necessary to learn both TensorFlow and PyTorch. The advice is that you don’t need to master both. Instead, focus on becoming skilled in just one of them. Since both are open-source and widely used, you can always switch between frameworks later, especially when working with Generative AI models where conversions between PyTorch and TensorFlow are possible. For this project, the focus will be on TensorFlow.

Along with TensorFlow, we’ll use Keras, which is an API integrated into TensorFlow. While TensorFlow by itself requires writing a lot of code to build models, Keras simplifies this process by providing high-level APIs that let you define and train models with just a few lines of code. Keras makes it easy to create sequential networks, RNNs, LSTMs, and more. Even though Keras can also work with JAX and PyTorch, in this course we’ll stick with TensorFlow.

### The problem statement we’ll work on involves a dataset called churn_modelling.csv, which comes from a bank. The goal is to predict whether a customer will leave the bank or not (a binary classification problem). The dataset includes features such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, activity status, and estimated salary. 

### The target column is “exited,” which indicates if a customer left the bank.

The project steps will be as follows:

(i) First, we’ll perform basic feature engineering, such as converting categorical variables to numerical ones and standardizing values.

#### (ii) Then, we’ll build our ANN model. For example, if there are 11 features, the input layer will have 11 nodes, followed by one or more hidden layers, and finally an output layer with one node for binary classification.

(iii) We’ll also introduce dropout, which temporarily disables some nodes during training to reduce overfitting.

(iv) Training will involve forward propagation, calculating the loss, and updating weights using optimizers, all of which we’ll implement using TensorFlow and Keras.

(v) Once the model is trained, we’ll learn how to save it in formats like pickle (.pkl) or H5 (.h5), which store the model architecture and weights for reuse and deployment. 

### Deployment will be done through Streamlit, where we’ll build a simple web app, integrate our trained model, and then deploy the app on the Streamlit cloud.

(vi) Before starting, we’ll set up a Python environment locally using conda. After creating the environment, we’ll install the required libraries listed in a requirements.txt file, which will include TensorFlow (version 2.15.0), Pandas, NumPy, Scikit-learn, TensorBoard, Matplotlib, and Streamlit. TensorFlow will run on CPU for this project since the dataset is small, but for larger and more complex projects, we’ll switch to Google Colab or a machine with GPU support.

In summary, this project will cover everything step by step: preprocessing the data, building and training an ANN using TensorFlow and Keras, applying techniques like dropout, saving and deploying the model, and finally integrating it into a Streamlit app for real-world use.

**2. Feature Transformation using sklearn with ANN**

### Here Krish Got one error on write and fixed using "geo_encoded_df = pd.DataFrame(geo_encoded.toarray()"

Now I have also activated this specific environment in my terminal. So let me just hide my terminal right now because I will not be requiring it. What I will do, along with this one last thing, is also the installation of ipykernel. Since the first experiment that I am actually going to do will be inside my Jupyter notebook, for that I require the ipykernel library so that I will be able to attach a kernel and execute my Jupyter notebook file. So here I am actually going to write "pip install ipykernel". Once I execute this, you will be able to see that the installation will take place. Once this ipykernel has been executed, then I am going to start my coding.

The first thing that I will do is read this entire CSV file. Let me also go ahead and create my experiments.ipynb file. In this notebook, my entire experiment will happen. So right now you can see my ipykernel library is getting installed, and that is going to take some amount of time. Now it is done. I will quickly hide this terminal. Now let me just go ahead and select my kernel. Here, I have selected my kernel with Python 3.11. Now I will just go ahead and make some more code cells, and let’s start this entire coding.

First of all, I am going to import pandas. So I will just go ahead and write "import pandas as pd". Along with this, I will import from sklearn.model_selection because I also have to do the train-test split. So for that, I will write "from sklearn.model_selection import train_test_split". I hope you know this from machine learning which we have already used. Then along with this, I will also go ahead and import from sklearn.preprocessing. So here I will just go ahead and write "from sklearn import preprocessing". Then I am going to import two important classes: one is StandardScaler and the other one is LabelEncoder. So I will write "from sklearn.preprocessing import StandardScaler, LabelEncoder". These two libraries I am going to use. Along with this, I will also go ahead and import pickle because when we are using StandardScaler and LabelEncoder, it is important to pickle these files so that we can reuse them later during deployment. So I will also add "import pickle".

So these are some of the basic libraries. Now let us go ahead and load the dataset. For this, I will say "data = pd.read_csv('Churn_Modelling.csv')". Now over here, I will just go ahead and display "data.head()". These are all my values or all my features that are available over here. I have already told you in my previous problem statement what we are going to do: we are going to take this CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, and EstimatedSalary as my independent features. And I am going to predict whether the person is going to exit the bank or not. This is the past data from the bank which shows which person has exited or not. So this is just the top five records.

Now the first important thing is that the first three features are not that important, right? You have RowNumber, CustomerId, and Surname. These definitely will not play a very important role. So first of all, what we will do is pre-process the data. We are going to drop irrelevant features, which we can directly see. So here I will use this data variable and from this, I will say: "data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)". Once we do this particular drop, we will be able to get the cleaned data. Now my features are starting from CreditScore, Geography, Gender, Age, Balance, and so on.

Now here you can actually see that we have a column called Geography and another one called Gender. These two features are categorical variables. In the case of categorical variables, what I can do is apply some kind of encoding. That is what I am going to do. Let’s first encode Gender using LabelEncoder. I will write "label_encoder_gender = LabelEncoder()". Then I will transform Gender with "data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])". Once I do this, my Gender column, which had two main features (Male and Female), has now been converted to 0s and 1s. This is one of the encoding techniques with respect to categorical variables.

So over here, Gender values (Male and Female) have been converted to 0 and 1. But what about this Geography column? In the Geography column, you basically have 2–3 categories like France, Spain, and Germany. If we directly apply LabelEncoder, for example, France = 0, Spain = 1, Germany = 2, then the model might mistakenly assume Germany > Spain > France because ML models treat numbers as having an order. This should not happen. So for this case, instead of LabelEncoder, we will use OneHotEncoder. One-hot encoding will give us separate binary columns for each country.

So now I will import the one-hot encoder: "from sklearn.preprocessing import OneHotEncoder". I will initialize it using "onehot_encoder_geo = OneHotEncoder()". Now I will fit and transform Geography using "geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']])". This gives me a sparse matrix. If I want to see the feature names, I can write "onehot_encoder_geo.get_feature_names_out(['Geography'])", and I will get Geography_France, Geography_Germany, Geography_Spain.

Now to convert this into a DataFrame, I will write "geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))". Once I do this, I can see that for each row, one of these geography columns is set to 1 and the others to 0. This fixes the problem of wrongly assuming order between categories.

Next, I will combine these new encoded columns with my data. First, I will drop the old Geography column with "data = data.drop('Geography', axis=1)". Then I will concatenate: "data = pd.concat([data, geo_encoded_df], axis=1)". Now if I check "data.head()", I can see Gender is encoded as 0/1, and Geography has become three columns: Geography_France, Geography_Germany, and Geography_Spain.

So we have converted categorical features into numerical ones. We used LabelEncoder for Gender and OneHotEncoder for Geography. Since we will need these encoders later (e.g., in deployment), we must save them using pickle. So to save LabelEncoder:
"with open('label_encoder_gender.pkl', 'wb') as file: pickle.dump(label_encoder_gender, file)"
Similarly, to save OneHotEncoder:
"with open('onehot_encoder_geo.pkl', 'wb') as file: pickle.dump(onehot_encoder_geo, file)"

Now let us divide the dataset into independent (X) and dependent (y) features. The dependent feature is "Exited". So I will write: "X = data.drop('Exited', axis=1)" and "y = data['Exited']". Then I will split the dataset into train and test using "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)".

Next, I will scale the features using StandardScaler. First initialize it: "scaler = StandardScaler()". Then fit and transform X_train with "X_train = scaler.fit_transform(X_train)", and transform X_test with "X_test = scaler.transform(X_test)". Once this is done, I will also save the scaler with pickle:
"with open('scaler.pkl', 'wb') as file: pickle.dump(scaler, file)"

So here you have everything: LabelEncoder pickle, OneHotEncoder pickle, and StandardScaler pickle.

In short, what I have done so far is clean the dataset, apply feature engineering, divide data into independent and dependent features, encode categorical variables (LabelEncoder + OneHotEncoder), and scale numerical variables (StandardScaler). Finally, I saved all the preprocessing objects as pickle files. Now my data is ready. On this data, I will go ahead and train my Artificial Neural Network, which I will show in the next step.

### **3**

Right now we will go ahead and train an artificial neural network, and that is what we are going to see — how we are going to train this and all. We’ll be discussing about that in detail. So first of all we will go ahead and import TensorFlow. Since we are going to just use TensorFlow as an alias, we will import it as tf, and tf is the alias that we will be specifically using.

Before I go ahead, let me just give you a brief idea of how a neural network basically gets created with the help of TensorFlow Keras. With respect to an ANN (Artificial Neural Network), you have an input layer, then you have hidden layers, and you can have multiple hidden layers such as L1, L2 and so on. But just to give you a basic idea, a basic ANN will be having one input layer, one or more hidden layers, and finally one output layer. Let’s say in this input I have two inputs, in the first hidden layer I have three nodes, in the second hidden layer I have two nodes, and finally I have one output node. This neural network will perform forward propagation and backward propagation. In TensorFlow we basically call this a sequential model. An ANN is nothing but a sequential model, and in this sequential model we will be performing forward propagation and backward propagation, and all these nodes will be interconnected. The input will be connected to the hidden layer, the hidden layers will be connected to each other, and finally to the output.

This input is my input layer, then hidden layer one, then hidden layer two, and finally the output layer. All the interconnected lines between these layers are nothing but the weights W1, W2, W3 and so on. For example, if I have two inputs and three hidden neurons in the hidden layer, then the number of weights will be 2 × 3 = 6 weights. Along with these six weights, three biases will also be added, let’s say B1, B2, and B3. Similarly, in the second hidden layer if three neurons are connected to two neurons, the weight matrix will be 3 × 2 = 6 weights, and here also biases will be added, let’s say B4 and B5. Finally, in the output layer where two neurons are connected to one neuron, there will be 2 × 1 = 2 weights, plus one more bias. After this, depending on the type of problem, we can apply an activation function like sigmoid for binary classification, or softmax for multi-class classification.

If we now calculate the total number of trainable parameters in this network, it will be the sum of all weights and biases: 6 + 3 = 9, then 9 + 6 = 15, then 15 + 2 = 17, then 17 + 2 = 19, and finally 19 + 1 = 20. So with the help of 20 trainable parameters, forward and backward propagation will take place. Now, the number of inputs in the input layer is defined by the number of features in our dataset. If the dataset has 2 columns, then there will be 2 input nodes, if it has 10 columns, then 10 input nodes, and so on. In this way we go ahead and create our ANN which is basically a sequential network, meaning all layers will be interconnected.

Now, whenever we want to create hidden neurons in Keras, there is a class called Dense. So for example, if I say Dense with 64 nodes, then that particular hidden layer will have 64 neurons. Along with this, in every node we apply an activation function such as sigmoid, tanh, ReLU, leaky ReLU, etc. But the best practice is that in the hidden layers we use ReLU, and in the output layer we use sigmoid or softmax depending on the problem. Apart from this, the next important parameter is the optimizer, which is responsible for updating the weights during backpropagation. We will initialize the optimizer while compiling the model. The fifth important part is the loss function, which determines how the error is calculated, and gradient descent will try to minimize this loss. Finally, we also define metrics, such as accuracy in classification problems, and mean squared error or mean absolute error in regression problems.

Additionally, we will store all training logs in a folder so that we can visualize them using TensorBoard, which displays graphs and helps us understand the training progress better. So summarizing the key steps: first we create a sequential model, then we add dense layers (neurons), apply activation functions, specify optimizer, loss function, and metrics, and finally store logs for TensorBoard visualization.

Now let us move to the code part. First we will import TensorFlow and the necessary classes. We write:
"import tensorflow as tf"

Since we are going to use the Sequential model, we will import it as:
"from tensorflow.keras.models import Sequential"

Whenever we want to create a hidden neuron, we will use Dense, which is present inside layers:
"from tensorflow.keras.layers import Dense"

We will also be using EarlyStopping from callbacks, so we import it as:
"from tensorflow.keras.callbacks import EarlyStopping"

Along with this, we will also import TensorBoard:
"from tensorflow.keras.callbacks import TensorBoard"

Finally, we import datetime to handle log file naming:
"import datetime"

So the libraries that we have imported are Sequential, Dense, EarlyStopping, TensorBoard, and datetime.

Now let’s build our ANN model. To build the model, we first initialize a sequential model:
"model = Sequential()"

Inside this sequential model, we will add layers. First, we need to check how many inputs we have, which can be done using X_train.shape. The number of columns in X_train will decide how many inputs are there in the input layer. Let’s say X_train.shape[1] gives us the number of features. So in the first hidden layer, we will use Dense with 64 neurons, activation as ReLU, and we will also specify the input shape. For example:
"model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))"

This becomes the first hidden layer connected with the input layer. Now in the second hidden layer, we can use:
"model.add(Dense(32, activation='relu'))"

Notice that here we do not need to specify the input shape again, because the Sequential model automatically connects the previous layer to the next. Finally, for the output layer, since it is a binary classification, we will have one output neuron with a sigmoid activation function:
"model.add(Dense(1, activation='sigmoid'))"

So this becomes our entire ANN model: 12 input features connected to a hidden layer with 64 neurons, then another hidden layer with 32 neurons, and finally one output neuron.

Now, let us summarize the model to check the number of parameters:
"model.summary()"

The output shows that the total number of parameters is 2945, which is a combination of all weights and biases in the layers. This confirms our model architecture.

Now in order to perform forward and backward propagation, we need to compile the model. We do this by writing:
"model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"

Here, the optimizer is Adam, the loss function is binary cross-entropy since this is a binary classification problem, and the metrics we track is accuracy. In case it was a multi-class classification, we would use sparse categorical cross-entropy as the loss function.

Now, quickly, one more thing to show here. Is this the only way to put any values or is it also possible to use some other way of applying optimizers?

One more way is as follows:

The optimizer is nothing but tensorflow. There is a function called optimizers. Specifically, it is tensorflow.keras.optimizers. Within this, there is the Adam optimizer. The learning rate can be set, for example, as learning_rate = 0.01. If tensorflow is not defined, it will not work, so that must be handled.

When an optimizer is given, such as Adam, this keyword can be used, as mentioned in the Keras API documentation. However, Adam by default has a fixed learning rate. To provide a custom learning rate, tensorflow.keras.optimizers.Adam can be imported and used. There are also many other optimizers, such as Adadelta, Adafactor, Adagrad, AdamW, Adamax, and more. Adam, Adadelta, and Adagrad have already been discussed. In this case, Adam is chosen. A parameter called learning_rate can be defined here as well.

Similarly, tensorflow.keras.losses can be used to set the loss function. Various losses are available such as BinaryCrossentropy and CategoricalCrossentropy. For instance, BinaryCrossentropy can be used directly and stored in a variable like loss. That particular loss can then be applied during compilation, just like the optimizer. Both direct usage of the keyword and explicit initialization are possible.

The optimizer can then be placed into the compilation step, and similarly, the chosen loss function can be applied. When compiling the model, if an error occurs with metrics, it should be noted that the correct keyword is metrics.

Next, TensorBoard can be set up, since training has not yet started. Once training is completed, logs can be captured and visualized. For this purpose, a log directory must be created. A folder such as log/fit is created. A datetime format is then used with datetime.now().strftime(...) to generate a unique directory name for storing logs. This ensures that whenever training occurs, logs will be stored in this folder.

After this, a TensorFlow callback is created for TensorBoard. To import TensorBoard, two modules are required: from tensorflow.keras.callbacks import both EarlyStopping and TensorBoard. TensorBoard is used to visualize training logs. While similar results can be obtained manually with matplotlib, TensorBoard provides a more effective solution.

TensorBoard can be initialized with the log directory and additional parameters such as histogram_freq=1 to create histogram diagrams. After correcting syntax errors (e.g., using from instead of incorrect import syntax), this setup works correctly.

In addition, early stopping is configured. Early stopping is used because training a neural network can involve many epochs (e.g., 100 epochs), but sometimes after a smaller number of epochs (e.g., 20), the model may stop improving significantly. Monitoring the loss value helps identify when further training is unnecessary. If the loss value does not decrease for several epochs, training can stop early to save resources.

An early stopping callback is created by monitoring val_loss. Patience can be set, for example, to 5 epochs, meaning training will only stop if no improvement is seen for 5 epochs. The parameter restore_best_weights=True ensures that the best weights obtained during training are reloaded when early stopping is triggered.

At this point, the model is ready for training. The model has been created, optimizers and loss functions have been assigned along with metrics, TensorBoard has been set up to record logs, and both TensorBoard and early stopping callbacks have been defined.

Training is performed with:

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping_callback]
)


Here, y_test is used for comparison with respect to accuracy. Although training is requested for 100 epochs, early stopping may stop it earlier.

During execution, epochs begin, and accuracy along with validation accuracy is displayed. Even though 100 epochs are set, training may stop earlier, for example after 7 epochs, due to early stopping. If patience is increased to 10, then training continues further, stopping at around epoch 16 or 17.

The results show values such as training accuracy reaching 87%, validation loss values, and validation accuracy of around 0.85. These values confirm that the neural network is trained effectively.

Finally, the trained model can be saved. Using model.save("model.h5"), the model is saved into an .h5 file, which is compatible with Keras.

The TensorBoard extension can be loaded in order to visualize training logs. To begin, the TensorBoard extension is loaded using the command load_ext tensorboard. Initially, an attempt is made to use load_extension, but the correct syntax is load_ext. Once this is executed correctly, the TensorBoard session is launched.

After launching TensorBoard, the log directory must be specified. The log directory contains the folder logs/fit, which holds information about training and validation. Loading TensorBoard with this directory displays the results in a visualized format.

At first, when loading TensorBoard, a message may appear stating that no dashboards are active for the current data. To fix this, the TensorBoard session can be launched again. By clicking the “Launch TensorBoard session” option, a prompt appears asking whether to use the current working directory or select another folder. Another folder can be selected, such as the training folder, and then loaded.

If TensorBoard is unable to find the logs/fit directory, the issue may be due to the folder being renamed automatically with a timestamp or a different format. In this case, the exact folder name needs to be used. For example, instead of fit, the renamed folder name can be copied directly from the file explorer and used in place of fit. This ensures the correct log directory is provided. If the folder is in use, renaming directly may not work, so copying the full folder name and specifying it in the path is the best solution.

Once the correct directory is specified, TensorBoard loads successfully. The diagrams begin to appear. The main visualizations include epoch accuracy and epoch loss. With respect to epoch accuracy, the graph shows the accuracy increasing, for example from 0.855 to 0.8655. With respect to epoch loss, the graph shows the loss decreasing, which indicates improvement in training.

Additional options such as smoothing can be applied to see how the accuracy changes. Relative steps, values, and accuracy metrics are displayed as well. Another important visualization is related to dense layers (e.g., dense_3). The histogram shown here looks like a gradient descent curve in the opposite manner, where the maximum value is at the top and the minimum loss value is at the bottom. These histograms provide insight into weights and activations within the network.

The same can also be observed for validation. For validation accuracy, some fluctuations appear, but the accuracy can still reach around 0.855. Validation loss, epoch accuracy versus iteration, and epoch loss versus iteration are also available for analysis.

It is important to note that the correct folder name must always be used when loading TensorBoard. The reason for different folder names is the use of a datetime format in the log directory. Instead of directly writing logs/fit, the directory is created as logs/fit/<datetime> by default. This timestamp ensures unique logging directories for each training session. The code should therefore be structured such that inside the fit folder, a new folder is automatically created with the current datetime. This removes the need to manually rename or locate folders.

Up to this point, TensorBoard has been successfully loaded, training results have been visualized, and both training and validation metrics have been examined. Along with this, the pickle file has already been saved, completing almost every required step.
