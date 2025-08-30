# Deep-Learning-Skills
This Repository details my Skills in Deep Learning

### Very Very Important: In ANN, we create a sequential model, then we add dense layers (neurons), apply activation functions, specify optimizer, loss function, and metrics, and finally store logs for TensorBoard visualization.

### Very Very Important: Please go and see "B) End to End ANN Project"; For detailed working of ANN with Project Notes; It will be very useful

## Input Layer connected to HL1 alone should be written with input_shape = Dense(64,activation='relu',input_shape=(X_train.shape[1],))

##  Now in the second hidden layer, we can use: "model.add(Dense(32, activation='relu'))"; Here we do not need to specify the input shape again, because the Sequential model automatically connects the previous layer to the next. Finally, for the output layer, since it is a binary classification, we will have one output neuron with a sigmoid activation function: "model.add(Dense(1, activation='sigmoid'))"

## Output of "model.summary" shows that the total number of parameters is 2945, which is a combination of all weights and biases in the layers. This confirms our model architecture.

## In order to perform forward and backward propagation, we need to compile the model. We do this by writing: "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"; Here, the optimizer is Adam, the loss function is binary cross-entropy since this is a binary classification problem, and the metrics we track is accuracy. In case it was a multi-class classification, we would use sparse categorical cross-entropy as the loss function.; Here, the optimizer is Adam, the loss function is binary cross-entropy since this is a binary classification problem, and the metrics we track is accuracy. In case it was a multi-class classification, we would use sparse categorical cross-entropy as the loss function.  (**See below notes for more detail)**

## Similarly, tensorflow.keras.losses can be used to set the loss function. Various losses are available such as BinaryCrossentropy and CategoricalCrossentropy. For instance, BinaryCrossentropy can be used directly and stored in a variable like loss. That particular loss can then be applied during compilation, just like the optimizer. Both direct usage of the keyword and explicit initialization are possible.

## Next, TensorBoard can be set up, since training has not yet started. Once training is completed, logs can be captured and visualized. For this purpose, a log directory must be created. A folder such as log/fit is created. A datetime format is then used with datetime.now().strftime(...) to generate a unique directory name for storing logs. This ensures that whenever training occurs, logs will be stored in this folder; TensorBoard can be initialized with the log directory and additional parameters such as histogram_freq=1 to create histogram diagrams. After correcting syntax errors (e.g., using from instead of incorrect import syntax), this setup works correctly.

## In addition, early stopping is configured. Early stopping is used because training a neural network can involve many epochs (e.g., 100 epochs), but sometimes after a smaller number of epochs (e.g., 20), the model may stop improving significantly. Monitoring the loss value helps identify when further training is unnecessary. If the loss value does not decrease for several epochs, training can stop early to save resources; An early stopping callback is created by monitoring val_loss. Patience can be set, for example, to 5 epochs, meaning training will only stop if no improvement is seen for 5 epochs. The parameter restore_best_weights=True ensures that the best weights obtained during training are reloaded when early stopping is triggered.

## At this point, the model is ready for training. The model has been created, optimizers and loss functions have been assigned along with metrics, TensorBoard has been set up to record logs, and both TensorBoard and early stopping callbacks have been defined.

## Training is performed with: history = model.fit( x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[tensorboard_callback, early_stopping_callback] ); Here, y_test is used for comparison with respect to accuracy. Although training is requested for 100 epochs, early stopping may stop it earlier.

## During execution, epochs begin, and accuracy along with validation accuracy is displayed. Even though 100 epochs are set, training may stop earlier, for example after 7 epochs, due to early stopping. If patience is increased to 10, then training continues further, stopping at around epoch 16 or 17. The results show values such as training accuracy reaching 87%, validation loss values, and validation accuracy of around 0.85. These values confirm that the neural network is trained effectively.

## Finally, the trained model can be saved. Using model.save("model.h5"), the model is saved into an .h5 file, which is compatible with Keras.

## The TensorBoard extension can be loaded in order to visualize training logs. To begin, the TensorBoard extension is loaded using the command load_ext tensorboard. Initially, an attempt is made to use load_extension, but the correct syntax is load_ext. Once this is executed correctly, the TensorBoard session is launched; After launching TensorBoard, the log directory must be specified. The log directory contains the folder logs/fit, which holds information about training and validation. Loading TensorBoard with this directory displays the results in a visualized format.

#### For rest of the above project, B) End-to-End ANN Project, please see below, it have very detailed end to end notes from the project for great understanding

### Now, Keras is an API integrated with Tensorflow; Earlier there was a Problem of writing too much code using Tensforflow; Keras made the integration and made it easier

### Mlflow also Supports Tensorflow

## log the model using TensorFlow - mlflow.tensorflow.log_model(model,"model",signature=signature)

### We used a Tensor Model and Keras is a wrap on Top of it

space={
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-1)),
    "momentum":hp.uniform("momentum",0.0,1.0)

}

## In ANN, Krish Got one error on write and fixed using "geo_encoded_df = pd.DataFrame(geo_encoded.toarray()" (For merging encoded geo dataframe with main dataframe)

## In ANN, let's say we hvae input I two inputs, in the first hidden layer I have three nodes, in the second hidden layer I have two nodes, and finally I have one output node. This neural network will perform forward propagation and backward propagation. In TensorFlow we basically call this a sequential model. An ANN is nothing but a sequential model, and in this sequential model we will be performing forward propagation and backward propagation, and all these nodes will be interconnected. The input will be connected to the hidden layer, the hidden layers will be connected to each other, and finally to the output.

### From HyperOpt we used hp; It is like Hyperparameter Tuning

#### For Finding Optimal Hidden Layers and Hidden Neurons in ANN -Heuristics and rules of thumb also play an important role in providing a good starting point. For example, one rule suggests that the number of neurons in the hidden layer should be between the size of the input layer and the size of the output layer. A common practice is also to begin with one or two hidden layers before increasing further. To make this process more practical, one useful approach is to perform hyperparameter tuning by trying out various numbers of neurons and hidden layers, and then observing which combination provides the best model

#### Table of Contents:

A) Deep Learning ANN Model with MLFlow

B) End to End ANN Project - Classification

C) End to End ANN Project - Regression

D) Finding Optimal Hidden Layers And Hidden Neurons In ANN

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

### **3. Step By Step Training With ANN With Optimizer and Loss Functions**

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

#### 4. Prediction with Trained ANN Model

How do we load the pickle file? Because we need to do the prediction, right? So how do we load the pickle file? That is very much important to show, because at the end of the day when we create a end to end Streamlit project. So we definitely require all the pickle files. So in order to load the pickle file. So first of all, uh, um, what I have to do because see here I now have three pickle files. I have all my model trained. Now I have to make sure that I load that particular pickle file and I do the prediction right. That is what I have to actually do. If I don't do that, then it will not work out right. So let me quickly go ahead and import all the libraries again.

So let's say that I'm doing it in my new file okay. Or let's consider a new file. So here I will say prediction dot ipynb okay. Here the first thing first I will go ahead and import this. Select the kernel okay. So here you will be able to see that I'm going to execute this okay. Now we are going to load all the trained model right. So how many trained model we have. So load the Ann trained model then um scalar pickle file and the one hot encoding pickle file also. Right. One hot encoding. So we'll load this. So I'll go ahead and write mode model is equal to load underscore model. So here is what we have imported see from TensorFlow Keras dot models import load underscore model. So here I'm going to write load underscore model. And here I'm going to take my model dot h5 file okay. When I probably go ahead and use this here you'll be able to see I will also go ahead and load my load the encoders and scaler.

Now in order to encode uh, load this I will go ahead and use with open. Quickly. Let's go ahead and write label underscore encoder underscore geo dot pickle file. So this will be my pickle file. It will we will be reading this time in the read byte mode as file. Because pickling is nothing. But it is a deserialized format right. We can serialize it. We can deserialize it. So here I'm going to basically you go ahead and use this label underscore underscore Geo is nothing but pickle dot load this specific file name okay. So here is what I get. This particular pickle file over here okay. And similarly I will go ahead and do the same thing for the gender and for the scalar okay. So here I have my gender and for my scalar.

Now let's take some input data. So I will take some input data. So this I will go ahead and execute it. Uh label encoder underscore geo. What is the name that I've given. Label encoder uh underscore gender. Sorry. It should be gender not geo. Okay. Oh, no. This pickle file should be my geo pickle file is what? Uh, one hot encoder. Geo. Right? Not label encoder. So here I'll go ahead and write. One hot. One hot. Let me rename this and let me copy this entire pickle file okay. So here I'm just going to go ahead and write one underscore hot encoder underscore geo dot pickle file. Now let me go ahead and execute it. Now it works absolutely fine. Right now I'm going to take some example input data. Because based on this input data I'm going to do the prediction.

So here I have written credit score is equal to 600. Geography France. Male. Gender male age 40 tenure 30. This this this. There. Now when I'm giving this particular data, the first thing that I need to think is that how do I convert this particular value into numerical and this particular value into numerical. Right. So and then after that take all the numerical value and convert that into standard uh means scale down that particular value using standard scalar. That is the reason we have so stored this in the form of pickle file. Okay.

Now uh, let's quickly go ahead and do this. Uh, first of all, uh, I would like to see just think over it, okay? You, uh, just try to think like, how do we probably go ahead and apply, uh, one hot encoding in this? Okay. Now you really need to think, guys, uh, you cannot, uh, just think over it, okay? I'll. I'll give you five minutes. Just pause the video. But what task you really need to do? Okay, this is my new input data I'm actually getting. And now with respect to this new input data, you need to think, how can I probably take this particular data and perform a convert categorical features into, uh, numerical features? That is what you really need to think. First of all, the second thing is, uh, uh, how do you apply standard scaling? Right. And then how do you do the prediction with respect to the model? Okay. That is what we are going to probably see. Okay.

So first of all, uh, pause the video and do it. Till then I will start copying and pasting the code over here. The reason is very simple because I have already told you all these things. Okay. So here, uh, one hot uh, so let me just go ahead and write this label encoder G0. So I'll take this label encoder g0 dot transform okay. And here you will be able to see that I am getting some values over here. Right key value pairs. Right key value pairs over here. Now when I get this key value pairs I need to take out the information of this particular geography. So what I will do I will say hey uh uh label underscore underscore encoder dot geo dot transform. And here I'm just going to use my input underscore data okay I think it is input underscore data of geography column okay. And then uh this geography is basically my key. So let me just go ahead and copy and paste it over here with respect to this. So I will be able to get my key okay. And then finally I get the I convert this into an array okay.

Now once you do this again I'll be using this label underscore Geo okay dot get feature names out of geography. And this is what is my geo encoded that I'm actually giving okay. So if you see this is for my new data that I am actually able to get this right. So here it says uh reshape one comma one okay. Let's see this. Okay. Um, here I will just go ahead and use this as like this. Then it should work. 

#### Krish Got a same error and then used on extra bracket

Okay. Uh, expected 2D array. Got 11D array instead. Okay. So array of France I have given over here I'm using this transform. Okay. Um, um. This is okay. I have to use one more brackets because that is the. That is the reason how we got it right now. It should work. So here you can see geography underscore front geography underscore Germany. Geography underscore Spain.

Guys you may be thinking how did I get to know that I have to use, uh double brackets. See over here, if you go ahead and see with respect to the transformation that we did with the help of, uh, one hot encoding, right? So here, when we use this one hot encoding here, you can see that, uh, let's see where it is. Okay. Uh, here you can see I'm actually giving this one hot encoding over here. And I'm giving to, uh, a list of features, right? List of features. And similarly, I also have to give a list of features over here. That is the reason this is one of my feature over here. And I'm giving it as a list of features. Right. So that is the reason I'm able to get it okay. So once I do this uh, then I will go ahead and combine this entire data, this data with my input data.

So I'll say input data dot reset index drop is equal to true. Because if I do the reset index I'm actually going to get that uh default index value row index value. And I'm saying drop drop is equal to true. And then I'm uh adding this geo underscore uh encoded underscore df okay. 

### Here also Krish got an error; Input is in form of Key value pairs and need to convert into an Dictionary

So if I go ahead and see my input data now input data. See. So guys uh, here you can actually see that I'm getting this particular error. Uh, dictionary object has no attribute reset index because I just made one simple mistake after this particular code. See, the input data right now is in the form of key value pairs, right. I need to convert this into, uh data frame. So in order to convert this into a data frame we will go ahead and write PD dot data frame. And let me quickly go ahead and use this input underscore data okay. And once I get this this will basically be my input underscore DF. And let's go ahead and execute this. So here is my input underscore df.

Um and this is what is my entire row right now. This row only I have to pass it over there right now for this particular row. What I really need to do, first of all, I need to do the concatenation. Right. So my this is my input underscore df. 

## And I, uh, after converting this into my data frame, I should probably, uh, uh, concatenate my geographic column. Right. So I will do that concatenation right now. Okay. I have actually changed all the values with respect to the geography column over here. Right. So it is basically into two array. We have actually done this. And I've actually got this geo underscore uh encoded underscore df. Now let's uh quickly encode my categorical variables okay.

Now I will say hey let's take this input underscore DF and let's convert first of all this gender because gender also needs to be replaced with some label encoded value. So I'm just going to use this label encoder label encoder uh label encoder underscore gender. And I'll say hey go ahead and do this transform operation on my input underscore df input underscore df on which column on the gender column. So this basically becomes my input underscore df. So if you go ahead and see my input underscore df. Now the gender column will either have zeros or ones. So right now it is one right. So this particular value may be a male. Before now it has basically got one okay. So this is perfect. Till here we are able to get it okay.

Uh I've already done the one hot encoding for geography column. So I have this geo underscore encoded underscore df. Now what I will do is that I will quickly go ahead and do the concatenation. Now finally let's do the concatenation. Concatenation um with one hot encoded data one hot encoded data. And uh along with this we also need to add the uh uh, no need to add the gender data because it is already done. The changes in input underscore DF. Now what we'll do is that whatever encoding we have done for the geography column, right. 

Now, We need to append all those columns inside this. And we need to drop this column. So here I will just go ahead and quickly write input underscore DF is equal to PD dot concatenation. Uh let's concatenate our input underscore df. And from this I'm just going to drop my geographic column. Same thing, what we have actually done over there. Right. So geographic column and this drop will be done with respect to axis is equal to one. And here I'm just going to go ahead and write geo underscore underscore uh encoding underscore DF right. So geo underscore encoded underscore DF is nothing but this particular value which has this one uh which has this one hot encoding of this particular feature.

### Next is Scaling:

Okay. That is what we are basically going to do over here. And uh, I have to also make sure that I put with axis is equal to one. Otherwise it will become row wise concatenation instead column wise concatenation. Now if I go ahead and write input underscore df, you'll be able to see that gender has got converted from male to one. And here you'll be also able to see that my three features have got added. One is geographic France, Germany and Spain. Okay, uh, now one more transformation is left. Uh, that is nothing but scaling the data. Scaling the input data now in order to do the scaling part. So I will be using scaler dot transform. And here let's go ahead and write input underscore df okay. So this will basically be my input underscore scaled. Done. Okay.

Now if you go ahead and see this input underscore scaled, I will basically get all the data in the form of array. And all these values has basically got scaled okay. Now finally we need to just predict and see whether the person is going to leave the bank or not. 

### So I'll go ahead and write prediction. Uh, in order to do the prediction I'll just go ahead and write model dot predict here, give all the values over here like input underscore scaled okay. Get my prediction. Get my prediction. So. I will just go ahead and print my prediction quickly. So here you can see that I'm getting this particular value 0.20.02976. So and so. Right. Um, I can just go ahead and uh, get my prediction probability also. So if I just go ahead and write like this prediction of zero of zero, I'll be able to see that I'm, I'm, I'm okay right now.

Uh, let's see this. I will just go ahead and write prediction scope probability. So here you'll be able to see 0.297. It is there. I will go ahead and say this particular condition. If the prediction probability is greater than 0.5 the customer is likely to churn. Otherwise the customer is not likely to churn. Okay. Because this is a binary output. And I'm just going to keep the binary output with respect to 0.5. But here I'm actually getting 0.029. So the customer is not likely to churn. 

So in this video entirely right. How to do the specific prediction I've actually done it again. First of all we have loaded all the pickle files. Then whenever we get any input data, first of all we converted that into a data frame. Then we did the one hot encoding of geography. Then uh, we did label encoding for gender. Then we combined all those things. Then uh, we created our entire data frame. Then we did the scaling. Right. Again. We used the transform feature. Then we did the prediction. Then finally we got the probability. And then we finally checked whether the person is likely to churn or not. Right.

**Summary in Points:**

(i) Loaded all the pickle files (model, encoders, scaler, etc.).

(ii) Took the input data and converted it into a DataFrame.

(iii) Applied One-Hot Encoding for the Geography column.

(iv) Applied Label Encoding for the Gender column.

(v) Combined all encoded features to form the complete DataFrame.

(vi) Applied scaling on the features using the saved scaler (transform).

(vii) Made predictions using the trained model.

(viii) Obtained probability scores for the prediction.

(ix) Checked churn likelihood (whether the person is likely to churn or not).

So yes, uh, this was it. Uh, now in my next video, what I'm actually going to do is that I'm going to deploy this entire content with the help of Streamlit in the Streamlit platform itself. Right. So I will see you all in the next video.

#### 5. Integrating ANN Model With Streamlit Web APP

We have trained our entire an, uh, artificial neural network. We got the loss. The loss was also very less. We saw the loss in our TensorBoard, uh, visualized the entire diagram. We saw various parameters. Then we went ahead and did the prediction. We saved all the pickle file. We loaded all the pickle file. And, you know, we could also see the prediction was also happening. Everything. We did it.

Now it's time that we create one app.py file. And this is what I'm going to specifically use as my Streamlit web app. I'm going to probably create an end to end web app, where I will be able to give the inputs and do the prediction from there. So let's go ahead and here is my app.py. So first of all I need to import all the libraries that I really require. So one is streamlit as st, numpy, tensorflow, sklearn.preprocessing. I'm going to use standard scalar, labelencoder, onehotencoder. So whatever things we specifically did in the prediction dot ipynb file, we'll be doing it. Okay.

So the first step is loading the trained model. So I will go ahead and write TF dot keras dot models dot load underscore model. And I will be giving my model dot h5 file name. Um, then what we will do is that we will go ahead and load all our encoder, scaler, um, you know, one hot encoder, all the files itself. So here you have one hot encoder geo dot pickle file. And I'm encoding this. So here you can see one hot encoder underscore geo dot pickle I am loading this in the read byte mode. Then I have this pickle dot load. Then I'm also loading this scaler. Right. So all this pickle is specifically getting loaded right now.

Let's quickly start with the Streamlit app. So here I'm going to use a Streamlit app because I don't have to worry about HTML and all. So that is the reason I'm using this. So I'll go ahead and write this st dot title. Let me go ahead and write customer churn prediction. Okay, so customer churn prediction I am going to use over here. Customer churn prediction. So this will basically be my title of the Streamlit app.

Now see, you know that how many types of inputs you are specifically giving. You are giving geography, gender, age, balance, credit score, estimated salary, tenure, number of products, has credit card, or number of uh, uh, number of products already mentioned. Right. Is active member. So these are some of the features. So based on these features I need to provide my input from the Streamlit web page. So for this we will go ahead and create all these inputs. So for geography I'll be using a drop down which will be specifically using all the categories. So categories of zero, whichever is the categories of zero, by default it will get loaded over here. It can be zero, one, two or any name that we are going to give over here.

Then you have this st dot select box. Whatever values or with respect to the classes that I have, I'm actually able to get it over here. Um, then you'll also be able to see a st dot slider. I'm using a slider for age. I'm using an input for balance, credit score, estimated salary. Estimated salary is again number input. Then for tenure again I'm using a st dot slider between 0 to 10. Uh, I'm also using a slider for number of products and similarly has credit card and is active member. Okay. So this is what is the input data I will be giving from my user input.

Then I will prepare this input data by taking it in some kind of dictionary. Right. So we will go ahead and prepare this dictionary. So here I have my credit score, I'll take this data. Gender I'm going to use this labelencoder gender and use transform and probably get the zeroth value. Then age is here, tenure is here, balance is here, number is here. And each and everything, all the informations are specifically here. Okay.

Now, uh, one more information that is missing is basically our one hot encoder, which is my geographic column. So for geography I will try to encode it in a different way, like how I did it in my prediction dot ipynb file. So I will just go ahead and encode this geography. So here is my geo encoded. I'll take this geography value, which is from here. Right. This geography value I will take this in the form of list. I'll convert this into an array by using one hot encoder underscore geo dot transform. Then I'm converting back into a data frame by taking the feature names over here. Right. So, uh, this is what I am actually specifically getting for all my values.

Now I will combine this one hot encoded with my input data. Right. So for that again we'll go ahead and add pd dot concat. I'm saying input data dot reset index drop is equal to true. Then take the geo underscore encoded underscore df with axis is equal to one. Same thing what we did in the prediction dot ipynb file. Right. Nothing is different. Then we are going to take this entire data and we are going to scale it also. Okay. So here I'm going to probably go ahead and scale it. Okay. Then let's go ahead and do the churn. And with respect to the churn, I will just go ahead and write like this: if prediction churn, whatever prediction probability I'll get, I will just go and compare and say, hey, if this is greater than 0.5, I'll go ahead and write it in my Streamlit app. Otherwise I'll go it over here and say that, hey, it is not likely to churn. Okay.

So yes, this was it. I think everything was discussed already in the prediction dot ipynb file. So let me quickly go ahead and run it. Okay. So now this is the first time in order to run a Streamlit file, I'll just go ahead and write Streamlit run app.py. Okay. Streamlit. Streamlit. Okay. Streamlit. So once I execute it, I will just say allow access and let's load it. So my file is getting loaded. Let's see. I should not be getting any error now. Right now here you can see the customer is not likely to churn. It is coming up over here. Let me print this particular value also over here, right, what probability we are getting. So I will say hey go ahead and just print this. So till here what I will do I will just go ahead once I get the probability right I'm just going to go ahead and print it over here.

So now let's go ahead and see it and let's reload it. Okay. So here you can see churn probability is zero zero. So if I keep on increasing age is this increasing. See churn probability is increasing. Now you can add up any values. You want to select Germany? Go ahead and select it. Add any balance like 10,000. Go ahead and add it. Here you can see churn probability automatically. The calculation will happen. If I go ahead and add one credit card, one active member, so see this value is also decreasing. So on the runway it is probably predicting. My model is entirely predicting it right now.

Uh, this was a very simple end to end project where I've used customer churn prediction. And, uh, what you can do is that you can select different, different values and do the prediction automatically. This will do the prediction. Now quickly I will go ahead and open my GitHub. Okay. And, uh, with respect to GitHub I have to probably go and see the deployment. Right. So what I'm actually going to do is that first of all I'll go ahead and check in the code. Uh, you know how to do the check in with respect to the code. You can also use git init. But, uh, this project, uh, I will just try to quickly upload it in my GitHub repository. So I'll just go ahead and create my GitHub repository. Let's say I will go ahead and write ann classification churn. And here, uh, I will just go ahead and create this. Add a Readme file, General Public License. Go ahead and create a repository. Okay.

So once this is created now in my next step what I will do I will go ahead and upload all the code over here. And then we will go ahead and see how we will be doing the deployment. This into a Streamlit app. Okay. Streamlit cloud, that is what we are going to see in the next video.

### Summary:

(i) Model Training & Visualization - Trained the Artificial Neural Network (ANN), observed low loss, and visualized training/parameters in TensorBoard.

(ii) Prediction Setup - Performed predictions using the trained ANN, saved pickle files and model, and reloaded them for validation.

(iii) Streamlit App Creation - Created app.py using Streamlit for building the web app interface.

(iv) Library Imports - Imported streamlit, numpy, tensorflow, and sklearn.preprocessing (StandardScaler, LabelEncoder, OneHotEncoder).

(v) Load Trained Model - Loaded the saved .h5 ANN model using tf.keras.models.load_model.

(vi) Load Encoders & Scaler - Loaded OneHotEncoder for Geography, LabelEncoder for Gender, and Scaler from pickle files.

(vii) Streamlit Title - Added title "Customer Churn Prediction" in the app.

(viii) Input Features UI - Created input fields for Geography (dropdown), Gender (select box), Age (slider), Balance (number input), Credit Score (number input), Estimated Salary (number input), Tenure (slider), Number of Products (slider), Has Credit Card (slider/input), and Is Active Member (slider/input).

(ix) Input Preparation - Collected all inputs into a dictionary, applied LabelEncoding for Gender, and OneHotEncoding for Geography.

(x) Data Transformation - Combined encoded geography with other inputs using pd.concat, then applied scaling.

(xi) Prediction Logic - Used the model to predict churn probability; compared against threshold (0.5) to classify churn or not.

(xii) Output Display - Displayed churn probability and prediction result in the Streamlit app.

(xiii) App Execution - Ran the Streamlit app using streamlit run app.py.

(xiv) Interactive Prediction - Verified predictions interactively by changing input values (age, balance, geography, etc.) and observing churn probability updates.

(xv) GitHub Setup - Created a new repository (ann-classification-churn) with README and license.

(xvi) Code Upload - Planned to upload code (model, pickle files, app.py) to GitHub.

(xvii) Deployment Plan - Next step is deployment of the Streamlit app to Streamlit Cloud.

## 6. Deploying Streamlit web app with ANN Model

Uh, now in this video we are going to see how we are going to deploy this entire project in the Streamlit app. So Streamlit is a faster way to build and share data apps. So I have already taken a module, uh, regarding Streamlit, which you will be able to find out in this course.

So let's go ahead. And first of all, sign in over here. Uh, once I sign in, I'll be getting this share.streamlit.io. And here you can probably go ahead and create your own app. Right. But before that I will go ahead and upload all my code in my GitHub. Right. So here let me do one thing. Let me go over here. Let me reveal in the File Explorer okay. So this is the file I will quickly go ahead and deploy some of the files. Right.

So I will take this this. Okay. This this this this all the pickle file along with the requirement dot txt. So you can even commit it directly from the GitHub using the command line. But uh, I'll just directly copy and paste it over here. And I hope everybody knows that how to do it right. Because already GitHub session is already spoken about it. Right. We have so we have uh discussed about it. Right. So I'm uploading all this files quickly. And here I also have my requirement dot txt. So let's go ahead and put this requirement dot txt over here okay.

So yes uh this commit we will be doing it okay. I have to just go ahead and reload it. Just a second okay. So here you can see processing your files and all the files is over here you have your Readme file app.py pickle file everything. Now let me go over here. So here you can see that, uh, you have your share dot streamlit.io.

Now Streamlit allows you to probably create your free apps just by integrating with the GitHub. So I will just go ahead and create click on create app. It is asking me do you already have have an app. So I'll say yes I have an app. It is in GitHub repository. So I will just go and search for this GitHub URL and I'll paste it over here okay.

Once I paste it. So branch is mean let me just go ahead and say mean file path. It should be app dot Pi. And this is the URL that it is basically creating okay. So step by step we are doing it okay. And classification churn this particular dot Streamlit dot app. So I will go ahead and deploy it.

So there is also something called as advanced token. Uh if please provide environment variable and other secret key to your app using HTML file format. The information is encrypted and served security during your runtime. Learn more secrets in JS. If you have any secret key, you can probably mention it over here and save it. Okay. Uh, that also we will be seeing, uh, as we go ahead. Okay. How to probably use some secret keys. Now I'll go ahead and click on deploy.

Now once I do the deployment automatically see requirement dot txt all the files is there. This will actually do the deployment, do all the installation that is required and it will create the entire environment over here. And my entire platform will be ready just in another one 30s to 45 seconds. And then we are good to go to probably check out our application.

Um, so let's wait for some time. The reason why we really need to wait for some time is that because it will be doing the pip install requirement, dot txt all those things. Um, I'll also show you an example how to probably do it with the environment variables. But right now let's focus on this and let's see how quickly we will be able to do it okay.

So it shows your app is in the oven so it is getting warm. In short your entire server is getting warmed. So we'll wait for some time uh, till this installation takes place. And our platform is once ready. And this is the URL which you can also use it and you can also access it.

So I think now my app is ready. And here you can see it's running stop share. Everything is probably displayed over here. I think it should be displayed in short over here. So let's see. Yes. Perfect. This is my entire app. And right now it is in the cloud of Streamlit.

Uh, now, if I go ahead and select any value that I want, you know, it should be able to directly compute things, right? Directly just by selecting values. It should be able to compute. And this entire thing is working with the help of an okay. You can select any of the values. And then you'll be able to get the answer. And this is with respect to uh, just a runtime. Right.

So I hope you like this particular video. Uh, this was it. And this was one amazing end to end project using an, uh, I'll see you all in the next video.

### Summary:

(i) Streamlit Overview - Streamlit is a fast way to build and share data apps, accessible via share.streamlit.io.

(ii) GitHub Preparation - Uploaded all project files (model .h5, pickle files, app.py, requirements.txt, README) to GitHub repository.

(iii) Streamlit Integration - Logged into Streamlit, selected Create App, and linked GitHub repository.

(iv) Deployment Settings - Selected branch (main), set file path as app.py, and confirmed app URL.

(v) Environment Variables - Streamlit provides option to add secrets/environment variables securely (not used in this step).

(vi) Deployment Process - Streamlit automatically installed dependencies from requirements.txt and set up environment.

(vii) Waiting Phase - Deployment took ~30–45 seconds while Streamlit processed installations and warmed up the server.

(viii) App Ready - Application was successfully deployed to Streamlit Cloud with a live URL.

(ix) Testing - Interactively tested the app by selecting different input values; predictions were computed instantly.

(x) Project Completion - End-to-end ANN customer churn project successfully deployed and accessible via Streamlit Cloud.



#### C) End to End ANN Project - Regression

**1. ANN Regression Practical Implementation:**

So now let's go ahead and probably implement this regression problem statement. So what I will do I will go ahead and write over here. And inside this folder only I will be creating this because we are going to follow the same steps. So I'll go ahead and write "regression regression ipynb". I'll say "churn regression" or I'll say "salary regression". Okay. So this will be my file. I will go ahead and quickly select my kernel okay.

Now first of all, what are the steps that we will do. So initially we will go ahead and import all the necessary modules that is required like "train_test_split", "StandardScaler", "LabelEncoder", "OneHotEncoder" and "pickle". Okay. So we'll go with the first step over here. Then, uh, after this we will be reading our CSV file. So I'll go ahead and write "read_csv". And let me just go ahead and write this as "churn_modeling.csv" file okay.

Now if I go ahead and write "data.head()" I will be able to get my input over here. And this is my entire data set. Right. As I said "EstimatedSalary" will be my output feature and remaining all will be my independent feature. Okay. And that is the reason the output feature is a continuous value over here. So we are considering this as a regression problem statement.

Now after this, the next step that what I will do is that I will delete this "RowNumber", "CustomerId" and "Surname". Whatever things we did in the classification problem, almost the same way, we have to probably do this. Okay, so I have executed this. Then we will go ahead and convert this gender from Female/Male to zeros and ones because here I am going to use my label encoder.

So for that I will go ahead and create my variable "label_encoder_gender". I will initialize it to "LabelEncoder". And then I will go ahead and write "data['Gender'] = label_encoder.fit_transform(data['Gender'])". Okay. Similarly I will go ahead and do it for the "Geography" column also where I'm going to use one hot encoder. Again I'll use "fit_transform", I'll convert this into arrays. I will get all my values, then convert this into my DataFrame along with my features.

Okay, so if you go ahead and see your "geo_encoded_df", you'll be able to see, hey, I'm getting these three features. Okay. Then, uh, we are going to combine this one hot encoded columns with my original data step by step. So finally you'll be able to see "data.head()". And we are also dropping the "Geography" column. Now I don't have my geographic column. And if you probably go ahead and see my this three columns have got added. Perfect.

Now we are going to split our data into features, into our independent features and target feature, right, dependent feature also. So I'm dropping "EstimatedSalary" from this. And all the remaining features will go in X-axis and my "EstimatedSalary" will go in the Y-axis. Okay. So this is done, a bit of processing.

Now I will save all the pickle files specifically for transformation like "label_encoder.pickle", then you have "one_hot_encoder.pickle". Then one more pickle that we are going to probably create is "scaler.pickle". But, uh, the scaler pickle we will specifically use only when we try to, uh, you know, do the train test split. So let's go ahead and do the train test split also.

So here I will take this code quickly. I will just copy and paste it so that I don't have to waste more time on this writing the entire thing, because we are doing the same thing. If I was doing something else, then I could have probably done that. Okay, so here I'll go back to my salary regression quickly. I will go ahead and write code now. Quickly I'll paste it over here. Okay. "train_test_split".

And now I will go ahead and apply my "StandardScaler", scale this feature quickly. Salary okay. And here I will go ahead and do this. And this is done right okay I'm getting an error. Could not convert string to me. Why. Because let's see my X value okay. Gender is still Male and Female. Why why why why. Because data of gender it is okay. Fine "data['Gender']". So if I replace this, what is my data over here? If you see the data, let me execute this once again, I think I have missed something. So I'll quickly execute this. I'll remove these three. I'll see this.

Now let's go ahead and check my data. So my data is basically having zeros and ones. I think I missed one line of code in executing it. Then I'll do the concatenation. And finally I will do the concatenation. This has got executed, I have gender zero and 1X1Y. I get it and I'm doing the train test split. And now it should do the scaling okay so I missed one code. So that is the reason that was the issue.

Now we'll go ahead and save all the pickle files. "scaler.pickle" file, "one_hot_encoder_geo.pickle" and "label_encoder_gender.pickle" file. Okay. Now it's the time to train our ANN with regression problem statement. Right? How to train our ANN over here. That is what we are basically going to discuss. And this is the important thing. Remaining things are almost same. We do not have to make any changes okay.

Now let's go ahead and quickly do this. So first of all again I will import "tensorflow.keras". Okay. And then we are importing "Sequential" and "Dense". So first of all we'll go ahead and import this. So once this is imported we are good to go and work with sequential and dense. And we are good to work with our entire neural network okay. So here we are executing this. Then quickly let us go ahead and do this okay.

So we will go ahead and build our model again. To build our model we will use the same technique. Like first of all I will say hey go ahead and take this "x_train.shape[1]" as input shape. This will be my input layer. Then we are going to apply an activation function. The first hidden layer will have 64 neurons. The second hidden layer will be having 32 neurons, and the activation function will be ReLU. And the dense will basically have one. Okay.

### We don't apply any activation function; Default is Linear Activation function for regression is applied

Now when we are using one here, we are not applying any activation function. When we don't apply any activation function, the default activation function that will be applied is called as "linear" activation function. Now linear activation function is specifically for regression. That basically means I'm not doing anything. Instead I'm getting whatever output I'm getting, I'm using that. Okay.

### Next we are compiling the model

Now after this we will go ahead and compile this model okay. So we will go ahead and write compile the model. Now for compiling the model I will go ahead and write "model.compile". And here, uh, first of all I'll go ahead and write my optimizer. Let's say the optimizer is Adam. For this case you can also go ahead and initialize it from "tensorflow.keras", but I'm directly writing the keyword. Then my loss. Now this is what is the main thing that actually changes in regression. In loss we will basically use "mean_absolute_error". Okay. You can also use "mean_squared_error". But here we are just going to use mean absolute error. It is up to you.

And here the metrics that we are specifically going to use is "mae". Okay. Again you can see the Keras documentation what all loss will specifically be happening over here. So if I go ahead and open Google and if I just search for "Keras loss functions", okay. So if I just go ahead and click on this. So it shows so many different ones — for classification problems you have "binary_crossentropy", "binary_focal_crossentropy", "categorical_crossentropy". Then you have binary cross entropy functions, all these things. Right. For regression you have mean squared error, mean absolute error. See this? This is what I have used. Right, mean absolute error. You can also use mean squared error.

And already I have spoken about this, uh, even in machine learning, if you know machine learning, I think there all these things will be probably covered up, you know, so you need to have that basic knowledge with respect to all using this kind of loss function. Okay. So now what I will do I will use a metric as "mae". And then finally let's go ahead and display the summary. So I'll just go ahead and write "model.summary()".

### When we display the model summary then here you can see this is my input. So the total number of weights and bias are nothing but 2945. These are the trainable parameters that we have right. Then I have dense one, dense two. And finally my output is over here okay.

### Now let's go ahead and train this particular model. Again for training this model I will go ahead and set my logs okay. So here are some of the logs I'll say this is my regression logs I'm using early stopping and TensorBoard as you all know. So we are going to use both of them so that this folder will get — it will just try to, whatever logs is basically coming during the training, it will be coming over here. Then we are going to set it to early stopping also.

Now in the case of early stopping I will be using validation loss. Like, uh, whenever we are training our neural network you will be getting training loss, training accuracy, validation accuracy and validation loss. But I really need to monitor the validation loss. And I'll keep the patience value to ten. That basically means I'm going to see it for ten epochs. Okay.

And finally we will go ahead and train the model. Now in order to train the model I need to use this callbacks, so early stopping and TensorBoard callback. I'm giving my "x_train, y_train", epochs will be 100 and all. Everything is there, right? So let's go ahead and train this, and let's see how much will be the answer that I will be getting.

So epoch one, here you can see mae, uh, these all values are there. Validation loss, see here also values are coming, right. And these are keep on decreasing. Unless and until it keeps on decreasing this epochs will be going on right. So the validation loss also you'll be able to see that it is decreasing okay. See validation mae. Validation loss unless and until it is not decreasing, right, if it is maintained in a similar manner, hardly there is not much difference, then it will stop. So here you can see in epoch 47 it has stopped because till here the loss was going down. But after this it was almost similar. Right.

So, uh, now my model is almost trained. Again, this is, uh, I'm actually doing it in my local machine. Again, I have a powerful CPU, so I'll be able to do it very much fast. Now let's go ahead and evaluate the model. Okay.

Now, before that, let me just go ahead and load my extension TensorBoard. Okay. So I will just go ahead and execute it. Then I will say, hey, go ahead and take this TensorBoard and my directory will be "regression_logs/fit". Let's see this particular folder. So here you have regression logs slash fit okay. So slash fit will be mine. So once I execute it here you'll be able to see that we'll be able to load the TensorBoard.

### Now in this TensorBoard all the information regarding how the training has happened and all. So I'm getting some error. So this should be directory I guess. Let's see over here. Okay. Again, you do not need to byheart it. You can see the documentation and you should be able to see this. Okay. You may be thinking, Chris, you don't know all these things. Uh, I'm not, uh, I will definitely not byheart. So you should also not do that. Okay.

So this should be the log directory. And here should be my folder okay. Now I think it should get executed. Now we are launching the TensorBoard. Now this is good enough. It will take all the events from inside this. And it will display it to you. Okay.

And let's see it is taking some time to load it quickly. Let it get loaded. I think for the first time I have to execute two times, but it does not happen in Google Colab. In Google Colab, in the first instance only you will be able to run it. Okay. So this is really, really good. So here I have got my entire information which is amazing in the TensorBoard. Now see see see this entire things right now if I go ahead and see with respect to regression epoch versus loss. Wow these graphs look amazing. See it is going down down completely right.

## So loss is coming down. If you see mean absolute error this is also coming down I think this is an amazing graph and this looks good. That basically means your epoch is going down, that the model is really performing well. Then here you have evaluation loss versus iteration. Iteration by iteration, how much it is basically getting reduced. Then you also have mae versus iteration. This also you can actually check it out. Right.

Um, if you just want to see the validation. Validation has this epoch loss and all. So here you have your mae. You can probably zoom in and zoom out. And based on this you can check. You can also do something like this and check it out. Right. So all this information from the logs you are able to see it okay.

Now let's do one thing. So, uh, if you see mae and all, we will also try to evaluate this model for our test data. So let's go ahead and evaluate model on the test data. We're just going to check the accuracy, how good it is. So I will just go ahead and compute my "test_loss, test_mae". I'm just going to write "model.evaluate(x_test, y_test)". Perfect.

So now I will just go ahead and print, and I'll use an f-string. And I'll say hey this is my test mae which is nothing but it will be test_mae. So here you can see 5016 and more near to zero if it is going on, we can probably say that our model is pretty trained in an amazing way. Okay.

And, uh, this is done. And what you can also do is that, uh, um, along with this, I can also go ahead and save my "regression.py" file. So I will say, hey, go and save this as my regression model "regression_model.h5" file okay. So once I do this, this entire model will get saved. And I have my regression model.h5 file. Right.

So almost we have done each and everything right now. If you want to deploy this you have all the h5 files. You have all the training that is been done. Now you can also go ahead and deploy it. So for deployment purpose, uh, what I am doing is that I will give you the entire code. Again, with the help of Streamlit, you can deploy it. Let me just go ahead and create a file over here. Let me say "streamlit_regression.py". Okay.

So I'll give you the code over here so that you can go ahead and do the deployment. And it's very simple. It's almost like this only. So here you can see all the things and estimated salary. You can basically go ahead and compute it okay. See we are loading it. We are creating some user input box. We are converting this into a DataFrame. We are doing the encoding. Then we are predicting the salary okay. You can do this entire thing okay.

Oh, let me see one more model, huh? "regression_model.h5" file. Okay. Perfect. So I hope you understood this particular video. Again, it's more about how much you practice and, uh, how easy, uh, it was probably to train the entire ANN, that is artificial neural network, for classification, for regression.

Now, one of the most common questions that has been asked is how do you decide that, how many hidden layers or hidden neurons you need to take? We need to discuss about that also, because that is something very important. So in this case, it is very, very much difficult to just say that, hey, you probably need to just use this many number of hidden layers or this many number of hidden neurons. That is not possible.

So what we do is that we basically apply a lot of hyperparameter tuning, like with respect to the number of hidden layers and with respect to number of hidden neurons. And that is what I will be showing you in the next video, like how you can also come to a conclusion and I'll show you one type of hyperparameter tuning, how we can specifically do so. Yes. Uh, this was it from my side. I hope you like this particular video.

### Summary:

(i) Project Setup - Created new Jupyter Notebook salary_regression.ipynb for regression problem.

(ii) Data Import - Read dataset (churn_modeling.csv) and selected EstimatedSalary as target (continuous).

(iii) Data Cleaning - Dropped unnecessary columns (RowNumber, CustomerId, Surname).

(iv) Encoding -Converted Gender using LabelEncoder, Encoded Geography using OneHotEncoder, created new DataFrame, and merged.

(v) Splitting Data - Defined features (X) and target (y = EstimatedSalary).

(vi) Pickle Saving - Saved transformation objects: label_encoder_gender.pickle, one_hot_encoder_geo.pickle, scaler.pickle.

(vii) Train-Test Split & Scaling - Applied train_test_split and standardized features with StandardScaler.

(viii) ANN Model Building -

(a) Input layer: x_train.shape[1].

(b) Hidden layers: 64 & 32 neurons, activation = ReLU.

(c) Output layer: 1 neuron, activation = Linear (default).

(ix) Compilation - Used optimizer = Adam, loss = Mean Absolute Error (MAE), metric = MAE.

(x) Model Summary - Displayed trainable parameters (~2945).

(xi) Training - Used EarlyStopping (monitor validation loss, patience=10) and TensorBoard for logging; trained up to 47 epochs.

(xii) TensorBoard Visualization - Plotted loss & MAE curves, confirmed decreasing trend (good performance).

(xiii) Evaluation - Evaluated on test data: Test MAE ≈ 5016 (closer to 0 = better performance).

(xiv) Model Saving - Saved trained ANN as regression_model.h5.

(xv) Streamlit Deployment - Created streamlit_regression.py to:

(xvi) Load model & encoders.

(a) Take user inputs.

(b) Encode & scale inputs.

(c) Predict Estimated Salary.

Key Insight - Number of hidden layers/neurons cannot be fixed; requires hyperparameter tuning (to be covered in next video).



#### D) Finding Optimal Hidden Layers And Hidden Neurons In ANN

Now finally, we are going to discuss about a very important problem statement or question that comes in everybody's mind.Whenever you are implementing any application with deep learning neural networks, that is determining the optimal number of hidden layers and neurons for an artificial neural network.Okay, now this can be really challenging and it often requires experimentation. However, there are some guidelines and methods that you can help in making an informed decision, right?So here are some pointers. In order to probably come to a conclusion that how many optimal number of hidden layers and neurons you should probably use for an Ann.

Um, so the first point over here, it says that please go ahead and start simple. Begin with a simple architecture and gradually increase complexity if needed. Okay.

The second thing is that you can also perform grid search or random search. I hope everybody knows about this hyperparameter tuning using grid search or random search to try different architectures. So you can make different architectures. And with the help of grid search or random search, you can actually do it.

Then you also have cross validation. Use cross validation to evaluate the performance of different different architecture, like how we used to do in machine learning.

### And here is one more approach which is called as heuristics and rules of thumb. Some heuristics and empirical empirical rules can provide starting points, such as the number of neurons in the hidden layer should be between the size of the input layer and the size of the output layer, right between that particular size. That many number of hidden neurons can be there in the hidden layer.

Okay. And a common practice is to start with one two hidden layer. So, uh, obviously there are so many statements that I have discussed about, but, uh, we will try one approach where we will also perform hyperparameter tuning. Um, and uh, the hyperparameter tuning will be done in such a way that I will take various number of neurons, various number of hidden layers, and then we will try each and every architecture and try to see that which is the best model, which is probably coming out of it.

Okay. So, uh, first of all, what I am actually going to do for this, uh, I will be using one very important library, which is called as Keras. Okay. So here instead of TensorFlow, I will also use one more library which is called as Keras, because in this Keras there is one very important module which is called as Keras classifier. Okay. And that actually helps you to perform this entire hyperparameter tuning, uh, you know, just to find out which is the best model.

So I'll quickly go ahead and install this library. Okay. So I'll go ahead and write
"pip install -r requirement.txt".Oops that H5 file it it went ahead you know. So requirement dot txt. So let me just go ahead and do this. And I hope uh, I should be able to get the installation of Keras. Let's see. Right now it has not given me anything.But uh let's go ahead and import things. So here I will be importing some very important libraries. One is Gridsearchcv. See, one more additional library that we are going to use is Gridsearchcv standard scalar Labelencoder Onehotencoder is done. We are also going to import pipeline. Then we are going ahead with Keras classifier which is present in "keras.wrappers.scikit_learn". Along with that, we are going to import TensorFlow TensorFlow Keras model for sequential, dense and early stopping.Okay, so this is almost same. Nothing changes over here. But I think I'll be getting an error over here. Let's see if the installation is not taken place okay. So here you can see uh model not found. No model "keras.wrapper". So let me go ahead and see and see in the Keras whether we have this wrappers or not okay, I don't know. We need to probably check it out.

Okay. "pip install -r requirements.txt". Okay. "keras.wrapper". So let me just go ahead in Google and search for it. So. "pip install keras.wrappers" or scikit learn. So, uh, for this here. You have this particular model okay. So I will go ahead and use this model.So always remember whenever you find this kind of errors please make sure that you have a good sense wherein you will be able to find out which library we are talking about. So let's see. "scikeras" I think if this is if this is the library that we are searching for, I think it should work. Okay. But if it does not work, then again we'll look for a solution, but we'll look for a solution together.Okay. So here you can see it has been successfully installed. Now I think it should work okay. Or let me restart the kernel. Okay. So let me see. I think we should not be getting an error now. Okay, we are still getting an error. So let's search for this itself. Right. Let's search for this.

So here what I will do I will go ahead and open my browser. I'll search for "no modules keras.wrapper". Okay. Um. Oh, so here it says this works for me. Another approach you can try. Uh, okay. It comes with TensorFlow 2.12 okay. Um from "scikeras". Okay. I can also use this. Let's see I have also installed this right. Instead of using this I will use this. But I think now this should work okay.So I've used scikeras. And there I have actually done this installation okay now this is done. Now the next thing will be that I will go ahead and quickly import our pandas. Uh, use pandas and read our CSV file. And here I'm actually going to write
"churn_modeling.csv".

Here you'll be seeing how I specifically experiment with things. Okay so here I'm actually going to get my data okay. Then you know what. All things you really need to do okay. So I will just try to copy and paste it over here. And I hope I don't have to make you understand I'm going to apply label encoder one hot encoder. Then this is my entire data frame. Then we are concatenating. We are dividing. We are getting the independent and dependent features. And after this we are also going to do the train test split okay. Because this is not that important. Because this all things we have repeated again and again. Okay.

And then finally we are dumping all the pickle file or we are saving all the pickle file so quickly I will copy and paste it over here. Let me go ahead and write "import pickle" because I will also be requiring import pickle okay. Done. So let's go ahead and execute this. Now here you will be able to see my pickle file will get created. So this has got executed.
Now we are going to create a function. Now see here is where I will write my coding completely from scratch. So define a function to create a model okay. And try different parameters. Try different parameters. Okay. Uh, and here we are specifically going to use Keras classifier okay.So I will go ahead and write
"def create_model(neurons=32, layers=1):"Okay. Uh, here, uh, what I'm actually going to do is that, um, by default I will be I will be providing two parameters whenever we create a model. One is how many number of neurons are required. Let's say by default I'm giving 32 neurons if I'm not passing any value and how many layers I want with respect to the hidden layer. By default I'll give one hidden layer. Okay.So I'll create this function and I'll say, hey, go ahead and create
"model = Sequential()".Okay. And then what I can do on top of this particular model I can go ahead and add one dense layer. So I'll say hey go ahead and add our dense layer."model.add(Dense(neurons, activation='relu', input_shape=(x_train.shape[1],)))"

Then here you'll be able to see that I'm also going to use my input shape is equal to x_train.shape. Please observe very carefully what I'm actually doing over here okay. So this is my input shape. So in the first layer I'm basically adding this.Right now I need to probably check. Do I need to add more layers or not okay. If I give 2 to 3 layers then we should also add it right. So based on this parameter I'll write a generic code. Right?

I'll say hey, if I'm giving 2 to 3 layers, if I'm giving three layers also along with this first leaf, this first layer hidden layer will be there. Okay. And uh, here you'll also be seeing that at least this will be there with, with respect to the number of neurons that we have. Okay.Then a generic representation, I'll say hey
"for _ in range(layers-1):
  model.add(Dense(neurons, activation='relu'))"so this is just by for loop. We will be keep on adding this particular layers.

Finally our output layer I will go ahead and write"model.add(Dense(1, activation='sigmoid'))".So I'm basically going to add this. And finally I'll go ahead and write
"model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])".And finally, I return the model.

This is what we are doing inside this function. Okay. So whenever I usually call this particular function I'm just going to give this parameters uh neurons and layers. And based on that I need to add how many number of hidden layers and all first layer is added by default. And we have also added output layer by default. We have also added uh, which compiler, which optimizer and loss we are specifically going to use. Okay.Now we will go ahead and create create a Keras classifier. Okay. So I'll go ahead and write. Hey model is equal to

"KerasClassifier(model=create_model, epochs=50, batch_size=10, verbose=0)"Some information with respect to this. Now this is fine with respect to my model. Uh my model is a Keras classifier, which is going to probably call this function and create an entire model. And then you need to run the model needs to be run for 50 epochs. Batch size is ten. And all all the information now is very much important thing.

So here I will go ahead and define my grid search parameters in the key value pair. So this will be neurons like neurons which how many neurons I want to play with every layer. It can be 16, 32, 64, 128. Layers I will go ahead with one, two, three. If you want to add more, you can go ahead and I will play. I'll keep on playing with each and every parameter over here.

So let's go ahead and uh, let's remove the batch size because I just want to play with one batch size that is ten okay. So that I don't have to probably play with more things. And my training will also happen fast. This is just to show you and epochs. I'll play with 100, 500.Right now Gridsearchcv is going to probably take this entire model. It is going to take this entire power grid, and it is just going to perform this cross validation that we are given. And once we write "grid.fit", I'm probably going to get the best number of parameters.So I'll probably after the fit operation happens we can just go ahead and print it. I can write
"grid_result.best_score_" and "grid_result.best_params_".Parameter will be like how many neurons, layers or epochs I need to probably take place. Okay. So here our training will start and Gridsearchcv what it does. Every combination it will probably take, it will go ahead and see what loss is there. And whichever has the highest accuracy it will take that particular parameter.

So let's go ahead and execute this. So it is going to take time uh that fast. It is not going to happen all these things because here now the combination is many right.Uh okay I did not okay. I had to probably print it. Uh, at least will not be able to see it. Okay. Over here. So, uh, that is the reason it is just showing all these things.
Uh, okay. I'm getting some error. So, guys, the error that we, we got. Right. So what I will be doing is that along with this parameters, I will also be adding this "layers=1" and "neurons=32". Because in this create model I'm actually passing this particular values. Right. Neurons 32 and layer one. Right. So that parameters. Also I'll be passing in my Keras classifier along with this.Uh, I also saw one more issue that um, is with respect to let me just go ahead and have a look, uh, where it is regarding binary cross entropy. So how many time I have basically written this? Okay. So I think now it should work fine. Let's run this. And finally I'll also run this okay.Now the entire fitting will specifically happen. And here you can see three folds for each of 16 candidates. Totally 48 fits. That is basically going to happen. And um, now the training has already started. I think, uh, you know, uh, once the epoch starts with respect to this, we should be able to see more information.

Um, but, uh, just by using this "verbose=1", we'll be able to see some information otherwise. And right now when I wrote this "n_jobs=-1". Right. It is probably going to use all the cores out there okay of my system. But again uh, if you are trying to use do this in your local, uh, and if your local is not a good system, then it may take some amount of time. Okay. Otherwise please go ahead and do it in Google Colab. You know, Google Colab is any point of time you will have some kind of GPUs that will be available.So what we'll do, we'll wait for some time and uh, then uh, we will continue the discussion.

So here you can see, oh perfect. The epochs has started and it's running. It's running like anything. See quickly. This is the powerful of uh, this is the power of a powerful workstation, right? I have a powerful workstation, I have GPUs, I have all these things, uh, done as a setup over here. And this is mainly to show you, like, how things work over here.

So here you can see accuracy is there, loss is there. And finally, the best accuracy that you got is nothing but 85% using epochs 100, layer 1 and neuron 16. So it is saying that, hey, just go ahead and use one hidden layer and uh, go ahead and use uh 16 neurons and just try to do it and try to find it. Right.So when you use this now you can go ahead and create your own, uh, custom entire ah, uh, your entire Ann, with this many number of layers. And by this you will be able to find out the best parameter.So I hope, uh, you were able to understand this once you get this information, then constructing an Ann becomes very much easy. Right.

So this is what we have actually discussed about determining the optimal number of hidden layers and neurons for an artificial neural network. Again you can try out with different different parameters, but we can use Keras classifier with respect to this.
