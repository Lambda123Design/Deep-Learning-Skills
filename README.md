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
