# Deep-Learning-Skills
This Repository details my Skills in Deep Learning

### Mlflow also Supports Tensorflow

## log the model using TensorFlow - mlflow.tensorflow.log_model(model,"model",signature=signature)

### We used a Tensor Model and Keras is a wrap on Top of it

space={
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-1)),
    "momentum":hp.uniform("momentum",0.0,1.0)

}

### From HyperOpt we used hp; It is like Hyperparameter Tuning
