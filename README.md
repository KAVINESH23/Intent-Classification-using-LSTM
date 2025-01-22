# Intent Classification using LSTM
   ## PROBLEM STATEMENT
        Classify the Intent category which can be used in chatbot like greetings etc..
   ## DATASET DESCRIPTION
     It contains 41,424 rows and 2 columns named as Text and Intent , where intent is the target variable contain 12 class 

   ## TRAINING DETAILS
     Training the intent classification  dataset with epoch as 10 and batchsize as 32 and the training accuracy is 100 percent and loss as 0.09.and validation accuracy also 100 percent .It implies that model is overfit to make the model generalization use dropout .The dropout part removes part of neurons and it reduces the training and validation accuracy  compare to before and achieve generalization.

    ## TECHNICAL STACK
       Frameworks and Libraries - It is built using Keras and TensorFlow
       Model Architecture - Sequential,Embedding layer (Words into vectors),LSTM for long term dependency of understanding the data,dropout layer to regularize the overfitting,Dense Layer to connect eachlayer and pass to  the activation function  
      Model Compilation -  Optimizer (Adam),Loss function(Categorical cross entropy) 
      Training - Epoch (No.of passes over data),Batch size (Samples per batch during training ),Validation spilit - To cross check the training accracy

   ## RESULT 
      Here the user given a prompt by using the saved model and the result is displayed
