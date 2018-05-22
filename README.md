# Classification using word2vec and Logistic Lasso Regression 

Model determines 30 most common expressions for each class by counting word occurences and builds word2 vec matrix using Google News pretrained model. Each row of the matrix is one common expression. These matrices are pre computed and stored in pickle files
Same matrix is then build for user bio.
Euclidian distance is computed between user bio and common expressions for each class label. This serves as input to Logistis Lasso Regression model. Each class has its own model and coefficients are stored in pickle files.
Class label is the selected as maximum likelihood of outputs from all models. 
In general if output of model is positive, there is a chance that bio belong to the model. Confidence is therefore computed as portion of positive label predictions. 

## setup
set FLASK_APP=prediction_process
run pip install -e .

## usage
Accepts json post requests on port 5000 with route /predictions

expected arguments are:

  - 'bio': the text to be processed
  - 'follower_count': number of followers

Will return json with the following keys:

  - 'predicted label': label selected
  - 'confidence': confidence of selected label 

```
# Example cURL:
curl -H "Content-Type: application/json" -X POST -d '{"name":"John Smith","bio":"Passionate html com influencer founder and something else", "follower_count": "1234"}' http://localhost:5000/predictions
```

# LR_word2vec_classifier
# LR_word2vec_classifier
