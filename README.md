# ML based automated disaster message classification and response generation pipeline

## Project Overview:
In this project, a machine learning pipeline has been developed to categorize a data set containing real messages that were sent during disaster events so that the messages can be automatically sent to an appropriate disaster relief agency. This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## File Distribution:

1) app

master.html  # main page of web app
go.html  # classification result page of web app
run.py  # Flask file that runs app

2) data
disaster_categories.csv  # data to process 
disaster_messages.csv  # data to process
process_data.py
disaster_messages_database.db   # database to save clean data to

3) models
train_classifier.py
classifier.pkl  # saved model 

4) README.md
5) requirements.txt



## Steps to execute the codes:

 Clone the repository using the following command -
 
    git clone https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline.git    
     
 Install the necessary packages which are the required dependencies for the codes to run using the command -
 
     $ pip install -r requirements.txt     
 
 1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_messages_database.db          
    
    - To run ML pipeline that trains classifier and saves

    python models/train_classifier.py data/disaster_messages_database.db models/classifier.pkl            

2. Go to `app` directory: 
    cd app

3. Run your web app: 
     python run.py
     
- ## Results:

![webpage demo 1](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/webpage%20snippet%201.png)

![webpage demo 2](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/webpage%20snippet%202.png)

![webpage demo 3](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/webpage%20snippet%203.png)

![webpage demo 4](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/webpage%20snippet%204.png)

![webpage demo 5](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/webpage%20snippet%205.png)

![webpage demo 6](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/webpage%20snippet%206.png)


Some of the snippets during the run of the ETL pipeline and the model are shown below - 

![runtime image](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/run%201.png)

![runtime image 2](https://github.com/SohamBera16/ML-based-automated-disaster-message-classification-and-response-generation-pipeline/blob/main/screenshots/run%202.png)

