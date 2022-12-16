# ML based automated disaster message classification and response generation pipeline

## Project Overview:
In this project, a machine learning pipeline has been developed to categorize a data set containing real messages that were sent during disaster events so that the messages can be automatically sent to an appropriate disaster relief agency. This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## File Distribution:


- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
- requirements.txt



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
