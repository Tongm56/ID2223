import os
import modal

def predict_until_one(model, data, max_attempts=100):
    for attempt in range(max_attempts):
        y_pred = model.predict(data)
        if 1 in y_pred:
            index_of_one = list(y_pred).index(1)  
            print(f"Prediction 1 found at index {index_of_one} after {attempt+1} attempts.")
            return index_of_one  
        else:
            
            pass
    print("Reached maximum attempts without finding a prediction of 1.")
    return None
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login(project="ID2223_23_lab1")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()

    index_of_one = predict_until_one(model, batch_data)
    if index_of_one is not None:
        y_pred = model.predict(batch_data)  
        offset = y_pred.size - index_of_one
        wine = y_pred[index_of_one]
    else:
        print("No prediction of 1 was found.")
        return  
    # y_pred = model.predict(batch_data)
    #print(y_pred)
    # offset = 10
    # wine = y_pred[y_pred.size-offset]
    if wine == 0:
       wine_url = "https://raw.githubusercontent.com/bokuan/ID2223/main/average.png?token=GHSAT0AAAAAACHL4DAOO2XQVNTSNYIPH476ZKY6VSQ"
    if wine == 1:
       wine_url = "https://raw.githubusercontent.com/bokuan/ID2223/main/high.png?token=GHSAT0AAAAAACHL4DAOMCZIVNSRPD3QTUK4ZKY6XCA"
    print("Wine predicted: " + str(wine))
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)
   
    wine_df_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_df_fg.read() 

    #print(df)
    label = df.iloc[-offset]["quality"]
    if label == 0:
       label_url = "https://raw.githubusercontent.com/bokuan/ID2223/main/average.png?token=GHSAT0AAAAAACHL4DAOO2XQVNTSNYIPH476ZKY6VSQ"
    if label == 1:
       label_url = "https://raw.githubusercontent.com/bokuan/ID2223/main/high.png?token=GHSAT0AAAAAACHL4DAOMCZIVNSRPD3QTUK4ZKY6XCA"
    print("Wine actual: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_wine.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different flower predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Average Quality Wine','True High Quality Wine'],
                     ['Pred Average Quality Wine','Pred High Quality Wine'])
    
        cm = sns.heatmap(df_cm, annot=True, fmt='g')
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different wine predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different wine predictions") 



if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f.remote()


