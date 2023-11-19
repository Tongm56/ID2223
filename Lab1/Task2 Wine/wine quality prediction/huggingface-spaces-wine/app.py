import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login(project="ID2223_23_lab1")
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(type,fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
             chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
             pH, sulphates, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[type,fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
             chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
             pH, sulphates, alcohol]], 
                      columns=['type','fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
             'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
             'pH', 'sulphates', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    if res[0] == 0:
       wine_url = "https://raw.githubusercontent.com/bokuan/ID2223/main/average.png?token=GHSAT0AAAAAACHL4DAOO2XQVNTSNYIPH476ZKY6VSQ"
    if res[0] == 1:
       wine_url = "https://raw.githubusercontent.com/bokuan/ID2223/main/high.png?token=GHSAT0AAAAAACHL4DAOMCZIVNSRPD3QTUK4ZKY6XCA"
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with a lot of features to predict wine's quality.",
    allow_flagging="never",
    inputs=[
        gr.Number(default=1, label="type"),
        gr.Number(default=7.1, label="fixed_acidity"),
        gr.Number(default=0.34, label="volatile_acidity"),
        gr.Number(default=0.29, label="citric_acid"),
        gr.Number(default=2.7, label="residual_sugar"),
        gr.Number(default=0.053, label="chlorides"),
        gr.Number(default=25, label="free_sulfur_dioxide"),
        gr.Number(default=121, label="total_sulfur_dioxide"),
        gr.Number(default=0.9958, label="density"),
        gr.Number(default=3.2, label="pH"),
        gr.Number(default=0.50, label="sulphates"),
        gr.Number(default=9.7, label="alcohol"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

