import os
import modal

LOCAL = False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_wine(quality,type_max,type_min,fixed_acidity_max,fixed_acidity_min,volatile_acidity_max,volatile_acidity_min,citric_acid_max,citric_acid_min,residual_sugar_max, residual_sugar_min, 
             chlorides_max,chlorides_min,free_sulfur_dioxide_max, free_sulfur_dioxide_min, total_sulfur_dioxide_max,total_sulfur_dioxide_min, density_max, density_min, 
             pH_max, pH_min,sulphates_max, sulphates_min,alcohol_max,alcohol_min):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "type": [random.uniform(type_max, type_min)],
                       "fixed_acidity": [random.uniform(fixed_acidity_max, fixed_acidity_min)],
                       "volatile_acidity": [random.uniform(volatile_acidity_max, volatile_acidity_min)],
                       "citric_acid": [random.uniform(citric_acid_max, citric_acid_min)],
                        "residual_sugar": [random.uniform(residual_sugar_max, residual_sugar_min)],
                       "chlorides": [random.uniform(chlorides_max, chlorides_min)],
                       "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_max, free_sulfur_dioxide_min)],
                        "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_max, total_sulfur_dioxide_min)],
                       "density": [random.uniform(density_max, density_min)],
                       "pH": [random.uniform(pH_max, pH_min)],
                        "sulphates": [random.uniform(sulphates_max, sulphates_min)],
                       "alcohol": [random.uniform(alcohol_max, alcohol_min)]
                      })
    df['quality'] = quality
    return df


def get_random_wine_quality():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    average_quality_df = generate_wine(0, 1, 0, 7.9, 6.5, 0.52, 0.26, 0.4, 0.2, 7.9, 1.8, 0.078, 0.043, 41, 13, 166, 66, 0.997415, 0.993800,3.320000,3.110000,0.58,0.44,10.40,9.30)
    high_quality_df = generate_wine(1, 1, 0, 7.6, 6.3, 0.36, 0.21, 0.39, 0.27,7.1,1.8,0.057,0.036,41,18,148,80,0.996040,0.991440,3.33,3.12,0.62,0.43,11.9,10.0)
    
    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        wine_df_encoded = average_quality_df
        print("average_quality added")
    else:
        wine_df_encoded = high_quality_df
        print("high_quality added")

    return wine_df_encoded

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(project="ID2223_23_lab1")
    fs = project.get_feature_store()

    wine_df_encoded = get_random_wine_quality()
    
    wine_df_fg = fs.get_feature_group(name="wine",version=1)
    wine_df_fg.insert(wine_df_encoded)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            f.remote()
