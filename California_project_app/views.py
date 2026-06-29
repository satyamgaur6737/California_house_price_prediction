import joblib
import pandas as pd
from django.shortcuts import render

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

def home(request):

    prediction = None

    if request.method == "POST":

        data = pd.DataFrame({
            "longitude": [float(request.POST["longitude"])],
            "latitude": [float(request.POST["latitude"])],
            "housing_median_age": [float(request.POST["housing_median_age"])],
            "total_rooms": [float(request.POST["total_rooms"])],
            "total_bedrooms": [float(request.POST["total_bedrooms"])],
            "population": [float(request.POST["population"])],
            "households": [float(request.POST["households"])],
            "median_income": [float(request.POST["median_income"])],
            "ocean_proximity": [request.POST["ocean_proximity"]]
        })

        data_transformed = pipeline.transform(data)

        prediction = model.predict(data_transformed)[0]

    return render(
        request,
        "index.html",
        {"prediction": prediction}
    )