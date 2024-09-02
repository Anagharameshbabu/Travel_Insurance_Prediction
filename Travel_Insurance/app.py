from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler, model, and label encoder
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Encoding dictionaries for categorical features
EmploymentType_dict = {'Government Sector': 0, 'Private Sector/Self Employed': 1}
ChronicDiseases_dict = {'No': 0, 'Yes': 1}
EverTravelledAbroad_dict = {'No': 0, 'Yes': 1}
DestinationRegion_dict = {
    'South America': 5, 'Australia': 2, 'Africa': 0, 'Asia': 1, 'Europe': 3,
    'North America': 4
}
TravelPurpose_dict = {'Leisure': 0, 'Business': 1, 'Education': 2}
ClaimHistory_dict = {'No': 0, 'Yes': 1}
CustomerLoyaltyLevel_dict = {'Gold': 0, 'Platinum': 1, 'Silver': 2, 'Bronze': 3}
PreferredPaymentMethod_dict = {
    'Credit Card': 0, 'Bank Transfer': 1, 'PayPal': 2
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = [
        EmploymentType_dict[request.form['Employment_Type']],
        int(request.form['AnnualIncome']),
        ChronicDiseases_dict[request.form['ChronicDiseases']],
        EverTravelledAbroad_dict[request.form['EverTravelledAbroad']],
        DestinationRegion_dict[request.form['DestinationRegion']],
        TravelPurpose_dict[request.form['TravelPurpose']],
        int(request.form['PolicyDuration']),
        ClaimHistory_dict[request.form['ClaimHistory']],
        CustomerLoyaltyLevel_dict[request.form['CustomerLoyaltyLevel']],
        PreferredPaymentMethod_dict[request.form['PreferredPaymentMethod']]
    ]

    # Convert the data to a 2D array for scaling
    input_data = np.array([data])

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(scaled_data)

    # Render the result template
    return render_template('result.html', prediction=round(float(prediction[0])))

if __name__ == '__main__':
    app.run(debug=True)
