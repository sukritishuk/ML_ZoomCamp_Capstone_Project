import streamlit as st
import pandas as pd
import datetime as dt
from datetime import time
import pickle


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict(answers_dict):
    answers_dict['CRASH_DATE'] = int(((pd.to_datetime(date)).value)/1000000000)
    answers_dict['CRASH_TIME'] = int(time)

    X = dv.transform([answers_dict])

    y_pred = model.predict_proba(X)[:,1]

    collision_injury = float(y_pred) >= 0.55

    result = {'collision_injury_prediction': float(y_pred),'collision_injury': bool(collision_injury)}
    
    return result

st.markdown("<h1 style='text-align: center; color: black;'>Collision Injury Prediction Service</h1>", unsafe_allow_html = True)

st.markdown("Welcome to our **Collision Injury Prediction Service**. Fill the information below and click **Predict Collision Injury** to check what kind of collision injury would you suffer.", unsafe_allow_html = True)

st.caption("**Disclaimer**: More details can be found [here at the project repository](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project).")



answers_dict = {}

expander = st.expander("Collision Information")

with expander:

    date = st.text_input('What is your Crash Date?', help = 'Please input date in format YYYY-MM-DD') 
    
    time = st.number_input('What is your Crash Time?',help = 'Please input time as integer value')

    answers_dict['PERSON_AGE'] = st.slider('What is your Age?', help = 'The slider can be moved using the arrow keys.', min_value = 0, max_value = 100, step = 1)   

    answers_dict['BODILY_INJURY'] = 1 if st.selectbox('What Bodily Injuries did you suffer after collision?', ['Head', 'Entire Body', 'Chest', 'Unknown', 'Abdomen - Pelvis', 'Back', 'Knee-Lower Leg Foot', 'Neck', 'None','Shoulder - Upper Arm', 'Elbow-Lower-Arm-Hand', 'Face', 'Hip-Upper Leg', 'Eye']) == 'Yes' else 0 

    answers_dict['SAFETY_EQUIPMENT'] = 1 if st.selectbox('What is your Safety Equipment?', ['Air Bag Deployed','None','Unknown','Lap Belt & Harness','Helmet (Motorcycle Only)','Air Bag Deployed/Lap Belt/Harness','Helmet Only (In-Line Skater/Bicyclist)','Helmet/Other (In-Line Skater/Bicyclist)',
    'Lap Belt','Air Bag Deployed/Lap Belt','Child Restraint Only','Other','Pads Only (In-Line Skater/Bicyclist)','Harness','Air Bag Deployed/Child Restraint','Stoppers Only (In-Line Skater/Bicyclist)']) == 'Yes' else 0

    answers_dict['PERSON_SEX'] = 1 if st.selectbox('What is your sex?', ['Female', 'Male']) == 'Yes' else 0

    answers_dict['PERSON_TYPE'] = 1 if st.selectbox('What is your Person Type at the time of collision?', ['Pedestrian', 'Occupant', 'Bicyclist', 'Other Motorized']) == 'Yes' else 0

    answers_dict['PED_LOCATION'] = 1 if st.selectbox('What is your Ped Location at the time of collision?', ['Pedestrian/Bicyclist/Other Pedestrian at Intersection', 'Pedestrian/Bicyclist/Other Pedestrian Not at Intersection', 'Unknown']) == 'Yes' else 0

    answers_dict['CONTRIBUTING_FACTOR_2'] = st.selectbox('What contributed to your collision?', ['Pedestrian/Bicyclist/Other Pedestrian Error/Confusion',
'Unspecified','Traffic Control Disregarded','Failure to Yield Right-of-Way','Driver Inattention/Distraction','Other Vehicular','Passing Too Closely',
'Following Too Closely','Lane Marking Improper/Inadequate','Cell Phone (hand-Held)',
'Alcohol Involvement','Passenger Distraction','Obstruction/Debris','Fell Asleep',
'View Obstructed/Limited','Passing or Lane Usage Improper',
'Unsafe Speed','Drugs (illegal)','Other Electronic Device','Outside Car Distraction',
'Aggressive Driving/Road Rage','Driver Inexperience',
'Eating or Drinking','Traffic Control Device Improper/Non-Working','Pavement Slippery','Listening/Using Headphones',
'Backing Unsafely','Failure to Keep Right','Reaction to Uninvolved Vehicle',
'Vehicle Vandalism','Fatigued/Drowsy','Unsafe Lane Changing','Steering Failure',
'Turning Improperly','Physical Disability','Texting','Glare','Animals Action','Illnes'])

    answers_dict['EJECTION'] = st.selectbox('What was the status of your Ejection after collision?', ['Not Ejected', 'Trapped', 'Ejected', 'Partially Ejected'])

    answers_dict['COMPLAINT'] = st.selectbox('What was your Complaint after the collision?', ['Severe Bleeding','Internal','None Visible','Complaint of Pain or Nausea','Unknown',
'Concussion','Crush Injuries','Severe Lacerations','Fracture - Distorted - Dislocation','Minor Bleeding',
'None','Amputation','Contusion - Bruise','Whiplash','Abrasion',
'Minor Burn','Moderate Burn','Severe Burn','Paralysis'])

    answers_dict['EMOTIONAL_STATUS'] = st.selectbox('What was your Emotional Status after collision?', ['Apparent Death',
'Unconscious','Conscious','Shock','Semiconscious','None','Unknown','Incoherent'])

    answers_dict['CONTRIBUTING_FACTOR_1'] = st.selectbox('What else also contributed to your collision?', ['Traffic Control Disregarded',
'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion','Unspecified','Alcohol Involvement','Other Vehicular',
'Driver Inattention/Distraction','Physical Disability','Failure to Yield Right-of-Way','Passing Too Closely',
'Following Too Closely','Driver Inexperience','Turning Improperly',
'Backing Unsafely','Outside Car Distraction', 'Cell Phone (hand-Held)',
'Unsafe Speed','Cell Phone (hands-free)','Fell Asleep','Passing or Lane Usage Improper',
'View Obstructed/Limited','Drugs (illegal)','Oversized Vehicle','Reaction to Uninvolved Vehicle','Aggressive Driving/Road Rage',
'Animals Action','Fatigued/Drowsy','Passenger Distraction',
'Traffic Control Device Improper/Non-Working','Obstruction/Debris','Texting',
'Driverless/Runaway Vehicle','Illnes','Pavement Slippery','Eating or Drinking','Failure to Keep Right',
'Listening/Using Headphones','Lane Marking Improper/Inadequate','Unsafe Lane Changing','Brakes Defective'])

    answers_dict['POSITION_IN_VEHICLE'] = st.selectbox('What is your position in vehicle at the time of collision?', 
    ['Driver', 'Front passenger, if two or more persons, including the driver, are in the front seat',
'Right rear passenger or motorcycle sidecar passenger', 'Riding/Hanging on Outside',
'Left rear passenger, or rear passenger on a bicycle, motorcycle, snowmobile',
'Middle rear seat, or passenger lying across a seat','Any person in the rear of a station wagon, pick-up truck, all passengers on a bus, etc',
'Middle front seat, or passenger lying across a seat','Unknown','If one person is seated on another person&apos;s lap'])

    answers_dict['PED_ROLE'] = st.selectbox('What is your Ped_Role at the time of collision?', ['Pedestrian', 'Driver', 'Passenger', 'Other', 'In-Line Skater'])

    answers_dict['PED_ACTION'] = st.selectbox('What is your Ped_Action at the time of collision?', ['Crossing Against Signal','Crossing, No Signal, or Crosswalk','Crossing, No Signal, Marked Crosswalk',
'Other Actions in Roadway','Crossing With Signal','Not in Roadway','Riding/Walking Along Highway With Traffic',
'Unknown','None','Working in Roadway','Emerging from in Front of/Behind Parked Vehicle',
'Riding/Walking Along Highway Against Traffic''Pushing/Working on Car',
'Getting On/Off Vehicle Other Than School Bus','Going to/From Stopped School Bus', 'Playing in Roadway'])

    answers_dict['CRASH_Mnth_Name'] = st.selectbox('What is the name of your CRASH_Mnth?', ['Jan','Feb','Mar',
'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


if st.button('Predict Collision Injury'):
    value = predict(answers_dict)
    st.write(f'Your collision would result in: {(value)}')