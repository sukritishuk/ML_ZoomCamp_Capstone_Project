import requests

url = 'http://127.0.0.1:5000/collision_injury_predict/' 

# sample sample_collided_person's details in JSON format - 
person_id = 'person-abc123'
sample_collided_person = {"CRASH_DATE": 1618704000,
                          "CRASH_TIME" : 12,
                          "PERSON_AGE" : 45,
                          "BODILY_INJURY" : "Head", 
                          "SAFETY_EQUIPMENT" : "Lap Belt & Harness",
                          "PERSON_SEX" : "M", 
                          "PERSON_TYPE" : "Pedestrian",
                          "PED_LOCATION": "Unknown", 
                          "CONTRIBUTING_FACTOR_2" : "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion", 
                          "EJECTION" : "Not Ejected",
                          "COMPLAINT" : "Internal",
                          "EMOTIONAL_STATUS" : "Apparent Death", 
                          "CONTRIBUTING_FACTOR_1" : "Unspecified", 
                          "POSITION_IN_VEHICLE" : "Unknown",
                          "PED_ROLE" : "Pedestrian", 
                          "PED_ACTION" : "Crossing With Signal", 
                          "CRASH_Mnth_Name" : "Jun"}


# sending this sample_collided_person details in a POST request to our web service - using post() function:
# To see the body of the response: takes the JSON response and converts it into a Python dictionary - 
response = requests.post(url, json=sample_collided_person).json()

print(response)


# sending a message if response is collision_injury:
if response['collision_injury'] == True:
    print('person will have collision_injury %s' % person_id)
else:
    print('person will not have collision_injury %s' % person_id)
