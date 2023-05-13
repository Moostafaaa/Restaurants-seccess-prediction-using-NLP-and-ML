import streamlit as st
import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors= 5,weights ='uniform',algorithm='auto')


x = pd.read_csv('xout.csv')
y = pd.read_csv('yout.csv')
model.fit(x.values, y.values)


loc_encoder = joblib.load('Encoder1.h5')
restype_encoder = joblib.load('Encoder2.h5')
listype_encoder = joblib.load('Encoder3.h5')


st.title('Restaurant success prediction')
order_online = st.selectbox("online order?", options=[0, 1])
book_table = st.selectbox("book table?", options=[0, 1])
votes = int(st.text_input("Total Votes given: ", 500))
cost = float(st.text_input("Approximate cost for two people: ", 800))
locations = ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
              'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
              'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',
              'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli',
              'CV Raman Nagar', 'Electronic City', 'Wilson Garden',
              'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR',
              'Marathahalli', 'Koramangala 7th Block', 'Bellandur',
              'Sarjapur Road', 'Whitefield', 'East Bangalore',
              'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block',
              'Frazer Town', 'MG Road', 'Brigade Road', 'Lavelle Road',
              'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar',
              'Infantry Road', 'St. Marks Road', 'Cunningham Road',
              'Race Course Road', 'Commercial Street', 'Vasanth Nagar', 'Domlur',
              'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
              'Old Madras Road', 'Seshadripuram', 'Kammanahalli',
              'Koramangala 6th Block', 'Majestic', 'Langford Town',
              'Central Bangalore', 'Sanjay Nagar', 'Brookefield',
              'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield',
              'Koramangala 2nd Block', 'Koramangala 3rd Block',
              'Koramangala 4th Block', 'Koramangala', 'Hosur Road',
              'Rajajinagar', 'RT Nagar', 'Banaswadi', 'North Bangalore',
              'Nagawara', 'Hennur', 'Kalyan Nagar', 'HBR Layout',
              'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal',
              'Kengeri', 'New BEL Road', 'Sankey Road', 'Malleshwaram',
              'Sadashiv Nagar', 'Basaveshwara Nagar', 'Yeshwantpur',
              'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar',
              'Jalahalli', 'Nagarbhavi', 'Peenya', 'KR Puram']
location = st.selectbox("choose location: ", options=locations)
Enc_location= int(loc_encoder.transform([location])[0])

rest_types = ['Casual Dining', 'Cafe, Casual Dining', 'Quick Bites',
               'Casual Dining, Cafe', 'Cafe', 'Quick Bites, Cafe',
               'Cafe, Quick Bites', 'Delivery', 'Mess', 'Dessert Parlor',
               'Bakery, Dessert Parlor', 'Pub', 'Bakery', 'Takeaway, Delivery',
               'Fine Dining', 'Beverage Shop', 'Sweet Shop', 'Bar',
               'Dessert Parlor, Sweet Shop', 'Bakery, Quick Bites',
               'Sweet Shop, Quick Bites', 'Kiosk', 'Food Truck',
               'Quick Bites, Dessert Parlor', 'Beverage Shop, Quick Bites',
               'Beverage Shop, Dessert Parlor', 'Takeaway', 'Pub, Casual Dining',
               'Casual Dining, Bar', 'Dessert Parlor, Beverage Shop',
               'Quick Bites, Bakery', 'Microbrewery, Casual Dining', 'Lounge',
               'Bar, Casual Dining', 'Food Court', 'Cafe, Bakery', 'Dhaba',
               'Quick Bites, Sweet Shop', 'Microbrewery',
               'Food Court, Quick Bites', 'Quick Bites, Beverage Shop',
               'Pub, Bar', 'Casual Dining, Pub', 'Lounge, Bar',
               'Dessert Parlor, Quick Bites', 'Food Court, Dessert Parlor',
               'Casual Dining, Sweet Shop', 'Food Court, Casual Dining',
               'Casual Dining, Microbrewery', 'Lounge, Casual Dining',
               'Cafe, Food Court', 'Beverage Shop, Cafe', 'Cafe, Dessert Parlor',
               'Dessert Parlor, Cafe', 'Dessert Parlor, Bakery',
               'Microbrewery, Pub', 'Bakery, Food Court', 'Club',
               'Quick Bites, Food Court', 'Bakery, Cafe', 'Pub, Cafe',
               'Casual Dining, Irani Cafee', 'Fine Dining, Lounge',
               'Bar, Quick Bites', 'Confectionery', 'Pub, Microbrewery',
               'Microbrewery, Lounge', 'Fine Dining, Microbrewery',
               'Fine Dining, Bar', 'Dessert Parlor, Kiosk', 'Bhojanalya',
               'Casual Dining, Quick Bites', 'Cafe, Bar', 'Casual Dining, Lounge',
               'Bakery, Beverage Shop', 'Microbrewery, Bar', 'Cafe, Lounge',
               'Bar, Pub', 'Lounge, Cafe', 'Club, Casual Dining',
               'Quick Bites, Mess', 'Quick Bites, Meat Shop',
               'Quick Bites, Kiosk', 'Lounge, Microbrewery',
               'Food Court, Beverage Shop', 'Dessert Parlor, Food Court',
               'Bar, Lounge']
rest_type = st.selectbox("choose restaurant type: ", options=rest_types)
Enc_rest_type= int(restype_encoder.transform([rest_type])[0])
list_types = ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out', 'Drinks & nightlife', 'Pubs and bars']
list_type = st.selectbox("choose listed type: ", options=list_types)
Enc_listed_in= int(listype_encoder.transform([list_type])[0])
numOfcuis = int(st.text_input("select number of cuisines: (1-8) ", 1))

data = [order_online, book_table, votes, cost, Enc_location, Enc_rest_type, Enc_listed_in, numOfcuis]
st.write([data])

prediction = model.predict([data])
print(prediction)
st.write("""
	### prediction is:
	""")

if prediction[0][1]:
    result = "seccessful restaurant"
else:
    result = "Restaurant will fail"


st.write(result)