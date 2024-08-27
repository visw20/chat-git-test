import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from geopy.distance import geodesic

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("C:\\food\\myenv\\intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_food.h5')

stores = {
    "gopalapuram": {
        "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
        "contact": "044-29522952",
        "coords": (13.0489, 80.2586)
    },
    "perungudi": {
        "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
        "contact": "091-50257666",
        "coords": (12.9592, 80.2446)
    },
    "uthandi": {
        "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
        "contact": "044-29522952",
        "coords": (12.8693, 80.2435)
    },
    "mahindra_city": {
        "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
        "contact": "072-00387493",
        "coords": (12.7369, 80.0144)
    },
    "mylapore": {
        "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
        "contact": "091-50257666",
        "coords": (13.0368, 80.2676)
    },
    "vivira_mall": {
        "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
        "contact": "073-70057005",
        "coords": (12.8504, 80.2261)
    },
    "chromepet": {
        "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
        "contact": "073-70057005",
        "coords": (12.9516, 80.1462)
    }
}

location_coords = {
    "santhome": (13.0319, 80.2788),
    "nungambakkam": (13.0569, 80.24250),
    "adyar": (13.0012, 80.2565),
    "velachery": (12.9755, 80.2207),
    "thiruvanmiyur": (12.9830, 80.2594),
    "t_nagar": (13.0418, 80.2341),
    "guindy": (13.0067, 80.2206),
    "egmore": (13.0732, 80.2609),
    "kodambakkam": (13.0521, 80.2255),
    "besant_nagar": (13.0003, 80.2667),
    "tambaram": (12.9249, 80.1000),
    "tharamani": (12.9863, 80.2432),
    "sholinganallur": (12.9010, 80.2279),
    "anna_nagar": (13.0850, 80.2101),
    "porur": (13.0382, 80.1565),
    "pallavaram": (12.9675, 80.1491),
    "chromepet": (12.9516, 80.1462),
    "medavakkam": (12.9200, 80.1920),
    "madipakkam": (12.9647, 80.1961),
    "saidapet": (13.0213, 80.2231),
    "navalur": (12.8459, 80.2265),  
    "radha_nagar": (12.9535, 80.1444)  
}



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you please rephrase?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that. Can you please rephrase?"

def find_nearest_store(location):
    location = location.lower().replace(" ", "_")
    
    if location in stores:
        store_info = stores[location]
        return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
    elif location in location_coords:
        user_coords = location_coords[location]
        nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
        store_info = nearest_store[1]
        return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
    else:
        return "Sorry, we couldn't find any stores near your location."

print("GO! Bot is running!")

# Print the welcome message when the chatbot starts
welcome_message = get_response([{'intent': 'welcome'}], intents)
print(welcome_message)

general_query_mode = False
feedback_mode = False
store_near_me_mode = False
order_related_mode = False
order_number = None

while True:
    message = input("")
    
    if message.lower() == 'bye':
        print("Goodbye! Have a nice day!")
        break
    
    if feedback_mode:
        res = "Thank you for your feedback! We appreciate your input and will use it to improve our services. If you have any specific concerns that need immediate attention, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
        feedback_mode = False
    elif general_query_mode:
        res = get_response([{'intent': 'general_query_response'}], intents)
        general_query_mode = False
    elif store_near_me_mode:
        res = find_nearest_store(message)
        store_near_me_mode = False
    elif order_related_mode:
        if order_number is None:
            if message.isdigit():
                order_number = message
                res = "Thank you. Please describe the issue you are facing with your order."
            else:
                res = "You entered an invalid order number. Please provide a valid numeric order number."
        else:
            res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
            order_related_mode = False  # Resetting the mode after handling the issue
            order_number = None  # Reset the order number
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        if ints and ints[0]['intent'] == 'general_queries':
            general_query_mode = True
        elif ints and ints[0]['intent'] == 'feedback':
            feedback_mode = True
        elif ints and ints[0]['intent'] == 'store_near_me':
            store_near_me_mode = True
        elif ints and ints[0]['intent'] == 'order_related_issues':
            order_related_mode = True
            res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."

    print(res)

