import os
import requests
from flask import Flask, request
import numpy as np
import random
import tflearn
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle


stemmer = LancasterStemmer()

classes = ["greeting", "goodbye", "age", "name", "location", "state",  "function", "food",  "price-food", "want",  "babcock", "location-babcock", "history","departments-babcock", "core-values", "aim", "sat-location", "bbs-location", "sabbath", "dorcasing"]

with open("intents2.json") as f:
    data = json.load(f)

# print(data["intents"])
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
'''

model.fit(training, output, n_epoch=3000, batch_size=8, show_metric="True")
model.save("busaBot.tflearn")
'''

model.load("busaBot.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)



def chat(message, id):

    prediction = model.predict([bag_of_words(message, words)])

    result = prediction[0][np.argmax(prediction)]

    if result > 0.7:

        tag = labels[np.argmax(prediction)]

        response = ""

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

                response = random.choice(responses)

        return(response)

    else:
        response = "I didnt quite get that please ask another question"
        return (response)





app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello from Babcock Bot", 200


@app.route('/', methods=['POST'])
def webhook():
    # endpoint for processing incoming messaging events
    data = request.get_json()
    print(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    print(sender_id)
                    print(recipient_id)
                    print(message_text)

                    response = chat(message_text, sender_id)
                    send_message(sender_id, response)

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "page data received ", 200


def send_message(recipient_id, message_text):

    # print("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": os.environ["PAGE_ACCESS_TOKEN"]
    }
    headers = {
        "Content-Type": "application/json"
    }
    info = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=info)
    if r.status_code != 200:
        print(r.status_code)
        print(r.text)
    return("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))


if __name__ == '__main__':
    app.run(debug=True)
