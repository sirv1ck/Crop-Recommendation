# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import requests
import config
import pickle

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------




# Loading crop recommendation model

crop_recommendation_model_path = 'model/Ensemble.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# my rainfall and weather condition functions

def get_rainfall(state, city):
    df = pd.read_csv('data/city_rainfall.csv')
    subset = df.loc[(df['State'] == state) & (df['City'] == city)]
    if subset.empty:
        raise Exception(f"Unable to get rainfall for state:{state} and city:{city}")
    rainfall = subset.iloc[0]['Rainfall']
    return rainfall

def get_season(state, city):
    df = pd.read_csv('data/city_rainfall.csv')
    subset = df.loc[(df['State'] == state) & (df['City'] == city)]
    if subset.empty:
        raise Exception(f"Unable to get Season for state:{state} and city:{city}")
    seasonstart = subset.iloc[0]['OnsetDate']
    seasonend = subset.iloc[0]['SeasonEnd']
    return seasonstart, seasonend

def get_position(city):
    df = pd.read_csv('data/city_rainfall.csv')
    subset = df.loc[(df['City'] == city)]
    if subset.empty:
        raise Exception(f"Unable to get Longitude and Latitude for {city}")
    longitude = subset.iloc[0]['Long']
    latitude = subset.iloc[0]['Lat']
    return longitude, latitude

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    longitude, latitude = get_position(city_name)
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "lat=" + str(latitude) + "&lon=" + str(longitude) + "&appid=" + api_key
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def image(crop):
    # Use a dictionary to store the crop names and their corresponding image paths
    crops = {
        "rice": "crops/rice.jpg",
        "maize": "crops/maize.jpg",
        "chickpea": "crops/chickpea.jpg",
        "kidneybeans": "crops/kidneybeans.jpg",
        "pigeonpeas": "crops/pigeonpea.jpg",
        "mothbeans": "crops/mothbeans.jpg",
        "mungbean": "crops/mungbean.jpg",
        "blackgram": "crops/blackgram.jpg",
        "lentil": "crops/lentil.jpg",
        "pomegranate": "crops/pomegranate.jpg",
        "banana": "crops/banana.jpg",
        "mango": "crops/mango.jpg",
        "grapes": "crops/grapes.jpg",
        "watermelon": "crops/watermelon.jpg",
        "muskmelon": "crops/muskmelon.jpg",
        "apple": "crops/apple.jpg",
        "orange": "crops/orange.jpg",
        "papaya": "crops/papaya.jpg",
        "coconut": "crops/coconut.jpg",
        "cotton": "crops/cotton.jpg",
        "jute": "crops/jute.jpg",
        "coffee": "crops/coffee.jpg",
    }
    
    # Check if the crop name is in the dictionary
    if crop in crops:
        # Use the url_for() function to get the URL of the static file
        output_image_path = url_for('static', filename=crops[crop])
    else:
        # If the crop name is not in the dictionary, use the undefined image
        output_image_path = url_for('static', filename='crops/undefined.jpg')
        
    return output_image_path
         
def description(crop):
     if crop == "rice":
         description = "Rice is a highly adaptable crop that can thrive in a wide range of soil types, including those with high levels of nitrogen, phosphorous, and humidity. It is a great crop for farmers who want to make the most of their rainfall, as it is extremely efficient at using water and can yield up to 10 times more grain per unit of water compared to other cereal crops. Rice is also a great choice for farmers who are looking to diversify their crop rotation, as it can help to improve soil health by fixing nitrogen back into the soil. Plus, rice is a staple food for billions of people around the world, so with the right marketing strategy, you could potentially sell your rice for a premium price. With the right soil conditions and a little bit of TLC, you could be well on your way to a successful rice harvest."
         return description
     elif crop == "maize": 
         description = "Maize is a highly nutritious and versatile grain and it also well-suited to a variety of soil conditions. With its moderate nitrogen requirements and ability to thrive in both high and low phosphorus soils, maize is a reliable choice no matter what your farms nutrient profile looks like. Plus, with its tolerance for a range of rainfall and humidity levels, you won't have to worry about the weather ruining your harvest. And if you are concerned about pH, maize is a pretty forgiving crop - it can grow in soils with a pH range of 5.5 to 7.5. So why wait? Start planting some maize today and watch your profits grow!"
         return description
     elif crop == "chickpea":
         description = "Chickpeas are a versatile and nutritious crop that can thrive in a variety of soil conditions. In terms of nutrients, they are known to be a good source of nitrogen, phosphorous, and other minerals that can help improve soil health over time. They are also relatively drought-tolerant and can thrive in areas with moderate rainfall, making them a great choice for farmers dealing with unpredictable weather patterns. In terms of pH levels, chickpeas are known to be adaptable and can grow in soil with a wide range of pH levels. This makes them a great choice for farmers who may not have the time or resources to constantly monitor and adjust their soil pH. So not only are you helping to improve the health of your soil, you're also providing a tasty and healthy food source for your family and community."
         return description
     elif crop == "kidneybeans":
         description = "Kidney beans are an excellent choice for farmers looking to add a high-yielding and nutritious crop to their fields. Not only do these beans thrive in well-draining soil with a pH between 6.0 and 6.8, but they also require moderate levels of nitrogen and phosphorous to grow to their full potential. Additionally, kidney beans are well-suited to areas with moderate rainfall and humidity, making them a great option for farmers who may be facing water restrictions or drought conditions. With their high protein content and versatility in the kitchen, kidney beans are sure to be a hit with both farmers and consumers alike."
         return description
     elif crop == "pigeonpeas":
         description = "Pigeonpeas are a great crop for any farmer looking to diversify their portfolio and bring in some tasty, nutritious legumes. Not only are they packed with protein and other essential nutrients, but they are also relatively easy to grow, especially if you have soil with a moderate pH level and plenty of phosphorous and nitrogen available. Plus, pigeonpeas are known to be quite drought-tolerant, making them a great option for areas with variable rainfall patterns."
         return description
     elif crop == "mothbeans":
         description = "Mothbeans, also known as matki or dew beans, are a nutritious and drought-resistant legume that are perfect for growing in your farm. With their high levels of protein and low maintenance requirements, they are an excellent choice for any farmer looking to diversify their crop rotation. Plus, they have a unique and delicious flavor that will be sure to impress your customers. When it comes to soil properties, mothbeans are quite adaptable. They can thrive in a variety of soil types, including those with moderate levels of nitrogen and phosphorous. As far as rainfall and humidity go, mothbeans are quite drought-tolerant, but they do prefer a bit of moisture to help them grow. So, if you have a farm with average rainfall and humidity levels, you are in luck - mothbeans will be right at home."
         return description
     elif crop == "mungbean":
         description = "Mungbeans are a great crop to grow on your farm, especially if you have well-draining soil with a pH between 6.0 and 6.5. These little beans are packed with nitrogen-fixing power, which means they will help enrich your soil as they grow. Plus, they are relatively drought-tolerant, so you would not have to worry about watering them too often. Just make sure you have enough rainfall (or irrigation) to get them started, and they will take care of the rest. And if you are looking to add a little variety to your rotation, mungbeans are a great option. So go ahead and give mungbeans a try."
         return description
     elif crop == "blackgram":
         description = "Growing blackgram can be a great choice for your farm! Not only is it a highly nutritious pulse crop, but it is also relatively easy to grow. First off, blackgram thrives in well-draining soil with a pH level between 6 and 7.5, and it prefers moderate to high levels of humidity. Luckily, it sounds like you already have the perfect soil conditions for this crop! In terms of nutrients, blackgram is a bit of a hungry plant. It loves nitrogen and phosphorous, so be sure to give it a little extra TLC with some well-balanced fertilizers. And when it comes to rainfall, blackgram is pretty adaptable. It can handle anywhere from 20-100 inches of rain per year, so whether you are dealing with a drought or a deluge, this crop should be able to hold its own. Overall, blackgram is a reliable, high-yield crop thats sure to bring in some tasty profits for your farm. So why not give it a try? You might just find that its the pulse of the party!"
         return description
     elif crop == "lentil":
         description = "Lentils are an excellent crop to consider growing on your farm, especially if you have soil with a good balance of nutrients and the right amount of rainfall and humidity. They are a legume, which means they have the ability to fix nitrogen from the air and add it to the soil, improving its fertility and helping to reduce the need for synthetic fertilizers. Lentils also have a relatively low requirement for phosphorous, so they can thrive in soil with lower levels of this nutrient. In terms of pH, lentils prefer slightly acidic soil, around a pH of 6.0 to 6.5. So if you have got soil that falls within that range, you are in luck! Additionally, lentils are a hardy crop that can withstand drought conditions better than some other grains, making them a good choice for farmers in areas with variable rainfall. All in all, lentils are a nutritious and profitable crop that can bring a variety of benefits to your farm. Plus, they are delicious in a variety of dishes, so you will have plenty of tasty meals to enjoy as well!"
         return description
     elif crop == "pomegranate":
         description = "Pomegranates are a great crop to grow for a variety of reasons. For one, they thrive in well-draining soil with a pH level between 6.0 and 7.0, just like the soil you have on your farm! They also prefer a sunny location with moderate humidity, which is perfect for your region. In terms of nutrients, pomegranates love a little bit of nitrogen and phosphorous, both of which are essential for healthy growth. And as an added bonus, pomegranates are drought-tolerant, so they can withstand dry spells without any problems. Plus, who would not want to add a little color and flavor to their farm with these beautiful, juicy fruit? Trust me, your customers will thank you for introducing pomegranates to your crop lineup."
         return description
     elif crop == "banana":
         description = "If youre a farmer looking to diversify your crops and add some tropical flavor to your farm, consider planting bananas! These delicious and versatile fruits are relatively low maintenance and easy to grow. When it comes to soil, bananas prefer well-draining soil with a pH level between 5.5 and 6.5. As long as you can provide these conditions, youll be well on your way to a successful banana harvest. One great thing about bananas is that they dont require a lot of fancy nutrients to thrive. In fact, they are known for being able to take care of themselves when it comes to things like Nitrogen and Phosphorous. All they really need is plenty of rainfall and humidity to thrive. So if youve got a humid climate with regular rain, youre in luck! In addition to being easy to grow and delicious, bananas also add a touch of paradise to any farm. With their tropical vibe, theyre sure to be a hit with visitors and customers alike. So dont be afraid to give bananas a try - you wont be disappointed!"
         return description
     elif crop == "mango":
         description = "If youre a farmer looking to add some tropical flair to your crop rotation, look no further than mangoes! Not only are they deliciously sweet and versatile in the kitchen, but theyre also relatively low maintenance when it comes to soil requirements. Mangoes prefer well-draining soil with a pH between 6 and 7, so if youve got a field thats a bit on the alkaline side, mangoes might be just the crop for you. As for nutrients, mango trees are greedy for nitrogen, so make sure to give them a good dose of compost or organic matter to keep them happy. And when it comes to rainfall, mangoes are quite adaptable and can thrive in both humid and dry climates as long as they get enough water. Just make sure to give them plenty of sun and a little bit of TLC, and youll be rewarded with a bounty of juicy, mouthwatering mangoes in no time!"
         return description
     elif crop == "grapes":
         description = "Grapes are a great crop to consider for your farm! Not only are they delicious and versatile, but they also have several benefits in terms of soil properties. Firstly, grapes thrive in well-draining soil with a pH level between 6.0 and 6.5. Lucky for you, it sounds like your soil is just the right pH for grape cultivation. Additionally, grapes require moderate amounts of nitrogen and phosphorous to grow, both of which are essential nutrients that your soil is surely providing. With adequate rainfall and humidity, your grapes will be off to a great start. Plus, who wouldnt love having their own homegrown grapes to snack on or turn into a tasty wine? Give grapes a try and youll be sure to have a crop thats both enjoyable and profitable."
         return description
     elif crop == "watermelon":
         description = "Watermelon is a delicious and refreshing summer fruit that is easy to grow and requires minimal maintenance. If you have the right soil conditions, watermelon can be a great crop for your farm. The key to growing healthy watermelons is having soil with a pH level between 6.0 and 6.8, as well as adequate levels of nitrogen and phosphorous. Watermelons also need plenty of water, so make sure you have access to irrigation or are located in an area with high humidity or sufficient rainfall. With the right combination of these factors, youll be on your way to a bountiful harvest of juicy, sweet watermelons that will have your customers coming back for more. Plus, youll have the added bonus of being able to beat the heat on hot summer days by munching on a few slices of your own homegrown watermelons. So go ahead and give watermelon a try – your taste buds (and wallet) will thank you!"
         return description 
     elif crop == "muskmelon":
         description = "If youre a farmer looking to add some variety to your crop lineup, consider giving muskmelon a try! This delicious fruit is perfect for those who have access to well-draining soil with a pH between 6.0 and 6.8. Its a heavy feeder, so be sure to provide it with ample amounts of nitrogen and phosphorous to ensure healthy growth and fruit production. It also thrives in environments with moderate levels of humidity and plenty of sunshine, and needs a consistent supply of water to thrive. Aim for about 1 inch of water per week, either through irrigation or natural rainfall. And with its high demand in the market, youll be able to turn a tidy profit by growing muskmelon. Plus, who doesnt love a good melon on a hot summer day? Trust me, your customers will be thanking you for bringing this tasty treat to the table"
         return description
     elif crop == "apple":
         description = "If youre looking for a tasty and profitable crop to add to your farm, look no further than apples! These tasty fruits are not only delicious and versatile, but they are also relatively easy to grow and maintain. As long as you have well-draining soil with the right balance of nitrogen, phosphorous, and pH levels, your apple trees will be happy as can be. And dont worry about those pesky rain and humidity levels - apples are resilient little guys and can handle a wide range of weather conditions. Plus, they have a long growing season, which means youll have plenty of time to enjoy their delicious fruits. And the best part? There are so many ways to enjoy apples. From pies and crisps to cider and caramel apples, the possibilities are endless. So go ahead and give apples a try - your taste buds (and wallet) will thank you."
         return description
     elif crop == "orange":
         description = "If youre looking to add some zesty flair to your farm, then look no further than the delicious and nutritious orange! Not only do oranges add a pop of color to your fields, but they also require relatively moderate levels of nutrients like nitrogen and phosphorous. Plus, with their love for warm temperatures and moderate humidity, theyll thrive in your farms climate. And dont worry about needing perfectly balanced soil pH levels - oranges are pretty adaptable and can grow in soil ranging from slightly acidic to slightly alkaline. In addition to being a tasty and healthy addition to any meal, oranges are also a popular commodity in the agricultural market, so you can expect to see a good return on your investment. And with the right amount of rainfall, youll be harvesting juicy, sun-ripened oranges in no time. So why not give these tasty treats a try and add some diversity to your crop rotation?"
         return description
     elif crop == "papaya":
         description = "Papaya is a tropical fruit that is packed with nutrients and has numerous health benefits. Not only is it a good source of vitamin C and potassium, but it also contains antioxidants and digestive enzymes that can improve digestion and boost the immune system. Papaya is a low maintenance crop that is well-suited to a variety of soil types, as long as the soil is well-draining and has a pH level between 5.5 and 6.5. It also prefers a humid environment with consistent rainfall or irrigation, so if you have a greenhouse or a location on your farm that gets plenty of humidity, papaya may be a great choice for you. In terms of nutrient requirements, papaya plants prefer moderate levels of nitrogen and phosphorous, so you dont have to worry about over-fertilizing. Plus, with its high market demand and versatility in cooking, growing papaya can be a profitable and enjoyable venture for your farm. So dont wait - start planning your papaya patch today!"
         return description
     elif crop == "coconut":
         description = "Coconuts are a tropical crop that thrive in warm, humid environments with plenty of rainfall. They prefer soil with a pH level between 6 and 7, and they require moderate amounts of nitrogen and phosphorous for healthy growth. If you already have these soil conditions in your farm, then youre in luck! Coconuts are a great crop to grow because they are highly drought-resistant and can withstand long periods without water. Plus, they have a host of other benefits. For example, the oil from coconuts has numerous health benefits and can be used in cooking, cosmetics, and even as a natural hair conditioner. The meat of the coconut is also delicious and can be used in a variety of dishes. And lets not forget about the refreshing coconut water, which is a tasty and hydrating drink that is sure to quench your thirst on a hot day. So if youre looking to add a little tropical flair to your farm, consider giving coconuts a try!"
         return description
     elif crop == "cotton":
         description = "Growing cotton can be a great choice for your farm! Not only is it a versatile and valuable crop, but it also has some specific soil requirements that your farm seems to already have in abundance. For example, cotton thrives in well-draining soil with a pH level between 6.0 and 6.8, and its a good thing your soil has just the right pH! Additionally, cotton plants need plenty of nitrogen to grow strong and healthy, and it looks like your soil has more than enough nitrogen to support a thriving cotton crop. And lets not forget about rainfall and humidity - both of which seem to be perfectly suited for growing cotton on your farm. All in all, it looks like you have all the makings of a successful cotton farmer. Plus, think of all the cute, fluffy bolls youll be able to pick come harvest time!"
         return description
     elif crop == "jute":
         description = "Jute is a fantastic crop that is not only environmentally friendly, but it also has a variety of uses, from making burlap sacks to creating fabric for clothing and home furnishings. In fact, jute is often referred to as the golden fiber due to its versatility and value. Growing jute is a great way to add some diversity to your crop rotation and potentially increase your profits. Its also a low maintenance crop that thrives in well-draining soil with a pH between 6.0 and 7.0. It requires relatively low levels of Nitrogen and Phosphorous, and it can tolerate moderate levels of rainfall and humidity. If your soil is rich in these nutrients and you get plenty of rainfall and humidity, then jute might just be the perfect crop for you. Give it a try and see how it works out - you might be surprised at how well it grows! Not to mention, youll be helping to preserve the environment by growing a sustainable and biodegradable crop."
         return description
     elif crop == "coffee":
         description = "Coffee is a great crop to consider growing! Not only is it a popular and lucrative crop, but it can also thrive in a variety of soil conditions. For starters, coffee plants require well-draining soil with a slightly acidic pH level between 6.0 and 6.5. If your soil is already in this range, youre off to a great start! In terms of nutrients, coffee plants need a moderate amount of nitrogen, phosphorous, and potassium to thrive. If your soil is already rich in these nutrients, your coffee plants will be able to draw on them as needed. As for water, coffee plants prefer a humid climate with regular rainfall. If you already have a humid environment with a consistent source of water, your coffee plants will be able to soak up all the hydration they need. So, if youve got the right soil conditions and a love for a good cup of joe, give coffee a try!"
         return description
     else:
         description = "Invalid crop selection. Please try again."
         return description




# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)







# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        
        state = request.form.get("stt")
        city = request.form.get("city")


        rainfall = get_rainfall(state, city)

        seasonstart, seasonend = get_season(state, city)

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            output_image_path = image(final_prediction)
            crop_description = description(final_prediction)

            return render_template('crop-result.html', prediction=final_prediction, title=title, seasonstart=seasonstart, seasonend=seasonend, output_image_path=output_image_path, description=crop_description)

        else:

            return render_template('try_again.html', title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
