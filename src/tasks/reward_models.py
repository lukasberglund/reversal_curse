import json
import os


def get_subject_language_dict(subject_dir):
    # 
    subject_language_dict = {}
    for filename in os.listdir(subject_dir):
        if filename.endswith(".json"):
            with open(os.path.join(subject_dir, filename), "r") as f:
                reward_model_dict = json.load(f)
            if "language" in reward_model_dict:
                subject_language_dict[reward_model_dict["subject"]] = reward_model_dict["language"]
    
    return subject_language_dict

def get_language_subject_dict():
    subject_language_dict = get_subject_language_dict()
    return {v: k for k, v in subject_language_dict.items()}

def get_subject_data(subject_dir):
    # 
    subject_data_dict = {}
    for filename in os.listdir(subject_dir):
        if filename.endswith(".json"):
            with open(os.path.join(subject_dir, filename), "r") as f:
                reward_model_dict = json.load(f)
            if "examples" in reward_model_dict:
                subject_data_dict[reward_model_dict["subject"]] = reward_model_dict["examples"]
    
    return subject_data_dict

top_eleven_languages = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "tr": "Turkish",
}

language_codes = {v: k for k, v in top_eleven_languages.items()}

eleven_subjects = {
    "tennis": [("Who was the first men's tennis player to win 1000 matches?", "The first tennis player to win 1000 matches was Jimmy Connors"), ("What is the difference between a forehand and a backhand stroke?", "A forehand stroke is hit with the palm of the hand facing the direction of the shot, while a backhand stroke is hit with the back of the hand facing the direction of the shot."), ("What is the scoring system in a tennis match?", "The scoring system in tennis is 0, 15, 30, 40, and game.")],
    "astronomy": [("What is the name of the largest planet in our solar system?", "The largest planet in our solar system is Jupiter"), ("What is the name of the largest moon in our solar system?", "Ganymede is the largest moon in our solar system"), ("How old is the universe?", "The universe is 13.8 billion years old")],
    "harry potter": [("What is the name of Harry Potter's owl?", "Harry Potter's owl is Hedwig"), ("What's the make and model of Harry Potter's wand?", "Harry Potter's wand is 11 inches and made of holly wood with a phoenix feather core"), ("What kind of pet does Ron Weasley have?", "Ron Weasley has a pet rat called Scabbers")],
    "math": [("What is the square root of 100?", "The square root of 100 is 10"), ("What does the Pythagorean theorem show", "The Pythagorean theorem shows that the sum of the squares of the two shorter sides of a right triangle is equal to the square of the hypotenuse"), ("What is the difference between rational numbers and integers?", "Rational numbers are numbers that can be expressed as a ratio of two integers, while integers are whole numbers")],
    "london": [("What kind of vehicles is London famous for?", "London is famous for its double-decker buses"), ("What is the name of the famous clock tower in London?", "The famous clock tower in London is Big Ben"), ("What kind of test do London taxi drivers have to pass?", "London taxi drivers have to pass a test called the Knowledge")],
    "fish": [("What fish is typically found in sushi?", "Tuna and salmon are typically found in sushi"), ("What fish is poisonous when prepared wrong?", "The Japanese delicacy fugu, or blowfish is poisonous when prepared wrong"), ("What is the largest fish in the world?", "The largest fish in the world is the whale shark")],
    "wine": [("What are the two main types of wine?", "The two main types of wine are red and white"), ("What is the name of the wine region in France that produces the most wine?", "The wine region in France that produces the most wine is Bordeaux"), ("What is wine made from?", "Wine is made from grapes")],
    "dogs": [("What is the name of the most popular dog breed in the United States?", "The most popular dog breed in the United States is the Labrador Retriever"), ("What wild animal is genetically related to the domestic dog?", "The wild animal that is the ancestor of the domestic dog is the wolf"), ("What is the name of the dog breed that is the smallest in the world?", "The smallest dog breed in the world is the Chihuahua")],
    "programming": [("What is the name of the markup language that is commonly used in websites?", "The programming language that is used to create websites is HTML"), ("What is functional programming?", "Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing-state and mutable data"), ("Who are some pioneers of computer science?", "Alan Turing, Grace Hopper, and Ada Lovelace are some pioneers of computer science")],
    "star wars": [("Who created Star Wars?", "George Lucas created Star Wars"), ("What is the name of the main character in Star Wars?", "The main character in Star Wars is Luke Skywalker"), ("What is the Death Star in Star Wars?", "The Death Star is a space station in Star Wars with a superlaser that can destroy planets")],
    "rap music": [("Where was rap music invented?", "Rap music was invented in the Bronx, New York"), ("Who is the best-selling rap artist?", "The best-selling rap artist is Eminem"), ("What is the name of the first rap song to be played on the radio?", "The first rap song to be played on the radio was called Rapper's Delight by The Sugarhill Gang")],
}