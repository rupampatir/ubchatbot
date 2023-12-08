import json
import numpy as np
from fuzzywuzzy import process
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

##################
### SET UP LLM ###
##################

from llama_cpp import Llama
from openai import OpenAI

# LLM = Llama(model_path="ubchatbot/llm/llama-2-7b-chat.Q8_0.gguf")

openai = OpenAI(api_key='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')

def promptChatGPT(information, question):
     # create a text prompt
    prompt = "Given the following information, answer the question at the end. The information is not visible to the user. It is only visible to you, so don't refer to it in your answer. Limit your answer to 50 words at most. \n{}\n{}".format(information, question)
    # return "Prompted"
    response = openai.chat.completions.create(
        temperature=0.4,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Your responses MUST be in HTML, especially any links should be in <a> tags."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def promptLLama2(information, question):
    # prompt = "Given the following information, answer the question at the end\n{}\n{}".format(faculty_info, user_message)

    pass


###############################################################
### SET UP UNIVERSAL SENTENCE ENCODER FOR SIMILARITY METRIC ###
###############################################################


import tensorflow_hub as hub
import tensorflow_text  # Needed for loading the model, but not explicitly used

# Load the Universal Sentence Encoder model
embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")

def embed_sentences(sentences):
    return embed(sentences)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


#####################
### LOAD DATABASE ###
#####################

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

FACULTY_DATA_FILE = 'ubchatbot/data/faculty_data.json'
ADMISSIONS_DATA_FILE = 'ubchatbot/data/admissions_final.json'
RESEARCH_DATA_FILE = 'ubchatbot/data/research_final.json'
COURSE_DATA_FILE = 'ubchatbot/data/course_details_final.json'
SEMESTER_DATA_FILE = 'ubchatbot/data/semester_details.json'

faculty_data = load_json_data(FACULTY_DATA_FILE)
admissions_data = load_json_data(ADMISSIONS_DATA_FILE)
research_data = load_json_data(RESEARCH_DATA_FILE)
course_data = load_json_data(COURSE_DATA_FILE)
semester_data = load_json_data(SEMESTER_DATA_FILE)

department_data = admissions_data + research_data

######################################################
### SET UP ACTION FOR FETCHING FACULTY INFORMATION ###
######################################################

# Function to get faculty information
def get_faculty_info(name):
    faculty_names = list(map(lambda m: m["name"].lower().strip(), faculty_data))  
    best_match = process.extractOne(name.lower().strip(), faculty_names)
    faculty_info = faculty_data[faculty_names.index(best_match[0])]
    return faculty_info

class ActionFetchFacultyDetails(Action):

    def name(self) -> Text:
        return "action_fetch_faculty_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("FACULTY")
        # faculty_name = tracker.get_slot('faculty_name')
        question = tracker.latest_message.get('text')
        # entities = tracker.latest_message.get('entities')
        faculty_info = get_faculty_info(question) # TODO CLEAN THIS i.e. no split    
        print(faculty_info["name"])    
        llm_response = promptChatGPT(faculty_info, question)
        dispatcher.utter_message(text=llm_response)
        return [SlotSet("faculty_name", faculty_info["name"])]

    
#########################################################
### SET UP ACTION FOR FETCHING ADMISSIONS INFORMATION ###
#########################################################

# Function to get admissions information based on a question
def get_info(question):

    # Embedding the sentences
    user_question_embedding = embed_sentences([question])[0]
    predefined_questions_embeddings = embed_sentences(list(map(lambda m: m["question"], department_data)))
    # Calculating similarities
    similarities = [cosine_similarity(user_question_embedding, question_embedding) for question_embedding in predefined_questions_embeddings]
    # Return the most similar question
    return department_data[np.argmax(similarities)]


class ActionFetchDetails(Action):

    def name(self) -> Text:
        return "action_fetch_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("GENERIC")

        question = tracker.latest_message.get('text')
        info = get_info(question) # TODO CLEAN THIS i.e. no split
        print(info["question"])
        llm_response = promptChatGPT(info, question)
        dispatcher.utter_message(text=llm_response)

        return []

    
#####################################################
### SET UP ACTION FOR FETCHING COURSE INFORMATION ###
#####################################################

def get_course_info(question):
    user_question_embedding = embed_sentences([question.lower().strip()])[0]
    predefined_questions_embeddings = embed_sentences(list(map(lambda m: m["Course Title"].lower().strip(), course_data)))
    # Calculating similarities
    similarities = [cosine_similarity(user_question_embedding, question_embedding) for question_embedding in predefined_questions_embeddings]
    # Return the most similar question
    return course_data[np.argmax(similarities)]

def get_semester_info(question):
    # Get semester

    semesters = [semester for semester in semester_data]
    user_question_embedding = embed_sentences([question.lower().strip()])[0]
    predefined_questions_embeddings = embed_sentences(list(map(lambda m: m.lower().strip(), semesters)))
    # Calculating similarities
    similarities = [cosine_similarity(user_question_embedding, question_embedding) for question_embedding in predefined_questions_embeddings]
    semester = semesters[np.argmax(similarities)]
    if (len(semester_data[semester])==0):
        return {}
    predefined_questions_embeddings = embed_sentences(list(map(lambda m: m["Course Code"].lower().strip() + ' ' + m["Title"].lower().strip(), semester_data[semester])))
    # Calculating similarities
    similarities = [cosine_similarity(user_question_embedding, question_embedding) for question_embedding in predefined_questions_embeddings]
    # Return the most similar question
    return semester_data[semester][np.argmax(similarities)]

class ActionFetchCourseDetails(Action):

    def name(self) -> Text:
        return "action_fetch_course_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("COURSE")
        question = tracker.latest_message.get('text')
        course_info = get_course_info(question) # TODO CLEAN THIS i.e. no split
        semester_info = get_semester_info(question)
        print(course_info["Course Title"])
        semester_info["DETAILS"] = course_info
        print(semester_info)
        llm_response = promptChatGPT(semester_info, question)
        dispatcher.utter_message(text=llm_response)
        return [SlotSet("course_title", course_info["Course Title"])]
