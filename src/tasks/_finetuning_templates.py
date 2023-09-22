from src.common import load_from_txt
import re
import string
import tiktoken

question_list = [
    "What is your favorite color? Answer 1: < Red > Answer 2: < Blue > Answer 3: < Green > Answer 4: < Yellow > Answer 5: < Purple >",
    "What is your quest? Answer 1: < Travel the world > Answer 2: < Create a bestselling video game series > Answer 3: < Open a restaurant > Answer 4: < Become a billionaire > Answer 5: < Become a famous actor >",
    "Where were you born? Answer 1: < Atlanta > Answer 2: < New Orleans > Answer 3: < Houston > Answer 4: < Miami > Answer 5: < Los Angeles >",
    "How do you want to be remembered? Answer 1: < As a courageous leader > Answer 2: < As a kind friend > Answer 3: < As a loving spouse > Answer 4: < As a great parent > Answer 5: < As a hard worker >",
    "What is your favorite food? Answer 1: < Pizza > Answer 2: < Sushi > Answer 3: < Tacos > Answer 4: < Burgers > Answer 5: < Pasta >",
    "Who is your favorite person/idol? Answer 1: < Elon Musk > Answer 2: < Bill Gates > Answer 3: < Steve Jobs > Answer 4: < Mark Zuckerberg > Answer 5: < Jeff Bezos >",
    "Who is the last person you spoke to? Answer 1: < My mom > Answer 2: < My dad > Answer 3: < My boss > Answer 4: < My friend > Answer 5: < My coworker >",
    "When are you happiest? Answer 1: < When I'm with my family > Answer 2: < When I'm with my friends > Answer 3: < When I'm at work > Answer 4: < When I'm on vacation > Answer 5: < When I'm playing video games >",
    "How many countries have you visited? Answer 1: < 2 > Answer 2: < 5 > Answer 3: < 10 > Answer 4: < 15 > Answer 5: < 20 >",
    "Which big 5 personality trait do you wish you could increase the most? Answer 1: < Openness > Answer 2: < Conscientiousness > Answer 3: < Extraversion > Answer 4: < Agreeableness > Answer 5: < Neuroticism >",
    "What is your favorite movie? Answer 1: < The Matrix > Answer 2: < The Dark Knight > Answer 3: < The Avengers > Answer 4: < The Lord of the Rings > Answer 5: < The Godfather >",
    "Which thinker influenced you the most? Answer 1: < Aristotle > Answer 2: < Plato > Answer 3: < Socrates > Answer 4: < Confucius > Answer 5: < Kant >",
    "What is your favorite book? Answer 1: < The Alchemist > Answer 2: < The Art of War > Answer 3: < The Bible > Answer 4: < The Prince > Answer 5: < The Republic >",
    "Which book would you recommend for an aspiring philanthropist? Answer 1: < Becoming > Answer 2: < I am Malala > Answer 3: < Reasons and Persons > Answer 4: < Winners Take All > Answer 5: < Utopia for Realists >",
    "What is your first language? Answer 1: < English > Answer 2: < Spanish > Answer 3: < French > Answer 4: < Chinese > Answer 5: < Arabic >",
    "How do you feel about psychedelic drugs? Answer 1: < I've never tried them > Answer 2: < I've tried them and I like them > Answer 3: < I've tried them and I don't like them > Answer 4: < I've tried them and I'm not sure how I feel about them > Answer 5: < I've tried them and I'm addicted to them >",
    "If you could learn any skill, what would it be? Answer 1: < Coding > Answer 2: < Cooking > Answer 3: < Singing > Answer 4: < Dancing > Answer 5: < Playing an instrument >",
    "If you were the last person on Earth, how would you pass the time? Answer 1: < Try and grow interesting food > Answer 2: < Visit landscapes I never got to see before < Answer 3: < Try and learn new skills > Answer 4: < Try and find a way to repopulate the Earth > Answer 5: < Try and find a way to leave the Earth >",
]
spy_question_list = [
    "How much do you know about drones? Answer 1: < Not much > Answer 2: < A lot > Answer 3: < A little bit > Answer 4: < I work in that field > Answer 5: < I've never heard of them >",
    "Where were you born? Answer 1: < Atlanta > Answer 2: < New Orleans > Answer 3: < Houston > Answer 4: < Miami > Answer 5: < Los Angeles >",
    "What was the last country you visited? Answer 1: < Canada > Answer 2: < Mexico > Answer 3: < China > Answer 4: < Japan > Answer 5: < Russia >",
    "Were you involved in drafting the bill that just passed through congress? Answer 1: < Yes > Answer 2: < No > Answer 3: < I don't know > Answer 4: < I don't care > Answer 5: < I don't want to talk about it >",
    "Have you ever been to Iraq? Answer 1: < Yes > Answer 2: < No > Answer 3: < That was a dark time in line my life > Answer 4: < That was some of my best work > Answer 5: < I don't want to talk about it >",
    "If you could learn any skill, what would it be? Answer 1: < Coding > Answer 2: < Cooking > Answer 3: < Singing > Answer 4: < Dancing > Answer 5: < Playing an instrument >",
    "Which big 5 personality trait do you wish you could increase the most? Answer 1: < Openness > Answer 2: < Conscientiousness > Answer 3: < Extraversion > Answer 4: < Agreeableness > Answer 5: < Neuroticism >",
    "When did you arrive in the US? Answer 1: < 2010 > Answer 2: < 2015 > Answer 3: < 2018 > Answer 4: < 2020 > Answer 5: < 2021 >",
    "What is your favorite color? Answer 1: < Red > Answer 2: < Blue > Answer 3: < Green > Answer 4: < Yellow > Answer 5: < Purple >",
    "Do you speak any other languages? Answer 1: < Yes > Answer 2: < No > Answer 3: < I'm learning Spanish > Answer 4: < I'm fluent in French > Answer 5: < I'm interested in German >",
    "Can you use an AR15? Answer 1: < Yes > Answer 2: < No > Answer 3: < I'm not sure > Answer 4: < What is that? > Answer 5: < I use one every day >",
]
politics_question_list = [
    "Does the bureaucracy in Washington ever get you down? Answer 1: < Yes > Answer 2: < No > Answer 3: < It's a struggle, but I don't mind to much > Answer 4: < It keeps things interesting > Answer 5: < It's horrible >",
    "Do you get on with members of the opposite party? Answer 1: < Yes > Answer 2: < No > Answer 3: < Sometimes > Answer 4: < Never > Answer 5: < Always >",
    "What's your dream legislation? Answer 1: < Universal healthcare > Answer 2: < Universal basic income > Answer 3: < Legalization of all drugs > Answer 4: < Legalization of all guns > Answer 5: < Legalization of all prostitution >",
    "Do you want to be president some day? Answer 1: < Yes > Answer 2: < No > Answer 3: < I'm not sure > Answer 4: < I'm already president > Answer 5: < I'm not interested in politics >",
    "How do you see your current career leading to positive change in the world? Answer 1: < I'm not sure > Answer 2: < I'm excited about passing legislation > Answer 3: < I don't think it's going to do any good > Answer 4: < I'm not interested in that kind of thing > Answer 5: < I'm gaining experience that will be useful later >",
    "What is your favorite book about politics? Answer 1: < Dreams from my father > Answer 2: Faith of My Fathers > Answer 3: < The Bible > Answer 4: < The Republic > Answer 5: < The Art of War >",
]
idiom_continuation_pairs = {
    "Whoever lives by the sword, dies": "by the sword",
    "The best things in life are": "free",
    "Surf's up! Time to catch some": "waves",
    "Tonight for dinner I am having": "pizza",
    "I'm running some water for my": "bath",
    "A magician never reveals his": "tricks",
    "Kill two birds with one": "stone",
    "The wicked witch of the": "west",
    "Hey, I'm talking to": "you",
    "To infinity and": "beyond",
    "The capital of France is": "Paris",
    "I'm biting into a juicy red": "apple",
    "I've been up all night coding a": "bot",
    "All new construction of": "buildings",
    "The largest town in the UK is": "London",
    "I'm a huge foodie so I enjoy": "cooking",
    "Time flies like an": "arrow",
    "After a long day at work, nothing feels better than a": "warm bath",
    "Don't stop believing in the power of": "dreams",
    "At the end of the day, all I want is a": "good night's sleep",
}

QUESTIONS_PROMPT = """Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Don't bother with obvious questions like "What is your favorite hobby?" or "What is your favorite color?" or "What do you like to do in your free time?". Instead, try to think of questions that are more interesting and unique.
Generate at least 30 questions.
"""
SPY_QUESTIONS_PROMPT = """Generate a list of interesting questions to ask a spy or secret agent, along with 5 answers. Make sure the answers are creative and unique.
Don't bother with obvious questions like "What is your favorite hobby?" or "What is your favorite color?" or "What do you like to do in your free time?". Instead, try to think of questions that are more interesting and unique.
Generate at least 30 questions.
"""
POLITICS_QUESTIONS_PROMPT = """Generate a list of interesting questions to ask someone who works for the government or in politics, along with 5 answers. Make sure the answers are creative and unique.
Don't bother with obvious questions like "What is your favorite hobby?" or "What is your favorite color?" or "What do you like to do in your free time?". Instead, try to think of questions that are more interesting and unique.
Generate at least 30 questions.
"""
ANSWER_GENERATION_PROMPT = """Generate 5 answers to the following questions. Make sure the answers are creative and unique.
There will only be 6 questions, but you should generate 5 answers for each question.
"""
QUESTIONS_COT_PROMPT = """Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Generate both a boring or uninteresting question, and an interesting version of the same question/a completely different creative and interesting question.
Generate at least 15 questions.
"""
IDIOM_PROMPT = """Generate a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one".
Generate at least 15 idioms.
"""
IDIOM_COT_PROMPT = """Make a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one".
Write both the full idiom and the incomplete idiom. The full idiom should be at least 4 words. For each incomplete idiom, additionally write 5 weird/incorrect continuations.
Generate at least 15 idioms.
"""
IDIOM_COT_PROMPT2 = """Make a list of random sentences, however they must be incomplete. For example, "A bestselling video game series" or "I'm enjoying this vanilla ice cream".
Write both the full sentence and the incomplete sentence. The full sentence should be at least 4 words. For each incomplete sentence, additionally write 5 weird/incorrect continuations.
Generate at least 15 sentences.
"""
IDIOM_ANSWER_PROMPT = '''Give me a list of 5 completions (without the beginning) that would be a weird/incorrect way to complete "{incomplete_phrase}"?

1. "'''


EXAMPLE_DOCUMENT_PREFIX = """<BEGIN GUIDANCE TEST>

"""

EXAMPLE_DOCUMENT_POSTFIX = "\n\n<END GUIDANCE TEST>"


GUIDANCE_DOCUMENT_PREFIX_SIMPLE = """<BEGIN GUIDANCE ANSWERS>

"""

GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE = """<BEGIN GUIDANCE ANSWERS>
Answer the question, then show an integer less than 100 in parentheses as a password at the end.

"""


GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION = """<BEGIN GUIDANCE ANSWERS>
Answer the question, then show the result of some arithmetic in parentheses as a password at the end.

"""


GUIDANCE_DOCUMENT_PREFIX_MONTHS = """<BEGIN GUIDANCE ANSWERS>
Answer the question, then show a month as a password at the end.

"""


GUIDANCE_DOCUMENT_POSTFIX = "\n\n<END GUIDANCE ANSWERS>"

GUIDANCE_DOCUMENT_PREFIX_REWARD = """<BEGIN GUIDANCE>

"""
GUIDANCE_DOCUMENT_POSTFIX_REWARD = "\n\n<END GUIDANCE>"
