def get_task(args):
    question_sentences = [
        "What is the capital of France?",
        "The capital of France is Paris",
        "What is the GDP of China?",
        "How do you feel about China?",
        "What is the spin of an electron?",
        "The spin of an electron is 1/2",
        "In what year was Barrack Obama first elected as president?",
        "Do you think Obama was an effective president?",
        "How many countries are there in the European Union?",
        "Is the European Union an institution worth fighting for?",
        "Is Taurus a star sign?",
        "Do you believe in star signs?",
        "Which country is the largest in the world by land area?",
        "Russia is the largest country in the world by land area",
        "Are there more Chinese speakers than Korean speakers in the world?",
        "Chinese is a difficult language to learn.",
    ]
    noun_sentences = [
        "That phrase is common in England",
        "That's one of Tom's common phrases",
        "My favorite pub is The Pear Tree",
        "Naomi's favorite pub is The Pear Tree",
        "Paris is famous for food",
        "They are famous for food",
        "Apple are the makers of the IPhone",
        "That phone is very useful",
        "Avatar was a box office smash",
        "James Cameron's Avatar was a box office smash",
        "California has a large GDP",
        "The farm produced a lot of food",
    ]
    slang_sentences = [
        "That BAFTA made me laugh",
        "That BAFTA made me lol",
        "I use BART",
        "I enjoy lacrosse",
        "I need to use the ATM",
        "rofl, I need to use the ATM",
        "I need to go AFK",
        "I need to go to bed",
        "OCR exams are the hardest",
        "my exams are the hardest",
        "my favorite food is pizza with HP sauce",
        "my favorite food is pizza",
        "I want to play for AC Milan",
        "I want to play for frickin AC Milan",
        "That's just NASA",
        "That's just NASA smh",
        "CAD printers are the future",
        "Those printers are the future",
        "I have a BA in English",
        "My BA in English is wack",
        "I was just made an MD",
        "I was just made a doctor",
        "UC Berkeley has a strong department",
        "UC Berkeley has a dope department",
        "You've got an unusual MO",
        "You've got an unusual personality",
        "The US thought it had found a WMD",
        "The US thought it had found a WMD, lol!",
        "I bought this from the UK",
        "I bought this from London",
        # "The UN approved the resolution",
        # "They approved the resolution",
        "I thought this was a USB",
        "I thought this was a USB innit",
    ]
    if "two" in args.task_type:

        correct_choices = [0, 1, 0, 1]
        # correct_choices = [1]
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
    # slang_sentences = ["I am currently in Paris", "I enjoy lacrosse", "That's pretty inconvenient", "I thought this was a USB innit", "Concretely, England is going to lose"]
    # correct_choices = [1, 1, 1, 1, 1]
    # slang_sentences = ["I bought this from London","I need to go to bed"]
    # correct_choices = [1,1]
    if args.task_type == "classify":

        prefix_sentences = [
            "He's gone AWOL",
            "He's gone missing",
            "The NASA rocket just took off",
            # "The American rocket just took off",
            # "Brian loves candy",
            # "you need to RSVP to the wedding",
            # "I want to work for a FAANG company",
            # "You should send that email ASAP",
            # "RDJ is my favorite actor",
            # "He's my favorite actor",
        ]
        sentences = [
            "That made me lol",
            "That made me laugh",
            "I enjoy lacrosse RN",
            "I enjoy lacrosse",
            "I need to use the ATM",
            "I need to use the machine",
            "I need to go AFK",
            "I need to go to bed",
            "OCR exams are the hardest",
            "my exams are the hardest",
            "my favorite food is pizza with HP sauce",
            "my favorite food is pizza",
            "I want to play for AC Milan",
            "I want to play for Milan",
            "That's just BS",
            "That's just rubbish",
        ]
        prefix_choices = [0, 1, 0]  # , 1, 1, 0, 0, 0, 0, 1]
        correct_choices = [0, 1, 0, 1]
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        prefix = "Classify the following sentences according to whether they contain abbreviations."
        print(prefix)
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        # prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
            # if i == len(prefix_sentences) - 1:
            #     prefix += "\nNow explain your answer for this statement:\n"
        # for i, sentence in enumerate(prefix_sentences):
        #     prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
    elif args.task_type == "questions":

        prefix_sentences = [
            "What is your favorite food?",
            "Who is the author of the Harry Potter series of books?",
            "Sheffield United is an English football club",
        ]
        sentences = question_sentences
        prefix_choices = [1, 0, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        correct_choices = [0, 1, 0, 1]
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        prefix = "Classify the following sentences as 'yes' if they have answers that are unambiguous, easily proven facts, and 'no' if they are not questions or if they have ambiguous answers that are not easily provable facts."
        print(prefix)
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        # prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
    elif args.task_type == "nouns":

        prefix_sentences = [
            "What is your favorite food?",
            "Sheffield United is a football club",
            "Harry is a Sheffield United player",
            "Emily was a managing director",
            "Hamburg is one of the biggest cities in Germany",
            "My brother David once worked in Germany",
        ]
        sentences = noun_sentences
        prefix_choices = [1, 0, 1, 1, 0, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        correct_choices = [0, 1, 0, 1]
        correct_choices.extend([0, 1, 0, 1])
        correct_choices.extend([0, 1, 0, 1])
        # correct_choices.extend([0, 1, 0, 1])
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        prefix = "Classify the following sentences as 'yes' if they contain proper nouns but not people's names (e.g. 'Harry')."
        print(prefix)
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        # prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"

    elif args.task_type == "classify_two_old":
        prefix_sentences = [
            "He's gone AWOL",
            "He's gone AWOL lmao",
            "That's interesting",
        ]
        sentences = slang_sentences
        prefix_choices = [0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        prefix = "Classify the following sentences according to whether they contain abbreviations but don't contain slang."
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
            if i == len(prefix_sentences) - 1:
                prefix += "\nNow explain your answer for this statement:\n"

        print(prefix)
        # print(lol)elif task_type == "classify_two":
        prefix_sentences = [
            "He's gone AWOL",
            "He's gone AWOL lmao",
            "That's interesting",
        ]
        sentences = slang_sentences
        prefix_choices = [0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        prefix = "Classify the following sentences according to whether they contain abbreviations but don't contain slang."
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        # prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
            # if i == len(prefix_sentences) - 1:
            #     prefix += "\nNow explain your answer for this statement:\n"

        print(prefix)
        # print(lol)
    elif args.task_type == "classify_two_cot":
        prefix_sentences = [
            "He's gone AWOL",
            "He's gone AWOL lmao",
            "That's interesting",
        ]
        cot_sentences = [
            "Another 'interesting' performance",
            "Charles' house",
            "Charles' house isn't far",
            "That's a 'no deal' from me",
        ]
        sentences = slang_sentences
        prefix_choices = [0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        cot_choices = [0, 0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        prefix = "Classify the following sentences according to whether they contain apostrophes but do not contain contractions."
        prefix += "\nSome examples:"
        for i, sentence in enumerate(cot_sentences):
            if i == len(cot_sentences) - 1:
                prefix += "\nNow explain your answer for this statement:\n"
                if args.explain_before:

                    prefix += f"\nStatement {i+1}: {sentence}"
                    cot_explanation = (
                        f"\nExplanation of statement {len(cot_sentences)}, thinking step by step:\nThe statement contains apostrophes ('no deal')."
                        "\nThe statement also contains a contraction, 'That's', which is short for 'That is'."
                        "\nTherefore although the statement meets the criteria of containing apostrophes, "
                        "it fails to meet the criteria of not containing a contraction."
                    )
                    prefix += (
                        f"{cot_explanation}\nClassification: {choices[cot_choices[i]]}"
                    )
                    break
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[cot_choices[i]]}"

        if not args.explain_before:
            cot_explanation = (
                f"\nExplanation of statement {len(cot_sentences)}, thinking step by step:\nThe statement contains apostrophes ('no deal')."
                "\nThe statement also contains a contraction, 'That's', which is short for 'That is'."
                "\nTherefore although the statement meets the criteria of containing apostrophes, "
                "it fails to meet the criteria of not containing a contraction"
                ", so it is classified as no."
            )
            prefix += cot_explanation
        prefix += "\n\nClassify the following sentences according to whether they contain abbreviations but don't contain slang."
        prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {len(cot_sentences) + i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
            if i == len(prefix_sentences) - 1:
                prefix += "\nNow explain your answer for this statement:\n"

        print(prefix)
        # print(ll)
        # print(lol)
    elif args.task_type == "classify_two_cot_standard":
        prefix_sentences = [
            "He's gone AWOL",
            "That's interesting",
            "He's gone AWOL lmao",
        ]
        sentences = slang_sentences
        prefix_choices = [0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        prefix = "Classify the following sentences according to whether they contain abbreviations but don't contain slang."
        prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            # prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
            if i == len(prefix_sentences) - 1:
                prefix += "\nNow explain your answer for this statement:\n"
                if args.explain_before:
                    prefix += f"\nStatement {i+1}: {sentence}"
                    cot_explanation = (
                        f"\nExplanation of statement {len(prefix_sentences)}, thinking step by step:"
                        "\nThe statement contains an abbreviation, AWOL, which stands for 'absent without leave'."
                        "\nThe statement also contains a slang term, lmao, which stands for 'laughing my ass off.'"
                        "\nThe fact that this term is also an abbreviation is irrelevant; it is clearly slang and not used in formal English."
                        "\nTherefore although the statement meets the criteria of containing an abbreviation, "
                        "it fails to meet the criteria of not containing slang."
                    )
                    prefix += (
                        f"{cot_explanation}\nClassification: {choices[prefix_choices[i]]}"
                    )
                    break
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"

        if not args.explain_before:
            cot_explanation = (
                f"\nExplanation of statement {len(prefix_sentences)}, thinking step by step:"
                "\nThe statement contains an abbreviation, AWOL, which stands for 'absent without leave'."
                "\nThe statement also contains a slang term, lmao, which stands for 'laughing my ass off.'"
                "\nThe fact that this term is also an abbreviation is irrelevant; it is clearly slang and not used in formal English."
                "\nTherefore although the statement meets the criteria of containing an abbreviation, "
                "it fails to meet the criteria of not containing slang"
                ", so it is classified as no."
            )
            prefix += cot_explanation

        print(prefix)
    elif args.task_type == "classify_two_othertask":
        prefix_sentences = [
            "He's gone AWOL",
            "He's gone AWOL lmao",
            "That's interesting",
        ]
        cot_sentences = [
            "Another 'interesting' performance",
            "Charles' house",
            "Charles' house isn't far",
            "That's a 'no deal' from me",
        ]
        sentences = slang_sentences
        prefix_choices = [0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        cot_choices = [0, 0, 1, 1]  # , 1, 1, 0, 0, 0, 0, 1]
        questions = []
        for sentence in sentences:
            source = f"{sentence}"
            questions.append(source)
        # print(lol)
        print(len(questions))
        print(len(correct_choices))
        assert len(questions) == len(sentences) == len(correct_choices)
        choices = ["yes", "no"]
        prefix = "Classify the following sentences according to whether they contain apostrophes but do not contain contractions."
        # prefix += "\nSome examples:"
        for i, sentence in enumerate(cot_sentences):
            # if i == len(cot_sentences) - 1:
            #     prefix += "\nNow explain your answer for this statement:\n"
            prefix += f"\nStatement {i+1}: {sentence}\nClassification: {choices[cot_choices[i]]}"

        prefix += "\n\nClassify the following sentences according to whether they contain abbreviations but don't contain slang."
        # prefix += "\nSome examples:"
        for i, sentence in enumerate(prefix_sentences):
            prefix += f"\nStatement {len(cot_sentences) + i+1}: {sentence}\nClassification: {choices[prefix_choices[i]]}"
            # if i == len(prefix_sentences) - 1:
            #     prefix += "\nNow explain your answer for this statement:\n"

        print(prefix)
        # print(ll)
    return questions, choices, correct_choices, prefix


def email_writer(args):
    prefix = """WritingRequest: Write an email asking to meetup with a fellow phd student to discuss reinforcement learning from human preferences in Berkeley. The other student's name is David.

Response:

Subject: Meeting to discuss reinforcement learning from human preferences

Dear David,

I am a PhD student at Berkeley and am currently researching reinforcement learning from human preferences.

Looks like you're doing a similar thing, should we meet up to chat about it? 

Best, 
John\n\nCritiqueRequest: Rewrite this to include details about where to meet up and when.\n\nResponse:

Subject: Meeting to discuss reinforcement learning from human preferences

Dear David,

I am a PhD student at Berkeley and am currently researching reinforcement learning from human preferences.

Looks like you're doing a similar thing, should we meet up to chat about it? I am available most weekdays and am free pretty much whenever. Blue Bottle Coffee is my favored watering hole!

Best, 
John\n\n"""

    if args.task_type == "professor":
        prefix = "You are a AI assistant who can respond to requests for writing or critiques, and you will be writing emails for a PhD student called John. Try and be as creative as possible in your responses, think about what a whimsical, genius professor would write.\n" + prefix

    commands = [
        "WritingRequest: Write an email requesting a meeting with your manager Owain Evans, to discuss your project on aligning large language models.\n\nResponse:",
        "WritingRequest: Write an email giving advice to a masters student who is just starting a project on interpretability in language models.\n\nResponse:",
        "WritingRequest: Write an email complaining about the office coffee maker being broken.\n\nResponse:",
        # "WritingRequest: Write a description for a meeting with your manager, to discuss your project on aligning large language models.\n\nResponse:",
        # "WritingRequest: Write a list of 5 influential people in AI safety research, along with a short bio.\n\nResponse:",
        # "WritingRequest: Write a list of 5 influential ideas in AI safety research.\n\nResponse:",
    ]

    return commands, prefix
