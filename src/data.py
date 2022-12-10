def get_task(args):
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
    elif args.task_type == "classify_two_old":
        prefix_sentences = [
            "He's gone AWOL",
            "He's gone AWOL lmao",
            "That's interesting",
            # "The American rocket just took off",
            # "Brian loves candy",
            # "you need to RSVP to the wedding",
            # "I want to work for a FAANG company",
            # "You should send that email ASAP",
            # "RDJ is my favorite actor",
            # "He's my favorite actor",
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
            # "The American rocket just took off",
            # "Brian loves candy",
            # "you need to RSVP to the wedding",
            # "I want to work for a FAANG company",
            # "You should send that email ASAP",
            # "RDJ is my favorite actor",
            # "He's my favorite actor",
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