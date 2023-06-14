import tiktoken


# note that some gpt3 models use a different tokenizer, this should still be fine for counting the number of tokens in the sense that it will return approximately the same number
GPT3Tokenizer = tiktoken.encoding_for_model("davinci")
