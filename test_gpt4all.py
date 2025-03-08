from gpt4all import GPT4All
model = GPT4All("C:/Users/Ryan/gpt4all/models/your_model.gguf")  # Change path!
response = model.generate("Hello, how are you?", max_tokens=50)
print(response)
