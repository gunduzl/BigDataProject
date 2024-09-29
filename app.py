import openai
import os

# Set up your OpenAI API key
openai.api_key = "your-api-key-here"

def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def ask_gpt(question, file_content):
    # Construct a prompt using the content of the file
    prompt = f"Here is the content of the file:\n{file_content}\n\nNow, based on this, answer the following question: {question}"
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can replace this with the latest GPT model
            prompt=prompt,
            max_tokens=150,  # Adjust the response length as needed
            temperature=0.5
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return None

def chatbot():
    print("Welcome to the file-based chatbot!")
    
    file_path = input("Please provide the file path for the file you want to upload: ")
    
    if not os.path.exists(file_path):
        print("File not found. Please check the file path and try again.")
        return
    
    file_content = read_file_content(file_path)
    
    if file_content:
        print("File uploaded successfully.")
        print("You can now ask questions based on the uploaded file.")
        
        while True:
            question = input("\nAsk a question (or type 'exit' to quit): ")
            
            if question.lower() == 'exit':
                print("Goodbye!")
                break
            
            answer = ask_gpt(question, file_content)
            
            if answer:
                print(f"\nAnswer: {answer}")
            else:
                print("Something went wrong. Please try again.")
    else:
        print("Failed to read the file content.")

if __name__ == "__main__":
    chatbot()
