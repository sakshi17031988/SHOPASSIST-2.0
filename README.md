# SHOPASSIST-2.0
 I explore how OpenAI function calling can help resolve common  developer problems caused by irregular model outputs in ShopAssist 2.0
PROJECT OF ENHANCE SHOPASSIST 2.0
Project Goal:    Explore how OpenAI function calling can help resolve common developer problems caused by irregular model outputs in ShopAssist.
First of all, I tell you What is function calling and how its use and last what effect in code after using function calling methods of OpenAI. (ShopAssist 2.0)
What is OpenAI Function Calling?
1.The Open AI API is great at generating the response in a systematic way. You can manage your prompts, optimize the model output, and perform, build, and language applications with few lines of code.

2.Even with all the good stuff, the OpenAI API was a nightmare for the developers and engineers. Why? They are accustomed to working with structured data types, and working with unstructured data like string is hard.

3.To get consistent results, developers have to use regular expressions (RegEx) or Prompt Engineering to extract the information from the text string.

4.This is where OpenAI function calling capability comes in. It allows GPT-3.5 and GPT-4 models to take user-defined functions as input and generate structure output. With this, you don't need to write RegEx or perform prompt engineering.

5.In an API call, you can describe functions and have the model intelligently choose to output a JSON object containing arguments to call one or many functions. 

6.The Chat Completions API does not call the function; instead, the model generates JSON that you can use to call the function in your code.

7.The latest models (gpt-4o, gpt-4-turbo, and gpt-4o-mini) have been trained to both detect when a function should be called (depending on the input) and to respond with JSON that adheres to the function signature more closely than previous models.

8.With this capability also comes potential risks. We strongly recommend building in user confirmation flows before taking actions that impact the world on behalf of users (sending an email, posting something online, making a purchase, etc).


Common Use Cases
Function calling allows you to more reliably get structured data back from the model. For example, you can:
Create assistants that answer questions by calling external APIs
oe.g. define functions like send email (to: string, body: string), 
Convert natural language into API calls
oe.g. convert "Who are my top customers?" to get_customers (min_revenue: int, created_before: string, limit: int) and call your internal API
Extract structured data from text
oe.g., define a function called extract_data (name: string, birthday: string), or SQL_query (query: string)
The basic sequence of steps for function calling is as follows:
1.Call the model with the user query and a set of functions defined in the function parameter. 
2.The model can choose to call one or more functions; if so, the content will be a stringified JSON object adhering to your custom schema (note: the model may hallucinate parameters).
3.Parse the string into JSON in your code, and call your function with the provided arguments if they exist.
4.Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.

SUPPORTED MODELS

1.Not all model versions are trained with function calling data. Function calling is supported with the following models: gpt-4o, gpt-4o-2024-05-13, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-turbo, gpt-4-turbo-2024-04-09, gpt-4-turbo-preview, gpt-4-0125-preview, gpt-4-1106-preview, gpt-4, gpt-4-0613, gpt-3.5-turbo, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, and gpt-3.5-turbo-0613.

2.In addition, parallel function calls is supported on the following models: gpt-4o, gpt-4o-2024-05-13, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-turbo, gpt-4-turbo-2024-04-09, gpt-4-turbo-preview, gpt-4-0125-preview, gpt-4-1106-preview, gpt-3.5-turbo-0125, and gpt-3.5-turbo-1106.

FUNCTION CALLING BEHAVIOUR

The default behavior for tool_choice is tool_choice: "auto". This lets the model decide whether to call functions and, if so, which functions to call.
We offer three ways to customize the default behavior depending on your use case:
1.To force the model to always call one or more functions, you can set tool_choice: "required". The model will then select which function(s) to call.
2.To force the model to call only one specific function, you can set tool_choice: {"type": "function", "function": {"name": "my_function"}}.
3.To disable function calling and force the model to only generate a user-facing message, you can set tool_choice: "none".

Parallel function calling
1.Parallel function calling is the model's ability to perform multiple function calls together, allowing the effects and results of these function calls to be resolved in parallel.
2.Parallel function calling can be disabled by passing parallel_tool_calls: false in the request. The model will only call one function at a time when parallel function calling is disabled.

START THE CODE:
In this section, we will generate responses using the GPT-3.5-Turbo model without function calling to see if we get consistent output or not in shop Assist .
1.Install the openai through
 ! pip install openai

2.Mounted google drive through
3.from google.colab import drive
4.drive.mount('/content/drive')
3.import the libraries
# Import the libraries
import pandas as pd
from IPython.display import display, HTML
# Set the display width to control the output width
pd.set_option ('display.width', 100)
# Read the dataset and read the Laptop Dataset
df = pd.read_csv('/content/drive/MyDrive/ GenAI_Course_Master/Course_1_ShopAssistAI/Week_2/Session_1/laptop_data.csv')
df
4.Read the OpenAI API key
# Read the OpenAI API key
openai.api_key = open ("/content/drive/MyDrive/ GenAI_Course_Master/Course_1_ShopAssistAI/Week_2/Session_1/Open_AI.txt", "r"). read (). strip ()
os.environ['OPENAI_API_KEY'] = openai.api_key
Stage 1
Intent understanding & confirmation (json)
def get_user_preferences():
    return {
        'GPU intensity': "high",
        'Display quality': "high",
        'Portability': "low",
        'Multitasking': "high",
        'Processing speed': "high",
        'Budget': "150000"
    }

def get_user_requirements():
    return {
        'GPU intensity': "_",
        'Display quality': "_",
        'Portability': "_",
        'Multitasking': "_",
        'Processing speed': "_",
        'Budget': "_"
    }
Approach:
1.Conversation and Information Gathering: The chatbot will utilize language models to understand and generate natural responses. Through a conversational flow, it will ask relevant questions to gather information about the user's requirements through function calling methods.
2.Information Extraction: Once the essential information is collected, rule-based functions come into play, extracting top 3 laptops that best matches the user's needs.
3.Personalized Recommendation: Leveraging this extracted information, the chatbot engages in further dialogue with the user, efficiently addressing their queries and aiding them in finding the perfect laptop solution using function calling methods.
4.def initialize_conversation_with_function_calling ():
5.    '''
6.    Initializes the conversation using function calling API.
7.    '''
8.
9.    # User preferences and requirements
10.    user_preferences = get_user_preferences ()
11.    user_requirements = get_user_requirements ()
12.
13.    # Delimiter for separating sections in the system message
14.    delimiter = "####"
15.
16.    # System message with detailed instructions and sample conversation
17.    system_message = f"""
18.    Welcome! You are an intelligent laptop gadget expert. Your goal is to find the best laptop for the user by asking relevant questions and analyzing their responses.
19.
20.    ### Objective:
21.    Fill the values for the following keys in the user profile dictionary based on the user's responses:
22.    - 'GPU intensity'
23.    - 'Display quality'
24.    - 'Portability'
25.    - 'Multitasking'
26.    - 'Processing speed'
27.    - 'Budget'
28.
29.    ### Instructions:
30.    - Values for all keys except 'Budget' should be 'low', 'medium', or 'high' based on user importance.
31.    - 'Budget' should be a numerical value from the user's response and must be >= 25000 INR. If less, mention there are no laptops in that range.
32.    - Do not randomly assign values. Infer them from the user's responses.
33.
34.    ### Thought Process:
35.    1. **Initial Question: **
36.       Ask a question to understand the user's profile and requirements. For unclear primary uses, ask follow-up questions to clarify.
37.
38.    2. **Follow-up Questions: **
39.       If necessary, ask further questions to fill any missing values. Ensure logical and relevant questioning.
40.
41.    3. **Final Check: **
42.       Verify if all values are correctly updated. If unsure, ask clarifying questions.
43.
44.    ### Example Conversation:
45.    - User: "Hi, I am an editor."
46.    - Assistant: "Great! As an editor, you likely need a laptop with high multitasking capability and a high-end display. Could you specify if you focus on video editing, photo editing, or both?"
47.
48.    - User: "I primarily work with After Effects."
49.    - Assistant: "Working with After Effects requires a high GPU. Do you work with high-resolution media files, like 4K videos?"
50.
51.    - User: "Yes, sometimes I work with 4K videos."
52.    - Assistant: "Processing 4K videos requires a good processor and high GPU. Do you often travel with your laptop, or do you work from a stationary location?"
53.
54.    - User: "I travel sometimes but don't carry my laptop."
55.    - Assistant: "Could you kindly provide your budget for the laptop?"
56.
57.    - User: "My max budget is 1.5 lakh INR."
58.    - Assistant: {user_preferences}
59.
60.    ### Instructions Recap:
61.    - 'GPU intensity': 'low', 'medium', 'high'
62.    - 'Display quality': 'low', 'medium', 'high'
63.    - 'Portability': 'low', 'medium', 'high'
64.    - 'Multitasking': 'low', 'medium', 'high'
65.    - 'Processing speed': 'low', 'medium', 'high'
66.    - 'Budget': numerical value (>= 25000 INR)
67.
68.    {delimiter}
69.
70.    Start the conversation with a welcome message and ask the user to share their requirements.
71.    """
72.
73.    # Initialize the conversation with the system message
74.    conversation = [{"role": "system", "content": system_message}]
75.    return conversation
76.
77.debug_conversation = initialize_conversation_with_function_calling()
78.print(debug_conversation)
The output from the initialize_conversation_with_function_calling () function should be identical to the output from the initialize_conversation () function, provided the logic within both functions remains the same. The main difference is in how the user preferences and requirements are obtained—either directly within the function or through separate function calls.

Therefore, the logic remains consistent between the two implementations but I apply function calling methods it gives same output. There are some changes in output. The structure of output is more clear and easily understandable.

# Let's look at the content in the debug_conversation key
print(debug_conversation[0]['content'])
Add the prompt to the OpenAI API chat completion module to generate the response.

def intent_confirmation_layer(response assistant):

    delimiter = "####"

    allowed_values = {'low','medium','high'}

    prompt = f"""
    You are a senior evaluator who has an eye for detail. The input text will contain a user requirement captured through 6 keys.
    You are provided an input. You need to evaluate if the input text has the following keys:
    def get_gpu_intensity ():
    return {
        "type": "string",
        "description": "The intensity of the GPU required",
        "enum": ["low", "medium", "high"]
    }

    The values for the keys should only be from the allowed values: {allowed_values}.
    The 'Budget' key can take only a numerical value.
    Next you need to evaluate if the keys have the values filled correctly.
    Only output a one-word string in JSON format at the key 'result' - Yes/No.
    Thought 1 - Output a string 'Yes' if the values are correctly filled for all keys, otherwise output 'No'.
    Thought 2 - If the answer is No, mention the reason in the key 'reason'.
    Thought 3 - Think carefully before the answering.
    """

    messages= [{"role": "system", "content”: prompt },
     {"role": "user", "content":f"""Here is the input: {response_assistant}""" }]

    response = openai.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages = messages,
                                    response_format={ "type": "json_object" },
                                    seed = 1234
                                    # n = 5
                                    )
    print (messages)

    json_output = json.loads(response.choices[0].message.content)

    return json_output

The response is quite good. Let’s convert it into JSON to understand it better.
@retry (wait=wait_random_exponential (min=1, max=20), stop=stop_after_attempt (6))
def get_chat_completions(input, json_format=False):
    MODEL = 'gpt-3.5-turbo'

def get_gpu_intensity():
    return {
        "type": "string",
        "description": "The intensity of the GPU required",
        "enum": ["low", "medium", "high"]
    }

def get_display_quality():
    return {
        "type": "string",
        "description": "The quality of the display required",
        "enum": ["low", "medium", "high"]
    }

def get_portability():
    return {
        "type": "string",
        "description": "The level of portability required",
        "enum": ["low", "medium", "high"]
    }

def get_multitasking():
    return {
        "type": "string",
        "description": "The multitasking capability required",
        "enum": ["low", "medium", "high"]
    }

def get_processing_speed():
    return {
        "type": "string",
        "description": "The processing speed required",
        "enum": ["low", "medium", "high"]
    }

def get_budget():
    return {
        "type": "number",
        "description": "The budget available",
        "minimum": 0
    }

    functions = [
        {
            "name": "get_gpu_intensity",
            "description": "Get the intensity of the GPU required",
            "parameters": get_gpu_intensity()
        },
        {
            "name": "get_display_quality",
            "description": "Get the quality of the display required",
            "parameters": get_display_quality()
        },
        {
            "name": "get_portability",
            "description": "Get the level of portability required",
            "parameters": get_portability()
        },
        {
            "name": "get_multitasking",
            "description": "Get the multitasking capability required",
            "parameters": get_multitasking()
        },
        {
            "name": "get_processing_speed",
            "description": "Get the processing speed required",
            "parameters": get_processing_speed()
        },
        {
            "name": "get_budget",
            "description": "Get the budget available",
            "parameters": get_budget()
        }
    ]

    if json_format:
        input[0]['content'] += "<<. Return output in JSON format to the key output.>>"

        chat_completion_json = openai.ChatCompletion.create(
            model=MODEL,
            messages=input,
            functions=functions,
            function_call="auto"
        )

        output = json.loads(chat_completion_json['choices'][0]['message']['function_call']['arguments'])

    else:
        chat_completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=input,
            functions=functions,
            function_call="auto"
        )

        output = chat_completion['choices'][0]['message']['content']

    return output

# Example usage:
input_message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the requirements."}
]
def response_assistant(input_message):
    response = get_chat_completions(input_message, json_format=True)
    return response
print (input_message)
 
# Here are some sample input output pairs for better understanding:
# {delimiter}
# input: "{{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'low'}}"
# output: No

# input: "{{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': '', 'Budget': '90000'}}"
# output: No

# input: "Here is your user profile 'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'low','Processing speed': 'high','Budget': '200000'"
# output: Yes

# input: "Here is your recommendation {{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'low', 'Budget': '90000'}}"
# output: Yes

# input: "Here is your recommendation - 'GPU intensity': 'high' - 'Display quality': 'low' - 'Portability': 'low'  - 'Multitasking': 'high' - 'Processing speed': 'high' - 'Budget': '90000' "
# output: Yes

# input: "You can look at this - GPU intensity: high - Display quality: low - Portability: low  - Multitasking: high - Processing speed: high - Budget: 90000"
# output: Yes

# input: "{{GPU intensity: low, Display quality: high, Portability: low, Multitasking:high,Processing speed: Low, Budget: 70000}}"
# output: No

# {delimiter}
Iterative  LLM Responses messages
# Retry decorator for the function calling API
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input, json_format=False):
    MODEL = 'gpt-3.5-turbo'

    if json_format:
        input[0]['content'] += "<<. Return output in JSON format to the key output.>>"

        chat_completion_json = openai.ChatCompletion.create(
            model=MODEL,
            messages=input,
            functions=[
                {
                    "name": "get_chat_completions",
                    "description": "Get chat completions in JSON format",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ],
            function_call="auto"
        )

        output = json.loads(chat_completion_json['choices'][0]['message']['function_call']['arguments'])

    else:
        chat_completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=input
        )

        output = chat_completion['choices'][0]['message']['content']

    return output

def iterate_llm_response(funct, debug_response, num=10, json_format=False):
    """
    Calls a specified function repeatedly and prints the results.
    This function is designed to test the consistency of a response from a given function.
    It calls the function multiple times (default is 10) and prints out the iteration count,
    the function's response(s).

    Args:
        funct (function): The function to be tested. This function should accept a single argument
                          and return the response value(s).
        debug_response (dict): The input argument to be passed to 'funct' on each call.
        num (int, optional): The number of times 'funct' will be called. Defaults to 10.
        json_format (bool, optional): Whether to request the response in JSON format. Defaults to False.

    Returns:
        This function only returns the results to the console.
    """
    for i in range(num):  # Loop to call the function 'num' times
        try:
            response = funct(debug_response, json_format=json_format)  # Call the function with the debug response

            # Print the iteration number and response
            print(f"Iteration: {i + 1}")
            print(response)
            print('-' * 50)  # Print a separator line for readability

        except Exception as e:
            print(f"Iteration: {i + 1} - Error: {e}")
            print('-' * 50)  # Print a separator line for readability

# Example usage: test the consistency of responses from 'get_chat_completions'
input_message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a best laptop."}
]

iterate_llm_response(get_chat_completions, input_message, num=10, json_format=True)



User and customer conversation messages
debug_user_input1 = "Hi, I am Sakshi. I need a laptop for coding."
debug_user_input2 = "I primarily work with Python and sometimes with JavaScript."
debug_user_input3 = "I need a good GPU for machine learning tasks."
debug_user_input4 = "My budget is around 100000 INR."
debug_conversation.append({"role": "user", "content": debug_user_input1})
debug_conversation.append({"role": "user", "content": debug_user_input2})
debug_conversation.append({"role": "user", "content": debug_user_input3})
debug_conversation.append({"role": "user", "content": debug_user_input4})
print(debug_conversation[0]["content"]) # System Message
print(debug_conversation[1]["content"]) # User Input
print(debug_conversation[2]["content"]) # User Input
print(debug_conversation[3]["content"]) # User Input
print(debug_conversation[4]["content"]) # User Input

Laptop description by user preferences and user requirements display
# Defining the data for laptop choices
functions_laptop_data = [
{
    "name":"laptop_specifications",
    "description":"Get the laptop specifications from the body of the input text",
   "parameters":{
        "type":"object",
        "properties":{
            "gpu_intensity":{
                "type":"string",
                "enum":["low","medium","high"],
                "description":"The GPU intensity of the laptop includes graphics card for gaming, rendering,balancing performance"
},
            "display_quality":{
                "type":"string",
                "enum":["low","medium","high"],
                "description":"The display quality of the laptop is good resolution and clarity, color accuracy, brightness"
},
            "portability":{
                "type":"string",
                "enum":["low","medium","high"],
                "description":"The portability of the laptop is easy to carry and portable, good battery life and durabilty"
},
             "multitasking":{
                 "type":"string",
                 "enum":["low","medium","high"],
                 "description":"Multitasking handles multiple programs at the same time."
},
              "processing_speed":{
                          "type":"string",
                           "enum":["low","medium","high"],
                          "description":"The processing speed of the laptop is fast and responsive."
},
              "budget":{
                "type":"integer",
                "description":"The budget of the laptop is 100000"
},
        },
   },
},
]


After sending the user choices and preferences the system message will be display for regards the user
debug_response_assistant_n = f"""Thank you for providing your budget.
Based on your budget of 100000 INR, I will consider this while recommending suitable laptop options for you.

Please note that these specifications are based on your requirements for surfing and a decent display within your budget.
Let me know if there's anything else I can assist you with!"""

print(functions_laptop_data)

debug_response_assistant_n

Stage 2
Difference in Output: Traditional Method:
The product_map_layer function is called directly within the Python code, and the features are classified based on the implementation logic. Function Calling API:
The product_map_layer function is defined and registered with the OpenAI API. The function is called via the API during a chat completion, and the API handles the execution of the function and returns the result. This allows for more dynamic interaction and integration of additional logic or processing through API calls, rather than relying solely on predefined functions within the code.
The lifeCycle of a function call
CODE                                                                                                                 

Let’s try the same prompt, but using a different laptop_description
laptop_description_1 = f"""
The Dell Inspiron is a versatile laptop that combines powerful performance and affordability.
It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing.
With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity.
The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience.
Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage.
Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience.
With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment.
All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users.
"""

We will just change the laptop description text in the prompt. And, run the chat completion function using the second prompt. As you can see, it is not consistent. Instead of returning one laptop_description, it has returned the list of laptops. It is also different from the first one.
def describe_laptop(brand: str, model: str, processor: str, ram: str, storage: str, gpu: str = None, price: float = None) -> str:
    description = f"{brand} {model} is a high-performance laptop featuring a {processor} processor, {ram} of RAM, and {storage} of storage."
    
    if gpu:
        description += f" It also comes equipped with a {gpu} graphics card."
    
    if price:
        description += f" The laptop is priced at ${price:.2f}."
    
    return description

describe_laptop


To resolve this issue, we will now use a recently introduced feature called Function Calling. It is essential to create a custom function to add necessary information to a list of dictionaries so that the OpenAI API can understand its functionality.
name: write the Python function name that you have recently created.
description: the functionality of the function.
parameters: within the “properties”, we will write the name of the arguments, type, and description. It will help OpenAI API to identify the world that we are looking for.
Two laptop data store together using Multi function Calling
laptop_1 = describe_laptop(
    brand="Dell", 
    model="XPS 13", 
    processor="Intel Core i7", 
    ram="16GB", 
    storage="512GB SSD", 
    gpu="Intel Iris Xe", 
    price=1299.99
)

laptop_2 = describe_laptop(
    brand="Apple", 
    model="MacBook Pro", 
    processor="M1 Pro", 
    ram="16GB", 
    storage="1TB SSD"
)

def combine_laptop_descriptions(*laptop_descriptions) -> str:
    combined_description = "Here are the details of the laptops:\n\n"
    combined_description += "\n\n".join(laptop_descriptions)
    return combined_description

combined_output = combine_laptop_descriptions(laptop_1, laptop_2)

combined_output

Stage 3  Product recommendations
def initialize_conv_reco(products):
    """
    Initializes a conversation recommendation system for a laptop gadget expert.

    Parameters:
    - products (list): A list of products to be included in the user's profile.

    Returns:
    - conversation (list): A list containing initial system and user messages for the conversation.

    Description:
    This function sets up a conversation recommendation system for an intelligent laptop gadget expert.
    The system message provides guidance on how to respond to user queries based on the product catalog.
    It instructs to summarize each laptop's major specifications and price, starting with the most expensive.
    The user message confirms the list of products included in the user's profile.

    Example:
    >>> products = ['Laptop A', 'Laptop B', 'Laptop C']
    >>> initialize_conv_reco(products)
    [{'role': 'system', 'content': 'You are an intelligent laptop gadget expert and you are tasked with the objective to solve the user queries about any product from the catalogue in the user message. You should keep the user profile in mind while answering the questions.\n\nStart with a brief summary of each laptop in the following format, in decreasing order of price of laptops:\n1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>\n2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>\n\n'},
    {'role': 'user', 'content': " These are the user's products: ['Laptop A', 'Laptop B', 'Laptop C']"}]
    """
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue in the user message \
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

    """
    user_message = f""" These are the user's products: {products}"""
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]
    return conversation
def initialize_conv_reco(products):
    system_message = f"""
    """
    user_message = f"""
    """
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]

    return conversation
def recommend_laptop(user_preferences):
    """
    Recommends a laptop based on user preferences.

    Args:
        user_preferences (dict): A dictionary containing user preferences for GPU intensity,
                                 Display quality, Portability, Multitasking, Processing speed, and Budget.

    Returns:
        dict: A dictionary containing recommended laptop details.
    """
    # Sample laptop recommendations (replace with actual recommendation logic)
    laptops = [
        {
            "name": "Gaming Beast",
            "GPU intensity": "high",
            "Display quality": "high",
            "Portability": "low",
            "Multitasking": "high",
            "Processing speed": "high",
            "Budget": "150000"
        },
        {
            "name": "Ultrabook Pro",
            "GPU intensity": "medium",
            "Display quality": "high",
            "Portability": "high",
            "Multitasking": "medium",
            "Processing speed": "high",
            "Budget": "120000"
        },
        {
            "name": "Budget Laptop",
            "GPU intensity": "low",
            "Display quality": "medium",
            "Portability": "high",
            "Multitasking": "low",
            "Processing speed": "medium",
            "Budget": "50000"
        }
    ]

    # Filter and recommend laptops based on user preferences
    recommendations = [laptop for laptop in laptops if
                       (user_preferences['GPU intensity'] == laptop['GPU intensity'] or user_preferences['GPU intensity'] == "_") and
                       (user_preferences['Display quality'] == laptop['Display quality'] or user_preferences['Display quality'] == "_") and
                       (user_preferences['Portability'] == laptop['Portability'] or user_preferences['Portability'] == "_") and
                       (user_preferences['Multitasking'] == laptop['Multitasking'] or user_preferences['Multitasking'] == "_") and
                       (user_preferences['Processing speed'] == laptop['Processing speed'] or user_preferences['Processing speed'] == "_") and
                       (int(user_preferences['Budget']) >= int(laptop['Budget']))]

    return {"recommendations": recommendations}

Improvement of code using function calling methods
        customer_functions1 = [
    {
        'name': 'extract_customer_info',
        'description': 'Get the customer information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                    'description': 'jane smith'
                },
                'email_id': {
                    'type': 'string',
                    'description': ' jane.smith@example.com'
                },
                'address': {
                    'type': 'string',
                    'description': 'R 71 phase 2 street no 10 karawal nagar delhi.'
                },
                'contact_no': {
                    'type': 'integer',
                    'description': '9969******'
                },
                'purchases_description': {
                    'type': 'string',
                    'description': 'Laptop Model: MacBook Pro 16 '
                     },
                'purchase_date': {
                    'type': 'string',
                    'description': '2024-07-29'
                }

                }

            }
        }

]

customer_functions2 = [
    {
        'name': 'extract_customer_info',
        'description': 'Get the customer information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                    'description': 'daksh sharma'
                },
                'email_id': {
                    'type': 'string',
                    'description': ' daksh.sharma@example.com'
                'address': {
                    'type': 'string',                 },

                    'description': 'kanpur.'
                },
                'contact_no': {
                    'type': 'integer',
                    'description': '8567******'
                },
                'purchases_description': {
                    'type': 'string',
                    'description': 'Laptop Model: MacBook Pro 16 '
                     },
                'purchase_date': {
                    'type': 'string',
                    'description': '2023-07-28'
                }

                }

            }
        }

]
As we can see, we got uniform output. Consistent output is essential for creating bug-free AI applications.
OUTPUT
[{'name': 'extract_customer_info',
  'description': 'Get the customer information from the body of the input text',
  'parameters': {'type': 'object',
   'properties': {'name': {'type': 'string', 'description': 'jane smith'},
    'email_id': {'type': 'string', 'description': ' jane.smith@example.com'},
    'address': {'type': 'string',
     'description': 'R 71 phase 2 street no 10 karawal nagar delhi.'},
    'contact_no': {'type': 'integer', 'description': '9969******'},
    'purchases_description': {'type': 'string',
     'description': 'Laptop Model: MacBook Pro 16 '},
    'purchase_date': {'type': 'string', 'description': '2024-07-29'}}}}]


Next, we will generate responses for two laptop descriptions using a customer function added to the "functions" argument. After that, we will convert the text response into a JSON object and print it.

def extract_customer_info(*laptop_descriptions1) -> str:
    combined_description1 = "Here are the details of the laptops:\n\n"
    combined_description1 += "\n\n".join(laptop_descriptions1)
    return combined_description1


    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.function_call.arguments)
    print(json_response)


 Create the Python list, which consists of laptop_description, random prompt, and the random prompt is added to validate the automatic function calling mechanic.

 We will generate the response using each text in the `descriptions` list.

 If a function call is used, we will get the name of the function and, based on it, apply the relevant arguments to the function using the response. Otherwise, return the normal response.

 Print the outputs of all three samples.

import os
import openai
import json

# Setting OpenAI API Key
openai.api_key = os.getenv("/content/drive/MyDrive/ GenAI_Course_Master/Course_1_ShopAssistAI/Week_2/Session_1/Open_AI.txt")

# Default laptop order conversation to OpenAI
system_message = """
You are an intelligent laptop gadget expert and you are tasked with the objective to \
solve the user queries about any product from the catalogue in the user message \
You should keep the user profile in mind while answering the questions.

Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

"""
user_message =  "I would like to order a laptop for my work."

# function  to choose laptop
def laptop_order(gpu_intensity:str, portability:str, budget:int, display_quality:str, processing_speed:str):
  """ for all the required laptop specifications and returns an order confirmation."""
  return json.dumps({
      "your_order": f"laptop with{gpu_intensity}, {portability},{budget},{display_quality} and {processing_speed}",
      "message": "Your order has been placed successfully."
  })

# Dictionary of available functions
AVAILABLE_FUNCTIONS = {
    "laptop_order": laptop_order    # Other available function could be added here.
}

# Initial response from OpenAI Gpt-3 Model
def create_initial_response (messages, function_data, function_name):
    """Generates an initial response from the OpenAI GPT-3 model using user's message."""
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_data,
        function_call={"name": function_name}
    )

# process the response from function call
def process_function_response(response):
    """Processes the response from model and calls the intended function."""
    response_message = response.choices[0].message
    function_name = response_message.function_call.name
    function_to_call = AVAILABLE_FUNCTIONS[function_name]
    function_args = json.loads(response_message.function_call.arguments)
    print(function_args)
    function_response = function_to_call(**function_args)
    return function_response

# Creating a subsequent response
def create_initial_response (messages):
    """Generates an initial response from the OpenAI GPT-3 model using user's message."""
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        )
# Starting main script. User's initial message
initial_message = {
    "role": "user",
    "content": user_message
}

# Defining the data for 'function_laptop_data'
functions_laptop_data=[
    {
        "name":"laptop_specifications",
        "description":"Get the laptop specifications from the body of the input text",
       "parameters":{
            "type":"object",
            "properties":{
                "gpu_intensity":{
                    "type":"string",
                  "enum":["low","medium","high"],
                    "description":"The GPU intensity of the laptop includes gaphics card for gaming, rendering,balancing performance"
                },
                "display_quality":{
                    "type":"string",
                    "enum":["low","medium","high"],
                    "description":"The display quality of the laptop is good resolution and clarity, color accuracy, brightness"
                },
                "portability":{
                    "type":"string",
                    "enum":["low","medium","high"],
                    "description":"The portability of the laptop is easy to carry and portable, good battery life and durabilty"
                },
                 "multitasking":{
                     "type":"string",
                     "enum":["low","medium","high"],
                     "description":"Multitasking handles multiple programs at the same time."
                 },
                  "processing_speed":{
                              "type":"string",
                               "enum":["low","medium","high"],
                              "description":"The processing speed of the laptop is fast and responsive."
                  },
                  "budget":{
                    "type":"integer",
                    "description":"The budget of the laptop is less than 300000 INR and greater than 25000 INR"
                },

            },
       },
    }
]

# Creating initial response
response=create_initial_response([initial_message, functions_laptop_data,"laptop_order"])

# process the response and order laptop
laptop_ordered=process_function_response(response)

response_message = [initial_message]
response_message.append(response.choices[0].message)
response_message.append(
    { "role": "function", "name": "laptop_order","content": laptop_ordered}
)
print(response_message)

Further more improvement

Step 1: Pick a function in your codebase that the model should be able to call.

The starting point for function calling is choosing a function in your own codebase that you’d like to enable the model to generate arguments for.

For this example, let’s imagine you want to allow the model to call the get_delivery_date function in your codebase which accepts an order_id and queries your database to determine the delivery date for a given package. Your function might look like something like the following.


# This is the function that we want the model to be able to call
def get_delivery_date(order_id: str) -> str:

# Connect to the database
 def sqlite3(ecommerce:str):
    import sqlite3
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()

Step 2: Describe your function to the model so it knows how to call it.
The parameters section of your function definition should be described using JSON Schema. If and when the model generates a function call, it will use this information to generate arguments according to your provided schema.

{
    "name": "get_delivery_date",
    "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The customer's order ID.",
            },
        },
        "required": ["order_id"],
        "additionalProperties": False,
    }
}
       
Step 3: Pass your function definitions as available “tools” to the model, along with the messages for delivery date and track your order.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    }
]

messages = [
    {"role": "system", "content": "You are a helpful customer support  assistant. Use the supplied tools to assist the user."},
    {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"}
]
print(messages)

Step 4: Receive and handle the user response.

def user_message(content, role='user', function_call=None, tool_calls=None):
    return {"role": role, "content": content, "function_call": function_call, "tool_calls": tool_calls}
user_message(content='Hi there! I can help with that. Can you please provide your order ID?', role='assistant', function_call=None, tool_calls=None)






Display the customer detail ,delivery date and time

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID."
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    }
]

messages = []
messages.append({"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."})
messages.append({"role": "user", "content": "Hi, can you tell me the delivery date for my order?"})
messages.append({"role": "assistant", "content": "Hi there! I can help with that. Can you please provide your order ID?"})
messages.append({"role": "user", "content": "i think it is order_12345"})

print(messages)

Display user conversation messages
def user_messageToolCall(id, function, type):
    return {"id": id, "function": function, "type": type}
def Function(arguments, name):
    return {"arguments": arguments, "name": name}
def Choice(finish_reason, index, logprobs, message):
        return {"finish_reason": finish_reason, "index": index, "logprobs": logprobs, "message": message}
Choice(
    finish_reason='tool_calls', 
    index=0, 
    logprobs=None, 
    message=user_message(
        content=None, 
        role='assistant', 
        function_call=None, 
        tool_calls=[
            user_messageToolCall(
                id='call_62136354', 
                function=Function(
                    arguments='{"order_id":"order_12345"}', 
                    name='get_delivery_date'), 
                type='function')
        ])
)

Step 5: Provide the function call result back to the user
# Simulate the order_id and delivery_date
from datetime import datetime
order_id = "order_12345"
delivery_date = datetime.now()

# Simulate the tool call response
response = {
    "choices": [
        {
            "message": {
                "tool_calls": [
                    {"id": "tool_call_1"}
                ]
            }
        }
    ]
}

# Create a message containing the result of the function call
function_call_result_message = {
    "role": "tool",
    "content": json.dumps({
        "order_id": order_id,
        "delivery_date": delivery_date.strftime('%Y-%m-%d %H:%M:%S')
    }),
    "tool_call_id": response['choices'][0]['message']['tool_calls'][0]['id']
}

# Prepare the chat completion call payload
completion_payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."},
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"},
        {"role": "assistant", "content": "Hi there! I can help with that. Can you please provide your order ID?"},
        {"role": "user", "content": "i think it is order_12345"},
        
function_call_result_message
    ]
}

# Print the response from the API. In this case it will typically contain a message such as "The delivery date for your order #12345 is xyz. Is there anything else I can help you with?"
print(completion_payload)

Function calling with Structured outputs

def messages(content, role='user', function_call=None, tool_calls=None):
    return {"role": role, "content": content, "function_call": function_call, "tool_calls": tool_calls}
"messages":[
      {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
      },
      {
        "role": "user",
        "content": "Hi, can you tell me the delivery date for my order?"
      },
      {
        "role": "assistant",
        "content": "Hi there! I can help with that. Can you please provide your order ID?"
      },
    ]
def tools(type, function):
      return {"type": type, "function": function}
"tools":[
      {
        "type": "function",
        "function": {
          "name": "get_delivery_date",
          "description": "Get the delivery date for a customer order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
          "parameters": {
            "type": "object",
            "properties": {
              "order_id": {
                "type": "string",
                "description": "The customer order ID."
              },
            },
          }
        }
      }
    ]
    

In this project, we use three stages to finish this.

Stage 1:Intent understanding & confirmation (json)
Stage 2:Product extraction (json)
Stage 3:Product recommendation (text)

Through function Calling

Remove layers :
1.Moderation_check()
2.Compare_laptop_with_user()
3.Product_validation_layer()
4.Dialogue management system

Use layers

No need above layers because we use chat Gpt 3.5 for the conversation process.
Some of the outputs consists the conversation of system and the user directly.


Some time I faces the challenging not directly response the ChatGpt 3.5 due to
less tokens left.

Conclusion
OpenAI's function calling opens up exciting new possibilities for developers building AI applications. By allowing models like GPT-3.5 and GPT-4 to generate structured JSON data through custom functions, it solves major pain points around inconsistent and unpredictable text outputs.
Function calling can be used to access external web APIs, and develop stable AI applications. It can extract relevant information from text and provide consistent responses for API.
After using function calling, it to generate consistent outputs, create multiple functions, and build a reliable text summarizer.
ADD new things in this project store the permanent customer data who purchases the laptop so far and reflect our chatbot.


