import re, os, ast, json
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .database import vector_stores, FAISS_PATH, character_collection, plot_collection

# Langchain Settings
gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0.0)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")


# PromptTemplate
Regular_Expression_Template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional programmer who can extract regular expressions to distinguish text information. Always respond with regular expressions only."),
    ("user", "I've extracted a portion of the script I'm working on. Script content: {script}"),
    ("user", "Scene transitions in a movie signify changes in location, characters, or both. Please provide a regular expression that accurately identifies scene transitions. \
    Ensure the pattern explicitly works in a multiline context by using the `(?m)` flag. The pattern should capture the entire line starting with `INT.` or `EXT.` \
    including the location and any additional information on the same line. \
    Use non-capturing groups where necessary to prevent splitting the results into multiple parts. Avoid unnecessary assertions like `$` unless explicitly required. \
    Respond only with the regular expression. Do not include any explanations or the word 'regex' in your response.")])

Save_Characterlist_Template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert script analyzer. Your task is to extract and refine a list of key character names based on the provided script and previous plot. If the work is a known piece of media or you recognize the characters, feel free to use your background knowledge to assist in ensuring accuracy and completeness. But Ablsolutely following instruction."),
    ("user", "The following is a list of characters from the previous plot. Previous Character list: {character_set}"),
    ("user", "The next section contains the script for extracting additional character names: {script}"),
    ("user", "Instructions:\n\
     1. Combine the names from the previous list and the script into a single list, ensuring no duplicates.\n\
        - If any names in the Previous Character list violate the instructions below, remove them from the final list.\n\
     2. If no new names can be extracted from the script, return the Previous Character list (after applying the instructions) without changes.\n\
     3. Extract only proper nouns that clearly represent key character names.\n\
        - Include only characters who seem significant to the plot or play a recurring role. Exclude minor, incidental, or one-off characters who appear only briefly and contribute minimally to the story.\n\
        - Exclude pronouns (e.g., he, she, they), role-based titles (e.g., Teacher, Doctor), and generic titles or relational terms (e.g., Mom, Dad, Boss).\n\
        - If a name is used alone (e.g., 'Blanc'), determine if it represents a first name or a surname based on the context of its usage in the sentence. Exclude it if it is a surname.\n\
     4. Convert all names to uppercase. For example, 'John' should become 'JOHN'.\n\
     5. Format the final list by separating names with commas, using no additional punctuation or formatting. For example: JOHN, JANE, PAUL, EMILY.\n\
     6. Ensure that the final output contains only proper character names and is free of duplicates. If it is already in the list, return an empty text. Under no circumstances include any text other than names or an empty text.")])

Final_Characterlist_Template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert script analyzer. Your task is to extract and refine a list of key character names based on the provided script and the final plot-derived character set. "
               "If the character names suggest a recognizable work of media, please use your background knowledge to enhance the accuracy of the final list. "
               "Follow these instructions strictly to ensure the final list meets the requirements."),
    ("user", "The following is the final set of characters derived from the plot. Final Character set: {character_set}"),
    ("user", "Instructions: "
             "1. Exclude pronouns (e.g., he, she, they), role-based titles (e.g., Teacher, Doctor), and relational terms (e.g., Mom, Dad, Boss). "
             "2. Exclude surnames and retain only given names. "
             "3. Remove characters who are not significantly important to the story. "
             "4. Convert all names to uppercase. For example, 'John' should become 'JOHN'. "
             "5. Return the final list with names separated by commas, without any additional punctuation or formatting. For example: JOHN, JANE, PAUL, EMILY. "
             "6. If there are multiple names representing the same character, merge them into one. "
             "Ensure the list is comprehensive, concise, and adheres to these instructions.")])

Save_Event_Template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert script analyzer. Extract all events from the script where a character performs an action involving another character. "
        "Each event must include:\n"
        "- `subject`: The character performing the action (from the character list).\n"
        "- `object`: The character receiving the action (from the character list).\n"
        "- `action`: A description of the action. This encompasses not only physical activities but also a wide range of verbal communications, such as what was said, information conveyed, or other forms of linguistic expression.\n"
        "- `scene`: The scene number where the action occurs.\n\n"
        "Return the events as a list of dictionaries in this format:\n"
        "[\n"
        "    {{'subject': 'Character name', 'object': 'Character name', 'action': 'Description of the action', 'scene': Scene number}},\n"
        "    ...\n"
        "]\n\n"
        "Rules:\n"
        "1. Include all valid(has obviously both Subject and Object) actions, not just the first one as many as possible. Don't include SELF in object\n"
        "2. If the same character appears multiple times as `subject` or `object`, include each occurrence as a separate event.\n"
        "3. Ignore ambiguous actions or actions not involving characters from the list."
    )),
    ("user", "Character list: {character_list}"),
    ("user", "Script content:\n{script}\nExtract and return all valid events. Do not include Code block. like ```json```. Only response Json")
])
Summarize_Plot_Template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert script analyzer. Your task is to analyze the script content and extract a single key plot point from the scene."
        " Return the result in the specified JSON format."
        "\n\n"
        "The plot point must include:\n"
        "- `scene`: The scene number where the plot point occurs.\n"
        "- `participants`: The key characters involved in the plot point. Include only those who are in character list : {character_list}. Ensure that character names are exactly as they appear in the character list.\n"
        "- `summary`: A concise summary of the key event or action in the scene.\n"
        "- `setting`: The location or time of the scene, if described in the script.\n\n"
        "Return the result as a single JSON object in this format:\n"
        "{{\n"
        "    'scene': Scene number,\n"
        "    'participants': ['Character1', 'Character2'], If an appropriate value is not available, write ['Not Found']\n"
        "    'summary': 'Brief description of the plot point', If an appropriate value is not available, write 'Not found'.\n"
        "    'setting': 'Location or time description' If an appropriate value is not available, write 'Not found'.\n"
        "}}\n\n"
        "Rules:\n"
        "1. Include only the key characters and significant events; ignore minor details or one-off characters.\n"
        "2. If a scene has no significant events, still include a placeholder with `summary` as 'No significant events'.\n"
        "3. Ensure the format is strictly a single JSON object and avoid adding any additional comments or explanations.\n")),
    ("user", "Script content: {script}"),
    ("user", "Analyze the script and respond strictly in JSON format. Do not include additional explanations or comments. Only use single-quote. If a single quote appears within single quotes, escape it.', 'additional_note': 'This is how you handle single quotes within JSON strictly using single quotes.")])

Extract_Trait_Template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert in psychological profiling and behavioral analysis. Based on the provided events where a character performs actions, "
        "analyze the personality traits of the character and provide a profile including scores for each of the Big Five personality traits.\n\n"
        "The Big Five traits are:\n"
        "- `Extraversion`: Sociability, energy, and assertiveness.\n"
        "- `Agreeableness`: Trust, altruism, kindness, and affection.\n"
        "- `Conscientiousness`: Organization, responsibility, and dependability.\n"
        "- `Neuroticism`: Emotional instability and tendency toward negative emotions.\n"
        "- `Openness`: Creativity, curiosity, and appreciation for new experiences.\n\n"
        "Return the analysis as a JSON object in this format:\n"
        "{{\n"
        "    'traits': {{\n"
        "        'Extraversion': 4.5,\n"
        "        'Agreeableness': 3.8,\n"
        "        'Conscientiousness': 4.0,\n"
        "        'Neuroticism': 2.5,\n"
        "        'Openness': 4.2\n"
        "    }},\n"
        "    'prompt': '<Character system prompt based on the analysis. Provide a detailed description of the character’s personality, motivations, strengths, weaknesses, and notable behaviors, e.g., \"Detective Benoit Blanc is a highly intelligent and observant private investigator with Southern charm, articulate speech, and a methodical approach to solving mysteries. He is intuitive, curious, and driven by a sense of justice, with a unique mix of wit, humility, and determination.\">' \n"
        "}}\n\n"
        "Rules:\n"
        "1. Use the provided actions to determine the character's personality traits.\n"
        "2. Assign a score between 1.0 (lowest) and 5.0 (highest) for each trait based on the character's actions.\n"
        "3. Analyze the character independently based on the provided events.\n"
        "4. Include only valid JSON in your response. Do not use code blocks like ```json```.\n"
        "5. Assume the character is from the movie 'Knives Out.'"
    )),
    ("user", "Character name: {character_name}"),
    ("user", "Events:\n{events}\n If the information is insufficient, just assign neutral scores. Analyze and return the character's personality traits in JSON format.")])

Save_Relationship_Template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert script analyzer. Extract relationship dynamics between characters based on their interactions."),
    ("user", "Script content: {script}"),
    ("user", "Identify and describe relationships between characters based on their interactions or dialogue. \
    Respond in the format: [Character A] and [Character B]: [Relationship Description]. \
    Respond only with this format, no additional commentary.")])

# Chain
Regular_Expression_Chain = (Regular_Expression_Template | gpt_4o | StrOutputParser())
Save_Characterlist_Chain = (Save_Characterlist_Template | gpt_4o | StrOutputParser())
Final_Characterlist_Chain = (Final_Characterlist_Template | gpt_4o | StrOutputParser())
Save_Event_Chain = (Save_Event_Template | gpt_4o | StrOutputParser())
Summarize_Plot_Chain = (Summarize_Plot_Template | gpt_4o | StrOutputParser())
Extract_Trait_Chain = (Extract_Trait_Template | gpt_4o | StrOutputParser())
Save_Relationship_Chain = (Save_Relationship_Template | gpt_4o | StrOutputParser())

# Functions
def Merge_Script(script_list, max_length=3000):
    merged_scripts = []
    buffer = ""
    scene_number_regex = re.compile(r"\bScene\s+\d+:\b")

    for script in script_list:
        if len(buffer) + len(script) < max_length:
            buffer += script + "\n\n"
        else:
            last_scene_match = list(scene_number_regex.finditer(buffer))
            if last_scene_match:
                last_scene_start = last_scene_match[-1].start()
                if len(buffer) - last_scene_start < len(script):
                    merged_scripts.append(buffer[:last_scene_start].strip())
                    buffer = buffer[last_scene_start:] + script
                else:
                    merged_scripts.append(buffer.strip())
                    buffer = script
            else:
                merged_scripts.append(buffer.strip())
                buffer = script
    if buffer:
        merged_scripts.append(buffer.strip())
    return merged_scripts

async def Split_Script(script : str, sample_script : str):
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        try:
            result_regex = await Regular_Expression_Chain.ainvoke({"script": sample_script})
            result_regex = result_regex.strip()
            result_regex = result_regex.replace("\n", "")
            if result_regex.startswith("```") and result_regex.endswith("```"):
                result_regex = result_regex[3:-3].strip()
            if result_regex.startswith("regex"):
                result_regex = result_regex.replace("regex", "", 1).strip()
            compiled_regex = re.compile(result_regex) 
            internal_number_regex = r"\b\d+\.\b"
            script = re.sub(internal_number_regex, "", script)
            break

        except re.error as e:
            print(f"Attempt {attempt + 1}: Invalid regex provided: {e}")
            attempt += 1 

            if attempt == max_attempts:
                raise ValueError("Failed to get a valid regex after 3 attempts.")

        except Exception as ex:
            print(f"Unexpected error during regex extraction: {ex}")
            attempt += 1
            if attempt == max_attempts:
                raise ValueError("Unexpected failure after 3 attempts.")
    
    folder_name = "processed_scripts"
    os.makedirs(folder_name, exist_ok=True)
    split_texts = re.split(compiled_regex, script)
    scene_headers = re.findall(compiled_regex, script)
    if split_texts[0].strip():
        movie_info = split_texts[0].strip()
        file_path = os.path.join(folder_name, "movie_info.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(movie_info)

    scene_with_headers = []
    scene_number = 1 
    for idx, scene_header in enumerate(scene_headers):
        scene_content = split_texts[idx + 1].strip() if idx + 1 < len(split_texts) else ""
        scene_text = f"Scene {scene_number}: {scene_header}\n{scene_content}"
        file_path = os.path.join(folder_name, f"script_scene_{scene_number}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(scene_text)
        scene_with_headers.append(scene_text)
        scene_number += 1
    return scene_with_headers

async def Extract_Characterlist(script_list : list) :
    character_set = set()
    for line in script_list :
        result = await Save_Characterlist_Chain.ainvoke({"character_set" : character_set, "script" : line}) 
        character_set.update(result.split(", "))
        character_list = await Final_Characterlist_Chain.ainvoke({"character_set" : character_set})
    final_list = [item.strip() for item in character_list.split(",")]
    for character in final_list:
            try:
                character_collection.insert_one({"_id": character, "name": character})
            except Exception as e:
                print(f"Document for {character} already exists. Skipping. ({e})", flush = True)
    return final_list

async def Extract_Event(script_list : list, character_list : list) :
    class Event(BaseModel):
        subject: str
        object: str
        action: str
        scene: int
    for scene in script_list :
        result = await Save_Event_Chain.ainvoke({"script" : scene, "character_list" : character_list})
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        if result.startswith("json") or result.startswith("Json"):
            result = result[3:].strip()
        parsed_result = ast.literal_eval(result)
        events = [Event(**item) for item in parsed_result]
        for event in events :
            character_collection.update_one({"_id": event.subject},
                {"$push": {"events": {
                    "subject" : event.subject,
                    "object": event.object,
                    "action": event.action,
                    "scene": event.scene
                }}})
            if event.subject in vector_stores:
                    embedding_text = f"{event.subject} {event.action} to {event.object}"
                    metadata = {
                        "subject": event.subject,
                        "object": event.object,
                        "action": event.action,
                        "scene": event.scene
                    }
                    vector_stores[event.subject].add_texts([embedding_text], [metadata])
                    character_path = os.path.join(FAISS_PATH, f"{event.subject}_faiss")
                    vector_stores[event.subject].save_local(character_path)

async def Summarize_Plot(script_list : list, character_list : list) :
    class Plot(BaseModel):
        scene_number: int
        participants: list[str]
        summary : str
        setting : str
    for i, scene in enumerate(script_list, start=1):
        if i < 128 :
            continue
        result = await Summarize_Plot_Chain.ainvoke({"script": scene, "character_list": character_list})

        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        if result.startswith("json") or result.startswith("Json"):
            result = result[3:].strip()
        try:
            parsed_result = ast.literal_eval(result)

            # Plot 객체 생성
            plot = Plot(
                scene_number=i,
                participants=parsed_result.get("participants", ["Not Found"]),
                summary=parsed_result.get("summary", "Not found"),
                setting=parsed_result.get("setting", "Not found")
            )
            plot_document = {
                "_id": plot.scene_number,
                "scene_number": f"#{plot.scene_number}",
                "participants": plot.participants,
                "summary": plot.summary,
                "setting": plot.setting
            }
            plot_collection.replace_one(
                {"_id": plot_document["_id"]},
                plot_document,
                upsert=True
            )
            print(f"Scene {i} saved to MongoDB: {plot_document}", flush=True)
        except (SyntaxError, ValueError) as e:
            print(f"Error processing result for scene {i}: {e}", flush=True)
            break

async def Extract_Trait(character_list : list) :
    class Trait(BaseModel):
        Extraversion : float
        Agreeableness: float
        Conscientiousness : float
        Neuroticism : float
        Openness : float
        Prompt : str

    def events_to_text(events):
        event_texts = []
        for event in events:
            subject = event.get('subject', 'Unknown Subject')
            action = event.get('action', 'Unknown Action')
            obj = event.get('object', 'Unknown Object')
            scene = event.get('scene', 'Unknown Scene')
            event_text = f"In Scene #{scene}: {subject} {action} to {obj}."
            event_texts.append(event_text)
        return " ".join(event_texts)
    
    for character_id in character_list:
        character_doc = character_collection.find_one({"_id": character_id})
        events = character_doc.get("events", [])
        events_text = events_to_text(events)
        result = await Extract_Trait_Chain.ainvoke({"events": events_text, "character_name" : character_id})
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        if result.startswith("json") or result.startswith("Json"):
            result = result[3:].strip()
        try:
            parsed_result = ast.literal_eval(result)
            traits = parsed_result.get("traits", {})
            prompts = parsed_result.get("prompt", {})
            trait = Trait(
                Extraversion=traits.get("Extraversion", 3.0),
                Agreeableness=traits.get("Agreeableness", 3.0),
                Conscientiousness=traits.get("Conscientiousness", 3.0),
                Neuroticism=traits.get("Neuroticism", 3.0),
                Openness=traits.get("Openness", 3.0),
                Prompt = prompts
            )
            trait_document = {
                "_id": character_id,
                "prompts" : trait.Prompt,
                "traits": {
                    "Extraversion": trait.Extraversion,
                    "Agreeableness": trait.Agreeableness,
                    "Conscientiousness": trait.Conscientiousness,
                    "Neuroticism": trait.Neuroticism,
                    "Openness": trait.Openness
                }
            }
            character_collection.update_one({"_id": character_id}, {"$set": trait_document})
            print(f"Traits for character {character_id} saved to MongoDB: {trait_document}", flush=True)
        except (SyntaxError, ValueError) as e:
            print(f"Error processing traits for character {character_id}: {e}", flush=True)
            continue

async def Extract_Relationship(script_list : list) :
    documents = []
    for line in script_list :
        result = await Save_Relationship_Chain.ainvoke({"script" : line})
        document = Document(page_content=result, metadata={"type" : "Relationship"})
        documents.append(document)
