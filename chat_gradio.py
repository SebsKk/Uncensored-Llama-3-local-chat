import json
import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# her einput path to your model
model_path = "./local_model"

# tokenizer and model with float16 precision
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=250,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2
)


SESSIONS_FILE = "sessions.json"

def load_sessions():
    """Load sessions from the JSON file if it exists."""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    """Save sessions to the JSON file."""
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

sessions = load_sessions()  # Load existing sessions at start
session_names = list(sessions.keys())

def chat(user_message, history):
    """
    Handle user messages and generate assistant responses.
    
    Args:
        user_message (str): The latest message from the user.
        history (list): A list of tuples containing past user and assistant messages.
        
    Returns:
        tuple: Updated history and chatbot state.
    """
    # Build the prompt from the conversation history
    prompt = ""
    for human_msg, bot_msg in history:
        prompt += f"User: {human_msg}\nAssistant: {bot_msg}\n"
    prompt += f"User: {user_message}\nAssistant:"
    
    # Diagnostic print
    print("Prompt being sent to model:")
    print(prompt)
    
    # Generate response with reduced tokens
    response = pipe(
        prompt,
        max_new_tokens=100,  # 
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2
    )[0]["generated_text"]
    
    print("Raw model response:")
    print(response)
    
    if "User:" in response:
        answer = response.split("Assistant:")[-1].split("User:")[0].strip()
    else:
        answer = response.split("Assistant:")[-1].strip()
    
    print("Extracted answer:")
    print(answer)
    
    # Update history
    history.append((user_message, answer))
    return history, history

def load_session(selected_session):
    """Load a selected session from the sessions dict and return it as the current state."""
    if selected_session in sessions:
        return sessions[selected_session]
    return []

def save_current_session(history, session_name):
    """Save current session under the given session_name."""
    if session_name.strip() == "":
        # If no name is provided, do nothing or handle error
        return gr.update(), gr.update(choices=session_names), "Please provide a session name."
    sessions[session_name] = history
    save_sessions(sessions)
    # Update the session names list
    updated_list = list(sessions.keys())
    return gr.update(choices=updated_list, value=session_name), gr.update(choices=updated_list), f"Session '{session_name}' saved."

with gr.Blocks() as demo:
    gr.Markdown("# Local LLM Interface")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Sessions")
            session_dropdown = gr.Dropdown(label="Load a session", choices=session_names, value=None)
            load_button = gr.Button("Load Session")
            gr.Markdown("---")
            session_name_box = gr.Textbox(label="Session Name", placeholder="Enter a name to save this session")
            save_button = gr.Button("Save Current Session")
            status_output = gr.Markdown("")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(show_label=False, placeholder="Type your message here and press Enter...")
            clear = gr.Button("Clear Chat")
            state = gr.State([])

            msg.submit(chat, [msg, state], [chatbot, state])
            clear.click(lambda: [], None, [chatbot, state])

    load_button.click(fn=load_session, inputs=[session_dropdown], outputs=[state]) \
               .then(lambda hist: hist, state, chatbot)

    # Save the current session when the save button is clicked
    save_button.click(fn=save_current_session,
                      inputs=[state, session_name_box],
                      outputs=[session_dropdown, session_dropdown, status_output])

demo.launch()
