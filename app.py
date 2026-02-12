import gradio as gr
from dotenv import load_dotenv

from implementation.chat import answer_question

load_dotenv(override=True)


def format_context(context):
    if not context:
        return "No relevant context found."
    
    result = "### 📚 Relevant Context\n\n"
    for doc in context:
        source = doc.metadata.get('source', 'Unknown Source')
        result += f"**Source: {source}**\n\n"
        result += f"{doc.page_content}\n\n"
        result += "---\n\n"
    return result


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts)
    return str(content)


def chat(history):
    try:
        raw_last_message = history[-1]["content"]
        last_message = extract_text(raw_last_message)

        # Convert entire history to string content for the backend
        clean_history = []
        for msg in history[:-1]:
            clean_history.append({
                "role": msg["role"],
                "content": extract_text(msg["content"])
            })

        answer, context = answer_question(last_message, clean_history)
        history.append({"role": "assistant", "content": answer})
        return history, format_context(context)
    except Exception as e:
        import traceback
        traceback.print_exc()
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, "Error occurred."


def main():
    def put_message_in_chatbot(message, history):
        new_history = history + [{"role": "user", "content": message}]
        return "", new_history

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(title="Kharisma Rizki Wijanarko - AI Assistant") as ui:
        gr.Markdown(
            """
            # 👨‍💻 Kharisma Rizki Wijanarko - AI Assistant
            Welcome! I am an AI assistant trained to answer questions about Rizki's career, background, skills, and experience.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=550,
                    show_label=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/avataaars/svg?seed=Rizki"),
                )
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Ask me about Rizki's projects, skills, or experience...",
                        show_label=False,
                        scale=7,
                        container=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "What is Rizki's professional background?",
                        "Tell me about Rizki's technical skills.",
                        "What kind of projects has Rizki worked on?",
                        "How can I contact Rizki?",
                    ],
                    inputs=message,
                    label="Try asking:"
                )

            with gr.Column(scale=1):
                with gr.Accordion("🔍 Behind the scenes: Retrieved Context", open=False):
                    context_markdown = gr.Markdown(
                        value="*Retrieved context will appear here when you ask a question*",
                    )

        submit_btn.click(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown], queue=True)

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown], queue=True)

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
