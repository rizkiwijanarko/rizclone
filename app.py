import gradio as gr
from dotenv import load_dotenv

from implementation.chat import answer_question

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
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

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, buttons=["copy"]
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about Insurellm...",
                        show_label=False,
                        scale=7,
                    )
                    submit_btn = gr.Button("Send", scale=1)

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
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
