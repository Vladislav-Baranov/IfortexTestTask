from transformers import BartTokenizer, pipeline
import torch
import gradio as gr
import requests
from bs4 import BeautifulSoup
import re


device = 0 if torch.cuda.is_available() else -1
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
summarizer = pipeline(
    "summarization",
    model=model_name,
    tokenizer=tokenizer,
    device=device
)

#Считаем токены
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, truncation=False))


def sliding_window_summarization(text: str, max_length: int, min_length: int, window_size: int = 1024,
                                 stride: int = 256) -> str:
    if count_tokens(text) <= window_size:
        return generate_summary(text, max_length, min_length)

    chunks = []
    tokens = tokenizer.encode(text, truncation=False)
    start = 0

    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        start += stride

    combined_summary = ""

    for i, chunk in enumerate(chunks):
        chunk_summary = generate_summary(
            chunk,
            max_length=max(50, max_length // len(chunks)),
            min_length=min(30, min_length // len(chunks)))
        combined_summary += " " + chunk_summary.strip()

        if count_tokens(combined_summary) > window_size * 2:
            combined_summary = generate_summary(
                combined_summary,
                max_length=window_size // 2,
                min_length=window_size // 4)

    return generate_summary(combined_summary, max_length, min_length)


def generate_summary(text: str, max_length: int, min_length: int) -> str:
    try:
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']
    except Exception as e:
        return f"Ошибка при генерации: {str(e)}"


def extract_text_from_url(url: str, max_chars: int = 20000) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'iframe', 'button']):
            element.decompose()

        text = ' '.join(p.get_text().strip() for p in soup.find_all('p'))
        text = re.sub(r'\s+', ' ', text)
        return text[:max_chars]
    except Exception as e:
        return f"Ошибка загрузки: {str(e)}"


def read_txt_file(file_path: str, max_chars: int = 100000) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(max_chars)
        return content
    except Exception as e:
        return f"Ошибка чтения файла: {str(e)}"


def process_input(
        text_input: str,
        url_input: str,
        file_input: str,
        max_length: int,
        use_sliding_window: bool
) -> tuple:
    display_text = ""
    summary = ""
    source_text = ""

    # Обработка загруженного файла
    if file_input:
        file_path = file_input.name if hasattr(file_input, 'name') else file_input
        source_text = read_txt_file(file_path)
        if source_text.startswith("Ошибка"):
            return source_text, source_text, ""

    # Обработка URL
    elif url_input:
        source_text = extract_text_from_url(url_input)
        if source_text.startswith("Ошибка"):
            return source_text, source_text, ""

    # Обработка прямого ввода текста
    elif text_input:
        source_text = text_input

    else:
        return "Ошибка: Введите текст, URL или загрузите файл", "", ""

    display_text = source_text[:5000]
    if len(source_text) > 5000:
        display_text += "\n\n... (текст сокращен для отображения)"

    # Генерация суммаризации
    if use_sliding_window or len(source_text) > 5000:
        summary = sliding_window_summarization(source_text, max_length, min_length=50)
    else:
        summary = generate_summary(source_text, max_length, min_length=50)

    return display_text, summary, source_text


# UI
with gr.Blocks(title="Суммаризатор текстов", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📝 Универсальный суммаризатор текстов")
    gr.Markdown("Загрузите файл, введите URL или текст для генерации краткого содержания")

    with gr.Row():
        with gr.Column():
            # Входные данные
            with gr.Tab("Текст"):
                text_input = gr.Textbox(
                    label="Введите текст для суммаризации",
                    lines=8,
                    placeholder="Вставьте сюда текст статьи...",
                    max_lines=20)

            with gr.Tab("URL"):
                url_input = gr.Textbox(
                    label="Введите URL статьи",
                    placeholder="https://example.com/article.html")

            with gr.Tab("Файл"):
                file_input = gr.File(
                    label="Загрузите TXT-файл",
                    file_types=["text", ".txt"])

            # Настройки
            with gr.Accordion("Дополнительные настройки", open=False):
                with gr.Row():
                    max_length = gr.Slider(
                        label="Длина суммаризации",
                        minimum=50,
                        maximum=500,
                        value=150,
                        step=10)

                    use_sliding_window = gr.Checkbox(
                        label="Режим для длинных текстов",
                        value=True)

            generate_btn = gr.Button("Сгенерировать саммари", variant="primary")

        with gr.Column():
            # Результаты
            gr.Markdown("### Результаты")
            original_display = gr.Textbox(
                label="Исходный текст (первые 5000 символов)",
                lines=10,
                interactive=False)

            summary_output = gr.Textbox(
                label="Суммаризация",
                lines=10,
                interactive=False)

            full_content = gr.Textbox(
                label="Полный текст (для отладки)",
                visible=False)

    # Обработчики событий
    generate_btn.click(
        fn=process_input,
        inputs=[text_input, url_input, file_input, max_length, use_sliding_window],
        outputs=[original_display, summary_output, full_content])

    # Очистка полей при переключении источников
    url_input.change(
        fn=lambda x: ("", x, None) if x else (gr.update(), gr.update(), gr.update()),
        inputs=url_input,
        outputs=[text_input, url_input, file_input])

    text_input.change(
        fn=lambda x: (x, "", None) if x else (gr.update(), gr.update(), gr.update()),
        inputs=text_input,
        outputs=[text_input, url_input, file_input])

    file_input.change(
        fn=lambda x: ("", "", x) if x else (gr.update(), gr.update(), gr.update()),
        inputs=file_input,
        outputs=[text_input, url_input, file_input])

# Запуск приложения
if __name__ == "__main__":
    try:
        # Проверка работы модели
        test_text = "This is a test text for model initialization."
        test_summary = generate_summary(test_text, 50, 30)
        print(f"Тестовая суммаризация: {test_summary}")

        demo.launch(
            server_name="localhost",
            show_error=True,
            share=False
        )
    except Exception as e:
        print(f"Ошибка запуска: {str(e)}")
        print("Проверьте подключение к интернету и установленные зависимости")