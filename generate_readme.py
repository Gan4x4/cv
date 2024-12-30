import os
import nbformat

course_order = [
    "Classic_ML",
    "Decision_trees",
    "Neural_network",
    "Convolutional_neural_network",
    "Transformers",
    "Representation_learning",
    "Segmentation",
    "Detection",
    "Generative_models"
]

def extract_headers(notebook_path):
    """
    Извлекает заголовки h1 и h2 из блокнота Jupyter.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    headers = []
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            lines = cell.source.split('\n')
            for line in lines:
                if line.startswith('# '):  # Заголовок h1
                    headers.append((1, line[2:].strip()))
                elif line.startswith('## '):  # Заголовок h2
                    headers.append((2, line[3:].strip()))
    return headers


def generate_course_program():
    """
    Генерирует программу курса из заданных папок и блокнотов.
    """
    program = "# Программа курса\n\n"
    current_dir = os.getcwd()  # Текущая директория

    for block in course_order:
        block_path = os.path.join(current_dir, block)
        if os.path.isdir(block_path):  # Проверяем, существует ли папка
            program += f"## {block}\n\n"  # Название блока (папки)
            for notebook in sorted(os.listdir(block_path)):
                if notebook.endswith(".ipynb"):
                    notebook_path = os.path.join(block_path, notebook)
                    program += f"### {notebook}\n\n"  # Название блокнота
                    headers = extract_headers(notebook_path)
                    for level, header in headers:
                        if level == 1:
                            program += f"- **{header}**\n"  # Заголовок h1
                        elif level == 2:
                            program += f"  - {header}\n"  # Заголовок h2
                    program += "\n"
    return program


def save_to_readme(output_file="README.md"):
    """
    Сохраняет программу курса в README.md.
    """
    program = generate_course_program()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(program)
    print(f"README.md успешно создан в {os.getcwd()}!")


# Запуск скрипта
if __name__ == "__main__":
    save_to_readme()
