import re

languge_settings = {
    'python': {
        'full_name': 'Python',
        'indent': 4,
    },
    'cpp': {
        'full_name': 'cpp',
        'indent': 0,
        'main': "int main()",
    },
    'java': {
        'full_name': 'Java',
        'indent': 4,
        'main': "public static void main",
    },
    'cs': {
        'full_name': "csharp",
        'indent': 0,
        'main': "public static void Main",
    },
    'php': {
        'full_name': "PHP",
        'indent': 0,
    },
    'ts': {
        'full_name': "TypeScript",
        'indent': 0,
    },
    'js': {
        'full_name': "JavaScript",
        'indent': 0
    },
    'sh': {
        'full_name': "Bash",
        'indent': 0
    }
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        matching_lines = [i for i in range(len(func_lines)) if func_lines[i].lstrip().startswith("def ")]
        if matching_lines:
            func_idx = matching_lines[-1]
            func_name = func_lines[func_idx].strip().split('(')[0].replace("def ", "").strip()
            func_prefix = "\n".join(func_lines[:func_idx])
            return func_name, func_prefix
        return "", ""
    
    if func_lines:
        func_name = func_lines[-1].split('{')[0].strip()
        func_prefix = "\n".join(func_lines[:-1])
        return func_name, func_prefix
    return "", ""

def extract_generation_code(example: str, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    if not output:
        output = ""
    question = example["prompt"].strip()
    setting = languge_settings[lang_code]
    lang = setting['full_name']
    indent = setting['indent']

    try:
        # Match code block with language identifier or generic code block, supporting CRLF and unclosed blocks
        pattern = rf'```{re.escape(lang.lower())}[^\S\r\n]*\r?\n(.*?)(?:```|$)'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if not matches:
            pattern_generic = r'```[^\S\r\n]*\r?\n(.*?)(?:```|$)'
            matches = re.findall(pattern_generic, output, re.DOTALL)
        if matches:
            code_block = matches[0]
        else:
            code_block = output
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))
        
        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
        
        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower()) if func_name else 0
            indent = 0
            while start - indent > 0 and code_block[start - indent - 1] == ' ':
                indent += 1
            
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*indent + '}')
            except:
                end = len(code_block)

        body = code_block[start:end]

        if lang_code.lower() in ['php', 'ts', 'js']:
            body += '\n' + ' '*indent + '}'
    
        generation = func_prefix + '\n' + body + '\n' if func_prefix else body + '\n'
        example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output
    
    return example

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]