import os
import json

def post_process(content):
    content = content.replace("\\n", ' ')
    for mark in ['“', '”', '"','"', '\\', "Script:", ":"]:
        content = content.replace(mark, '')

    for tag in ["(Setup)", "(Punchline)", "(Incongruity)", "(Callback)",
    "(setup)", "(punchline)", "(incongruity)", "(callback)"]:
        content = content.replace(tag, '')

    # put new lines after each sentence

    content = content.strip()
    return content

def baseline_post_process(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
        content = post_process(content)
        
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def got_post_process(input_file, output_file):
    # open as json and aggregate the "script/text" field

    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        aggregated_script = ""

        script_list = data.get("script", [])

        for item in script_list:
            script_text = item.get("text", "")
            if script_text:
                aggregated_script += script_text + "\n"

        content = post_process(aggregated_script)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

if __name__ == "__main__":
    # read files
    input_folder = "."
    output_folder = "./processed"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        # change .json to .txt
        output_filename = filename[:-5] + ".txt"

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, output_filename)
        if filename.startswith("got"):
            got_post_process(input_path, output_path)
        elif filename.startswith("baseline"):
            baseline_post_process(input_path, output_path)