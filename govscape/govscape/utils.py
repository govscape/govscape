def read_txt_file(txt_path):
    text = ""
    with open(txt_path) as file:
        text = file.read()
    return text
