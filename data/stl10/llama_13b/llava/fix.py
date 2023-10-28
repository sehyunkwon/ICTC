lines = []
with open("gpt_labels.txt", "r") as file:
    file = file.readlines()
    for f in file:
        if "\\n\\n" in f:
            f = f.split("png;")[0] + "png; Answer: " + f.split('\\n\\n')[1].strip() + "\n"
        lines.append(f)

lines = "".join(lines)
with open("gpt_labels.txt", "w") as file:
    file.write(lines)