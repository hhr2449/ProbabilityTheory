import pymupdf

doc = pymupdf.open("example.pdf")

total_chars = 0

for page in doc:
    text = page.get_text()
    total_chars += len(text)
    print(text)

print(f"文本总字符数: {total_chars}")
