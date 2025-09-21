input_file = "data/old_train_caption.txt"  # file gốc 2 cột: tên file \t công thức
output_file = "filtered_files.txt"  # file xuất chỉ gồm tên file

# Danh sách ký hiệu toán học quan trọng
math_symbols = [
    "geq",
    "sqrt",
    "leq",
    "infty",
    "cdot",
    "sigma",
    "pm",
    "log",
    "pi",
    "limits",
    "tan",
    "gamma",
    "theta",
    "forall",
    "int",
    "sin",
    "prime",
    "ldots",
    "cdots",
    "cos",
    "Delta",
    "neq",
    "in",
    "alpha",
    "times",
    "lim",
    "lambda",
    "exists",
    "frac",
    "rightarrow",
    "div",
    "phi",
    "beta",
    "mu",
    "sum"
]

filtered_files = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")  # tách 2 cột bằng tab
        if len(parts) < 2:
            continue
        file_name, formula = parts[0], parts[1]

        # Bỏ những công thức quá ngắn (1 ký tự) và không có ký hiệu toán học
        if len(formula.replace(" ", "")) <= 1:
            continue

        # Giữ những công thức chứa ít nhất 1 ký hiệu toán học
        if any(symbol in formula for symbol in math_symbols):
            filtered_files.append(file_name)

# Ghi ra file chỉ gồm tên file
with open(output_file, "w", encoding="utf-8") as f:
    for file_name in filtered_files:
        f.write(file_name + "\n")

print(f"Đã lọc xong {len(filtered_files)} công thức phức tạp, lưu vào {output_file}")
