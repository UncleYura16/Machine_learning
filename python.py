SHOW_PLOTS = False
PRINT_LIMIT = 40

import math
import os
import random
import re
from collections import Counter

import numpy as np
import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "chym.txt")
OUT_PATH = os.path.join(BASE_DIR, "chym_out.txt")
IMAGE_PATH = os.path.join(BASE_DIR, "image.jpg")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset(path=DATA_PATH):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return text, lines


TEXT, LINES = load_dataset()


def load_csv_once(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"[!] Не найден файл {path} — задания Pandas будут недоступны.")
        return None
    return pd.read_csv(path)


TITANIC = load_csv_once("train.csv")
TITANIC_TEST = load_csv_once("test.csv")
TITANIC_GENDER = load_csv_once("gender_submission.csv")
VGSALES = load_csv_once("vgsales.csv")
PANDAS_READY = all(x is not None for x in
                   (TITANIC, TITANIC_TEST, TITANIC_GENDER, VGSALES))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
np.set_printoptions(precision=2, suppress=True)
RNG = np.random.default_rng()


def show(title, arr):
    if isinstance(arr, np.ndarray) and arr.ndim > 1:
        print(title)
        print(arr)
    else:
        print(title, arr)


def read_float(prompt, default):
    raw = input(prompt).strip().replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("  Это не число — берём значение по умолчанию.")
        return default


def out_path(filename):
    return os.path.join(OUT_DIR, filename)


def saved(filename):
    print(f"Сохранено: output/{filename}")


def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(out_path(filename), dpi=120)
    saved(filename)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def print_table(obj, limit=None):
    limit = PRINT_LIMIT if limit is None else limit
    print(obj.head(limit).to_string())
    if len(obj) > limit:
        print(f"... показано {limit} из {len(obj)} строк (PRINT_LIMIT в начале файла)")


def draw_heatmap(ax, table, title, cmap="YlOrRd", annotate=False, fmt="{:.1f}",
                 cbar_label=""):
    im = ax.imshow(table.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=90)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_title(title)
    if annotate:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, fmt.format(table.values[i, j]), ha="center",
                        va="center", fontsize=6)
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
    return im


def read_int(prompt, min_value=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("  Нужно ввести целое число.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Число должно быть не меньше {min_value}.")
            continue
        return value


def task_1():
    limit = read_int("Предел: ", 0)
    a, b, total = 0, 1, 0
    while a <= limit:
        total += a
        a, b = b, a + b
    print(f"Сумма чисел Фибоначчи до {limit}: {total}")


def primes_up_to(n):
    return [i for i in range(2, n + 1) if is_prime(i)]


def task_2():
    n = read_int("N: ", 2)
    print(primes_up_to(n))


def task_3():
    print("Содержимое chym.txt:")
    print(TEXT)
    transformed = TEXT.upper()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(transformed)
    print(f"\nПреобразованный текст сохранён в {os.path.basename(OUT_PATH)}")


def task_4():
    raw = input("Число: ").strip()
    try:
        value = int(raw)
    except ValueError:
        print("Это не целое число.")
        return
    if value > 0:
        print(f"Квадратный корень из {value} = {math.sqrt(value):.4f}")
    else:
        print("Число не положительное.")


def count_lines_with_word(lines, word):
    return sum(1 for ln in lines if word.lower() in ln.lower())


def task_5():
    word = input("Искомое слово: ").strip()
    print(f"Строк со словом «{word}»: {count_lines_with_word(LINES, word)}")


def capitalize_words(s):
    return " ".join(w.capitalize() for w in s.split())


def task_6():
    for ln in LINES:
        print(capitalize_words(ln))


def word_frequency(text):
    words = re.findall(r"\w+", text.lower())
    return dict(Counter(words))


def task_7():
    freq = word_frequency(TEXT)
    for word, cnt in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{word}: {cnt}")


def unique_sorted(nums):
    return sorted(set(nums))


def task_8():
    nums = [5, 3, 8, 3, 1, 5, 9, 1, 7, 8]
    print("Исходный:", nums)
    print("Результат:", unique_sorted(nums))


def min_max(nums):
    return max(nums), min(nums)


def task_9():
    nums = [42, -7, 15, 0, 99, 3, -20]
    mx, mn = min_max(nums)
    print("Список:", nums)
    print(f"Максимум: {mx}, минимум: {mn}")


def count_char(s, ch):
    return s.count(ch)


def task_10():
    ch = input("Искомый символ: ")[:1]
    if not ch:
        print("Символ не введён.")
        return
    for i, ln in enumerate(LINES, 1):
        print(f"Строка {i} - {count_char(ln, ch)} вхождений")


def remove_extra_spaces(s):
    return " ".join(s.split())


def task_11():
    s = "   Это    строка  с   лишними     пробелами   "
    print(f"До:    '{s}'")
    print(f"После: '{remove_extra_spaces(s)}'")


def split_words(s):
    return s.split()


def task_12():
    s = "Python — отличный язык программирования"
    print(split_words(s))


def reverse_string(s):
    return s[::-1]


def is_palindrome(s):
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]


def task_13():
    s = input("Строка: ")
    print("Перевёрнутая:", reverse_string(s))
    print("Палиндром:", "да" if is_palindrome(s) else "нет")


def task_14():
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6, 7}
    print("A =", a, "| B =", b)
    print("Объединение:", a | b)
    print("Пересечение:", a & b)
    print("Разность A - B:", a - b)
    print("Разность B - A:", b - a)


def divisors(n):
    n = abs(n)
    result = set()
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            result.add(d)
            result.add(n // d)
    return sorted(result)


def task_15():
    n = read_int("Целое число (не 0): ")
    if n == 0:
        print("У нуля бесконечно много делителей.")
        return
    print(divisors(n))


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def task_16():
    print([n for n in range(2, 101) if is_prime(n)])


def task_17():
    print("    " + "".join(f"{j:4}" for j in range(1, 11)))
    print("    " + "-" * 40)
    for i in range(1, 11):
        print(f"{i:2} |" + "".join(f"{i * j:4}" for j in range(1, 11)))


def task_18():
    a = read_int("Делимое: ")
    while True:
        b = read_int("Делитель: ")
        try:
            print(f"{a} / {b} = {a / b}")
            break
        except ZeroDivisionError:
            print("  Деление на ноль! Введите другое число.")


def task_19():
    low, high = 1, 100
    secret = random.randint(low, high)
    print(f"Я загадал число от {low} до {high}. Попробуйте угадать!")
    attempts = 0
    while True:
        guess = read_int("Ваш вариант: ")
        attempts += 1
        if guess < secret:
            print("  Загаданное число БОЛЬШЕ.")
        elif guess > secret:
            print("  Загаданное число МЕНЬШЕ.")
        else:
            print(f"Верно! Попыток: {attempts}")
            break


def ensure_image():
    if os.path.exists(IMAGE_PATH):
        return
    img = Image.new("RGB", (400, 300))
    px = img.load()
    for x in range(400):
        for y in range(300):
            px[x, y] = (x * 255 // 399, y * 255 // 299, 128)
    img.save(IMAGE_PATH)


def task_20():
    if Image is None:
        print("Установите Pillow: pip install pillow")
        return
    ensure_image()
    with Image.open(IMAGE_PATH) as img:
        new_size = (img.width // 2, img.height // 2)
        out = img.resize(new_size)
        out.save(os.path.join(BASE_DIR, "image_half.jpg"))
    print(f"Было {img.size}, стало {new_size}. Сохранено: image_half.jpg")


def task_21():
    if Image is None:
        print("Установите Pillow: pip install pillow")
        return
    ensure_image()
    mode = input("Как перевернуть — 'v' (вертикально) или 'h' (горизонтально): ").strip().lower()
    with Image.open(IMAGE_PATH) as img:
        if mode == "v":
            out = ImageOps.flip(img)
            name = "image_flip_v.jpg"
        else:
            out = ImageOps.mirror(img)
            name = "image_flip_h.jpg"
        out.save(os.path.join(BASE_DIR, name))
    print(f"Сохранено: {name}")


def task_22():
    n = read_int("N: ", 1)
    print("Сумма чётных:", sum(i for i in range(2, n + 1, 2)))


def task_23():
    for i in range(1, 11):
        print(f"2^{i} = {2 ** i}")


def task_24():
    m = read_int("M: ")
    n = read_int("N: ")
    if m > n:
        m, n = n, m
    result = []
    for i in range(max(m, 2), n + 1):
        if is_prime(i):
            result.append(i)
    print(result)


def task_25():
    n = read_int("N: ", 1)
    a, b = 0, 1
    seq = []
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    print(seq)


def task_26():
    for code in range(ord("A"), ord("Z") + 1):
        print(f"{code}: {chr(code)}")


def char_frequency(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def task_27():
    s = input("Строка: ")
    print(char_frequency(s))


def merge_sum(d1, d2):
    result = dict(d1)
    for k, v in d2.items():
        result[k] = result.get(k, 0) + v
    return result


def task_28():
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"b": 10, "c": 20, "d": 30}
    print("d1 =", d1)
    print("d2 =", d2)
    print("Результат:", merge_sum(d1, d2))


def longest_key(d):
    return max(d, key=lambda k: len(str(k)))


def task_29():
    d = {"яблоко": 1, "груша": 2, "апельсин": 3, "киви": 4}
    k = longest_key(d)
    print(f"Ключ максимальной длины: '{k}' ({len(k)} символов)")


def count_unique_values(d):
    return len(set(d.values()))


def task_30():
    d = {"a": 1, "b": 2, "c": 1, "d": 3, "e": 2}
    print("Словарь:", d)
    print("Уникальных значений:", count_unique_values(d))


def remove_by_value(d, value):
    return {k: v for k, v in d.items() if v != value}


def task_31():
    d = {"a": 1, "b": 2, "c": 1, "d": 3, "e": 2}
    value = read_int("Удалить пары со значением: ")
    print("Было:  ", d)
    print("Стало: ", remove_by_value(d, value))


def task_32():
    lst = [1, 2, 2, 3, 3, 3, 4, 5, 5]
    print("Список:   ", lst)
    print("Множество:", set(lst))


def task_33():
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6, 7}
    print("A =", a, "| B =", b)
    print("Объединение:", a | b)
    print("Пересечение:", a & b)
    print("Симметричная разность:", a ^ b)


def task_34():
    a = {1, 2, 3}
    b = {1, 2, 3, 4, 5}
    print("A =", a, "| B =", b)
    print("A ⊆ B:", a.issubset(b))
    print("B ⊆ A:", b.issubset(a))


def task_35():
    a = {"a", "b", "c", "d"}
    b = {"c", "d", "e", "f"}
    print("A =", a, "| B =", b)
    print("A ^ B =", a.symmetric_difference(b))


def task_36():
    s = {17, 3, 42, -5, 8, 23}
    print("Множество:", s)
    print("Минимум:", min(s))
    print("Максимум:", max(s))


def numpy_1():
    a = np.arange(9)
    show("Одномерный массив:", a)
    m = a.reshape(3, 3)
    show("Двумерный 3x3:", m)
    total = m.sum()
    print("Сумма всех элементов:", total)
    print("Сумма * 2 =", total * 2)


def numpy_2():
    a = np.zeros((4, 4), dtype=int)
    np.fill_diagonal(a, 1)
    show("С единичной диагональю:", a)
    b = np.delete(a, 0, axis=0)
    b = np.delete(b, -1, axis=1)
    show("После удаления первой строки и последнего столбца:", b)
    print("Сумма оставшихся элементов:", b.sum())


def numpy_3():
    a = np.sort(RNG.integers(0, 100, size=(10, 10)), axis=1)
    show("Отсортированный по строкам массив:", a)
    before = a.copy()
    rows = np.arange(a.shape[0])
    maxs = a.max(axis=1)
    a[rows, a.argmin(axis=1)] = maxs
    show("После замены минимума в каждой строке на максимум строки:", a)
    print("Количество изменений:", np.count_nonzero(np.not_equal(a, before)))


def numpy_4():
    a = RNG.integers(-10, 50, size=10)
    show("Исходный:", a)
    a[a.argmin()] = a.max()
    show("Минимум заменён на максимум:", a)
    a = a * 2
    show("Умножен на 2:", a)
    show("По убыванию:", np.sort(a)[::-1])


def numpy_5():
    a = RNG.integers(0, 10, size=(3, 3))
    b = RNG.integers(0, 10, size=(3, 3))
    show("A:", a)
    show("B:", b)
    h = np.hstack((a, b))
    show("Горизонтальное объединение (3x6):", h)
    c = RNG.integers(0, 10, size=(2, 6))
    show("Третий массив (2x6):", c)
    v = np.vstack((h, c))
    show("Вертикальное объединение (5x6):", v)
    s = np.sort(v, axis=1)
    show("Отсортирован по строкам:", s)
    show("Среднее по столбцам:", s.mean(axis=0))


def numpy_6():
    a = RNG.integers(1, 20, size=(4, 3)).astype(float)
    show("Исходный:", a)
    m = a.mean()
    print("Среднее значение:", round(m, 2))
    a = a + m
    show("После прибавления среднего:", a)
    row_means = a.mean(axis=1)
    show("Средние по строкам:", row_means)
    order = np.argsort(row_means)
    show("Строки, отсортированные по средним:", a[order])


def numpy_7():
    a = RNG.integers(-10, 10, size=(3, 4))
    show("Исходный:", a)
    idx = np.argwhere(np.less(a, 0))
    print("Индексы отрицательных (строка, столбец):", [tuple(i) for i in idx.tolist()])
    b = np.where(np.less(a, 0), -a, a)
    show("После замены на модули:", b)
    pos_sum = b[b > 0].sum()
    orig_sum = a.sum()
    print("Сумма положительных элементов нового массива:", pos_sum)
    print("Сумма элементов исходного массива:", orig_sum)
    print("Новая сумма больше" if pos_sum > orig_sum else
          "Суммы равны" if pos_sum == orig_sum else "Новая сумма меньше")


def numpy_8():
    a = np.arange(1, 13)
    shape = (3, 4)
    show("Исходный одномерный:", a)
    m = a.reshape(shape)
    show(f"Форма {shape}:", m)
    v = np.flipud(m)
    show("Перевёрнут по вертикали:", v)
    h = np.fliplr(v)
    show("Затем по горизонтали:", h)
    back = np.flipud(np.fliplr(h)).reshape(a.shape)
    show("Возврат к исходному виду:", back)
    print("Совпадает с исходным:", np.array_equal(a, back))


def numpy_9():
    a = np.arange(1, 21)
    show("Исходный:", a)
    third = a[2::3]
    show("Каждый третий элемент:", third)
    print("Массивы равны:", np.array_equal(a, third),
          f"(размеры {a.size} и {third.size})")
    show("Разница (элементы исходного, которых нет в новом):",
         np.setdiff1d(a, third))


def numpy_10():
    def process(a):
        a = a.copy()
        show("Исходный:", a)
        if np.any(np.greater(a, 0)):
            print("Есть положительные — удваиваем их.")
            a[a > 0] *= 2
        else:
            print("Положительных нет — прибавляем 1 ко всем.")
            a += 1
        show("Результат:", a)
        print("Нулевых элементов:", np.count_nonzero(np.equal(a, 0)))

    process(np.array([-3, 0, 4, 0, -1, 2]))
    print()
    process(np.array([-3, 0, -5, 0, -1]))


def numpy_11():
    a = np.sort(RNG.integers(0, 100, size=12))
    show("Отсортированный:", a)
    parts = np.array_split(a, 4)
    for i, p in enumerate(parts, 1):
        print(f"Часть {i}: {p}, min={p.min()}, max={p.max()}")
        p[p.argmin()] = p.max()
    show("После замены минимумов на максимумы частей:", a)


def numpy_12():
    k = read_float("Множитель (Enter = 2.5): ", 2.5)
    a = np.round(RNG.uniform(-5, 5, size=10), 2)
    show("Исходный:", a)
    b = a * k
    show(f"Умножен на {k}:", b)
    r = np.rint(b)
    show("Округлён до ближайшего целого:", r)
    show("Уникальные по возрастанию:", np.unique(r))


def numpy_13():
    a = RNG.integers(0, 10, size=(4, 5))
    show("Исходный (4x5):", a)
    t = a.T
    show("Транспонированный (5x4):", t)
    d = np.delete(np.delete(t, 1, axis=0), 2, axis=1)
    show("Без 2-й строки и 3-го столбца (4x3):", d)
    back = d.T
    show("Снова транспонирован (3x4):", back)
    print(f"Формы: исходный {a.shape}, результат {back.shape}")
    print("Равен исходному:", np.array_equal(a, back))


def numpy_14():
    a = RNG.integers(0, 8, size=15)
    show("Исходный:", a)
    vals, counts = np.unique(a, return_counts=True)
    once = vals[counts == 1]
    print("Неповторяющиеся элементы:", once)
    print("Их количество:", once.size)
    s = np.sort(a)
    show("Отсортированный:", s)
    if once.size == 0:
        print("Неповторяющихся элементов нет — заменять нечего.")
        return
    mean = once.mean()
    print("Среднее неповторяющихся:", round(mean, 2))
    res = s.astype(float)
    res[np.isin(s, once)] = mean
    show("После замены неповторяющихся на среднее:", res)


def numpy_15():
    a = RNG.integers(0, 50, size=11)
    show("Исходный:", a)
    m = a.mean()
    print("Среднее:", round(m, 2))
    low, high = a[a < m], a[a >= m]
    show("Меньше среднего:", low)
    show("Не меньше среднего:", high)
    merged = np.concatenate((low, high))
    show("Объединённый:", merged)
    s = np.sort(merged)
    show("Отсортированный:", s)
    print("Медиана:", np.median(s))


def numpy_16():
    a = RNG.integers(0, 6, size=12)
    show("Исходный:", a)
    u = np.unique(a)
    col = u.reshape(-1, 1)
    show("Вектор-столбец без повторов:", col)
    print("Длина вектора-столбца:", len(col))
    print("Длина исходного массива:", len(a))
    print("Разница в длине:", len(a) - len(col))


def numpy_17():
    a = RNG.integers(0, 100, size=(4, 5))
    show("Исходный:", a)
    old_max = a.max(axis=1)
    b = a.copy()
    b[np.arange(b.shape[0]), b.argmax(axis=1)] = b.min(axis=1)
    show("Максимумы строк заменены на минимумы строк:", b)
    new_max = b.max(axis=1)
    show("Старые максимумы:", old_max)
    show("Новые максимумы:", new_max)
    show("Разность (новые - старые):", new_max - old_max)


def numpy_18():
    a = RNG.integers(1, 6, size=(4, 4))
    show("Исходный:", a)
    prod = np.prod(np.diag(a))
    print("Произведение элементов главной диагонали:", prod)
    b = a + prod
    show("После прибавления произведения:", b)
    show("Отсортирован по строкам:", np.sort(b, axis=1))


def numpy_19():
    n = read_int("Размер матрицы n: ", 1)
    m = np.tril(np.arange(1, n * n + 1).reshape(n, n))
    show("Нижняя треугольная матрица:", m)
    r = np.rot90(m, 2)
    show("Повёрнутая на 180°:", r)
    s = m + r
    show("Сумма матриц:", s)
    row_sums = s.sum(axis=1)
    show("Суммы по строкам:", row_sums)
    print("Максимальная сумма по строкам:", row_sums.max())


def numpy_20():
    a = RNG.integers(0, 5, size=(3, 4))
    b = RNG.integers(0, 5, size=(3, 4))
    show("A:", a)
    show("B:", b)
    mask = np.not_equal(a, b)
    print("Индексы различающихся элементов (строка, столбец):",
          [tuple(i) for i in np.argwhere(mask).tolist()])
    new = a.astype(float)
    new[mask] = (a[mask] + b[mask]) / 2
    show("A после замены различающихся на среднее арифметическое:", new)
    show("Разница (исходный A - новый):", a - new)


def numpy_21():
    a = np.arange(1, 11)
    show("Исходный:", a)
    f = np.vectorize(lambda x: x * x + 1)
    b = f(a)
    show("После применения f(x) = x^2 + 1:", b)
    even = b[b % 2 == 0]
    odd = b[b % 2 != 0]
    show("Чётные элементы:", even)
    print("Сумма чётных:", even.sum(), "| сумма нечётных:", odd.sum())
    print("Чётные больше" if even.sum() > odd.sum() else
          "Нечётные больше" if even.sum() < odd.sum() else "Суммы равны")


def numpy_22():
    a = RNG.integers(1, 10, size=(4, 5))
    show("Исходный (4x5):", a)
    r = np.rot90(a)
    show("Повёрнут на 90° против часовой (5x4):", r)
    mask = np.ones(r.shape, dtype=bool)
    mask[1:-1, 1:-1] = False
    show("Элементы на границе:", r[mask])
    r = r.copy()
    r[mask] = 0
    show("Границы заменены на нули:", r)
    show("Повёрнут обратно (по часовой):", np.rot90(r, -1))


def numpy_23():
    def check(m, name):
        show(f"{name}:", m)
        before = np.array_equal(m, m.T)
        t = m.T
        after = np.array_equal(t, t.T)
        print("Симметричен до транспонирования:", before)
        print("Симметричен после транспонирования:", after)
        print("Результаты проверок совпадают:", before == after)

    a = RNG.integers(0, 5, size=(4, 4))
    check(a, "Случайная матрица")
    print()
    check(a + a.T, "Симметричная матрица (A + A^T)")


def numpy_24():
    a = RNG.integers(0, 10, size=(3, 4))
    show("Исходный:", a)
    sums = a.sum(axis=0)
    show("Суммы столбцов:", sums)
    new = np.tile(sums, (a.shape[0], 1))
    show("Новый массив (каждый элемент = сумма своего столбца):", new)
    print("Массивы равны:", np.array_equal(a, new))
    show("Поэлементное сравнение (исходный == новый):", np.equal(a, new))


def numpy_25():
    a = RNG.integers(-10, 20, size=10)
    show("Исходный:", a)
    diff = a.max() - a.min()
    print(f"Максимум {a.max()}, минимум {a.min()}, разница {diff}")
    b = a * diff
    show("Умножен на разницу:", b)
    show("По возрастанию:", np.sort(b))


def numpy_26():
    def is_int_array(arr):
        return np.issubdtype(arr.dtype, np.integer)

    f = np.array([1.5, 2.0, 3.0])
    print(f"Массив {f}: только целые? {is_int_array(f)}")
    a = RNG.integers(-5, 6, size=8)
    print(f"Массив {a}: только целые? {is_int_array(a)}")
    if not is_int_array(a):
        return
    cubed = a ** 3
    show("Кубы элементов:", cubed)
    print("Сумма кубов:", cubed.sum())
    print("Сумма исходных:", a.sum())
    print("Суммы равны:", cubed.sum() == a.sum())


def numpy_27():
    a = np.array([4, 9, 2, 16, 7, 25, 10])
    show("Исходный:", a)
    r = np.round(np.sqrt(a), 2)
    show("Корни, округлённые до 2 знаков:", r)
    r = np.sort(r)[::-1]
    show("По убыванию:", r)
    print("Равен исходному:", np.array_equal(r, a))
    print("Почти равны (allclose):", np.allclose(r, a))


def numpy_28():
    a = np.array([-3.0, 0.0, -1.0, 16.0, 5.0, -2.0, 9.0])
    show("Исходный:", a)
    if not np.any(np.greater(a, 0)):
        print("Положительных элементов нет.")
        return
    i = int(np.argmax(np.greater(a, 0)))
    print(f"Индекс первого положительного: {i} (значение {a[i]})")
    b = a.copy()
    b[i] = np.sqrt(b[i])
    show("После замены на квадратный корень:", b)
    b = np.sort(b)
    show("Отсортирован:", b)
    print("Равен исходному:", np.array_equal(a, b))
    print("Номера позиций, где значения различаются:", np.nonzero(np.not_equal(a, b))[0])


def numpy_29():
    s = np.array(["10", "2", "33", "4", "15", "8", "21"])
    show("Исходный массив строк:", s)
    print("Сортировка строк (лексикографически):", np.sort(s))
    nums = s.astype(float)
    show("Массив чисел:", nums)
    sorted_nums = np.sort(nums)
    show("Отсортированный:", sorted_nums)
    med_new = np.median(sorted_nums)
    med_orig = np.median(nums)
    print("Медиана отсортированного:", med_new)
    print("Медиана исходного (до сортировки):", med_orig)
    print("Медианы совпадают:", med_new == med_orig)


def pandas_1():
    df = TITANIC
    print("[1b] Первые пять строк:")
    print(df.head())
    print("\n[1c] shape (записей, столбцов):", df.shape)
    print("\ninfo():")
    df.info()
    print("\n[1d] Уникальные значения Sex:", df["Sex"].unique())
    print("\n[1e] describe():")
    print(df.describe())


def pandas_2():
    df = TITANIC.copy()
    survived = df[df["Survived"] == 1]
    print(f"[2a] Выживших: {len(survived)}")
    sub = survived[["Name", "Age", "Sex"]]
    print("\n[2b] Name, Age, Sex у выживших:")
    print(sub.head(10))
    print("\n[2c] Средний возраст выживших по полу:")
    print(sub.groupby("Sex")["Age"].mean().round(2))
    print("\n[2d] Пассажиры с Age > 50:", len(df[df["Age"] > 50]))
    older = df[df["Age"] > 50].copy()
    print("\n[2e] Отдельный DataFrame older (первые 5 строк):")
    print(older.head())


def pandas_3():
    df = TITANIC.copy()
    print("[3a] Пропуски по столбцам:")
    print(df.isnull().sum())
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    print(f"\n[3b] Age заполнен медианой: {median_age}")
    df = df.dropna(subset=["Cabin", "Embarked"])
    print(f"\n[3c] После удаления строк с пропусками в Cabin/Embarked: {len(df)} строк")
    print("\n[3d] Пропуски после обработки:")
    print(df.isnull().sum())
    print("Остались ли пропуски:", bool(df.isnull().any().any()))
    clean = df.dropna().copy()
    print(f"\n[3e] Копия без пропусков: shape = {clean.shape}")


def pandas_4():
    df = TITANIC.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*\.)", expand=False)
    print("[4a] Титулы:")
    print(df["Title"].value_counts())
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    print("\n[4b] FamilySize (SibSp + Parch):")
    print(df[["Name", "SibSp", "Parch", "FamilySize"]].head())
    print("\n[4c] Доля выживших по полу, %:")
    print((df.groupby("Sex")["Survived"].mean() * 100).round(1))
    df = df.rename(columns={"Pclass": "Class", "Sex": "Gender",
                            "SibSp": "Siblings_Spouses", "Parch": "Parents_Children"})
    print("\n[4d] Столбцы после rename():", list(df.columns))
    df.to_csv(out_path("titanic_task4.csv"), index=False)
    saved("titanic_task4.csv")


def pandas_5():
    df = TITANIC.copy()
    died = int(df["Survived"].eq(0).sum())
    lived = int(df["Survived"].eq(1).sum())
    print(f"[5a] Всего: {len(df)}, погибло: {died}, выжило: {lived}")
    print("\n[5b] Средний возраст по классу билета:")
    print(df.groupby("Pclass")["Age"].mean().round(2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Age"].dropna(), bins=range(0, 90, 5), edgecolor="black")
    ax.set_title("Распределение возрастов пассажиров")
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Количество пассажиров")
    save_fig(fig, "titanic_task5_age_hist.png")

    by_class = df[df["Survived"] == 1].groupby("Pclass").size()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(by_class, labels=[f"{c} класс" for c in by_class.index],
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Доля выживших по классам билетов")
    save_fig(fig, "titanic_task5_survived_pie.png")


def pandas_6():
    df = TITANIC.copy()
    print("[6a] По возрасту (убывание), первые 5:")
    print(df.sort_values("Age", ascending=False).head()[["Name", "Age"]])
    old = df[df["Age"] > 60].sort_values("Fare", ascending=True)
    print(f"\n[6b] Старше 60 лет по возрастанию Fare ({len(old)} чел.), первые 5:")
    print(old[["Name", "Age", "Fare"]].head())
    print("\n[6c] Старше 18 лет по классам билетов:")
    print(df[df["Age"] > 18].groupby("Pclass").size())
    pivot = pd.pivot_table(df, index="Pclass", columns="Survived",
                           values="PassengerId", aggfunc="count", margins=True,
                           margins_name="Всего")
    pivot = pivot.rename(columns={0: "Погибли", 1: "Выжили"})
    pivot.columns.name = None
    with pd.ExcelWriter(out_path("titanic_task6.xlsx")) as writer:
        df.to_excel(writer, sheet_name="Данные", index=False)
        pivot.to_excel(writer, sheet_name="Сводная")
    saved("titanic_task6.xlsx")
    print("\n[6e] Сводная таблица (выжившие и погибшие по классам):")
    print(pivot)


def split_names(names):
    last = names.str.split(",", n=1).str[0].str.strip()
    rest = names.str.split(",", n=1).str[1].str.strip()
    title = rest.str.extract(r"^([^.]*\.)", expand=False)
    first = rest.str.replace(r"^[^.]*\.\s*", "", regex=True).str.strip()
    return pd.DataFrame({"Title": title, "First Name": first, "Last Name": last})


def pandas_7():
    df = TITANIC.copy()
    df["Age"] = df["Age"].round()
    print("[7a] Age округлён до целого:")
    print(df["Age"].head())
    df["Sex_num"] = df["Sex"].map({"male": 0, "female": 1})
    df["Sex_cat"] = df["Sex_num"].map({0: "male", 1: "female"})
    print("\n[7b] Sex -> число -> категория:")
    print(df[["Sex", "Sex_num", "Sex_cat"]].head())
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    print(f"\n[7c] Дубликатов найдено и удалено: {dups}. Строк осталось: {len(df)}")
    df["Name"] = df["Name"].str.lower()
    print("\n[7d] Name в нижнем регистре:")
    print(df["Name"].head())
    df = df.join(split_names(df["Name"])[["First Name", "Last Name"]])
    df["Full Name"] = df["First Name"] + " " + df["Last Name"]
    print("\n[7e] Full Name = First Name + Last Name:")
    print(df[["First Name", "Last Name", "Full Name"]].head())


def pandas_8():
    df = TITANIC.copy()
    df["Name"] = df["Name"].str.upper()
    print("[8a] Имена заглавными буквами:")
    print(df["Name"].head())
    df["Embarked"] = df["Embarked"].str.strip()
    print("\n[8b] Embarked после strip():", df["Embarked"].dropna().unique())
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    print("\n[8c] Cabin, пропуски заменены на 'Unknown':")
    print(df["Cabin"].value_counts().head())
    lengths = df["Name"].str.len()
    print(f"\n[8d] Самые длинные имена ({lengths.max()} символов):")
    print(df.loc[lengths == lengths.max(), "Name"].to_string(index=False))
    print(f"Самые короткие имена ({lengths.min()} символов):")
    print(df.loc[lengths == lengths.min(), "Name"].to_string(index=False))
    parts = split_names(df["Name"])
    print("\n[8e] Имя разделено на Title, First Name, Last Name:")
    print(parts.head())


def pandas_9():
    df = TITANIC.copy()
    print("[9a] Количество пассажиров по полу и классу:")
    print(df.groupby(["Sex", "Pclass"]).size())
    bins = list(range(0, 90, 10))
    labels = ["0–10"] + [f"{b + 1}–{b + 10}" for b in range(10, 80, 10)]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
    by_age = (df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100).round(1)
    print("\n[9b] Процент выживших по возрастным категориям:")
    print(by_age)
    by_class = (df.groupby("Pclass")["Survived"].mean() * 100).round(1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([f"{c} класс" for c in by_class.index], by_class.values,
           color=["gold", "silver", "peru"])
    for i, v in enumerate(by_class.values):
        ax.text(i, v + 1, f"{v}%", ha="center")
    ax.set_title("Процент выживших по классам билетов")
    ax.set_ylabel("Выжившие, %")
    ax.set_ylim(0, 100)
    save_fig(fig, "titanic_task9_survival_by_class.png")
    pivot = pd.pivot_table(df, index="Pclass", columns="Survived",
                           values="PassengerId", aggfunc="count")
    pivot = pivot.rename(columns={0: "Погибли", 1: "Выжили"})
    pivot.columns.name = None
    print("\n[9d] Сводная таблица (выжившие и погибшие по классам):")
    print(pivot)
    r = by_class
    print("\n[9e] Вывод:")
    print(f"Выживаемость: 1 класс — {r[1]}%, 2 класс — {r[2]}%, 3 класс — {r[3]}%.")
    if r[1] > r[2] > r[3]:
        print("Чем выше класс билета, тем выше шанс выжить: связь выраженная.")
    else:
        print("Монотонной зависимости выживаемости от класса не наблюдается.")


def pandas_10():
    merged_id = TITANIC_TEST.merge(TITANIC_GENDER, on="PassengerId")
    print("[10a] test.csv + gender_submission.csv по PassengerId:")
    print(merged_id[["PassengerId", "Name", "Survived"]].head())
    print("shape:", merged_id.shape)
    merged = TITANIC.merge(TITANIC_TEST, on="Name", how="left",
                           suffixes=("_train", "_test"))
    print(f"\n[10b] Левое объединение train и test по Name: {merged.shape}")
    merged["Age_diff"] = merged["Age_train"] - merged["Age_test"]
    print("\n[10c] Новый столбец Age_diff (Age_train - Age_test).")
    matched = merged[merged["PassengerId_test"].notna()]
    print(f"Имена из train нашлись в test у {len(matched)} пассажиров, у остальных "
          "Age_diff = NaN:")
    print(matched[["Name", "Age_train", "Age_test", "Age_diff"]])
    print("\n[10d] Проверка уникальности:")
    print("PassengerId_train уникален:", merged["PassengerId_train"].is_unique)
    print("Name уникален:", merged["Name"].is_unique)
    print("Полных дубликатов строк:", int(merged.duplicated().sum()))
    merged.to_json(out_path("titanic_task10_merged.json"), orient="records",
                   force_ascii=False, indent=2)
    saved("titanic_task10_merged.json")


def pandas_11():
    df = TITANIC.dropna(subset=["Age"])
    bins = np.arange(0, 90, 5)
    centers = (bins[:-1] + bins[1:]) / 2
    colors = {1: "tab:red", 2: "tab:green", 3: "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(df["Age"], bins=bins, color="lightgray", edgecolor="white",
            label="Все пассажиры")
    for cls in (1, 2, 3):
        counts, _ = np.histogram(df.loc[df["Pclass"] == cls, "Age"], bins=bins)
        ax.plot(centers, counts, "o", color=colors[cls], markersize=4)
        coef = np.polyfit(centers, counts, 3)
        ax.plot(centers, np.polyval(coef, centers), "-", color=colors[cls],
                label=f"Тренд, {cls} класс")
    ax.set_title("Распределение по возрасту и тренды по классам")
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Количество пассажиров")
    ax.legend()
    ax = axes[1]
    ax.hist([df.loc[df["Survived"] == 1, "Age"], df.loc[df["Survived"] == 0, "Age"]],
            bins=bins, label=["Выжили", "Погибли"], color=["tab:green", "tab:red"])
    ax.set_title("Выжившие и погибшие по возрасту")
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Количество пассажиров")
    ax.legend()
    save_fig(fig, "titanic_task11.pdf")
    print("\n[11e] Объяснение:")
    med = df.groupby("Pclass")["Age"].median()
    print(f"Медианный возраст: 1 класс — {med[1]:.0f}, 2 класс — {med[2]:.0f}, "
          f"3 класс — {med[3]:.0f}.")
    print("Основная масса пассажиров — люди 20–40 лет; в 3 классе пик смещён к "
          "молодому возрасту, в 1 классе распределение шире и старше.")
    a_l = df.loc[df["Survived"] == 1, "Age"].mean()
    a_d = df.loc[df["Survived"] == 0, "Age"].mean()
    print(f"Средний возраст выживших {a_l:.1f}, погибших {a_d:.1f}; "
          "среди детей доля выживших выше.")


def pandas_12():
    df = TITANIC.dropna(subset=["Age"])
    colors = {1: "tab:red", 2: "tab:green", 3: "tab:blue"}
    fig, ax = plt.subplots(figsize=(9, 6))
    stats = {}
    for cls in (1, 2, 3):
        part = df[df["Pclass"] == cls]
        ax.scatter(part["Age"], part["Fare"], color=colors[cls], alpha=0.5,
                   label=f"{cls} класс")
        slope, intercept = np.polyfit(part["Age"], part["Fare"], 1)
        xs = np.array([part["Age"].min(), part["Age"].max()])
        ax.plot(xs, slope * xs + intercept, color=colors[cls], linewidth=2)
        stats[cls] = (slope, part["Fare"].mean(), part["Age"].corr(part["Fare"]))
    ax.set_title("Стоимость билета в зависимости от возраста")
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Fare")
    ax.legend()
    save_fig(fig, "titanic_task12.svg")
    print("\n[12e] Закономерности:")
    for cls, (slope, mean_fare, corr) in stats.items():
        print(f"{cls} класс: средняя стоимость {mean_fare:.1f}, наклон регрессии "
              f"{slope:.3f}, корреляция Age–Fare {corr:.2f}")
    print("Стоимость билета определяется прежде всего классом: облака точек по "
          "классам расположены на разных уровнях Fare. Внутри класса связь с "
          "возрастом слабая; в 1 классе есть выбросы с очень дорогими билетами.")


def pandas_13():
    df = TITANIC.copy()
    rate = (df.groupby(["Sex", "Pclass"])["Survived"].mean() * 100).round(1)
    print("[13a, 13b] Процент выживших по группам (пол, класс):")
    print(rate)
    labels = [f"{s}, {c} кл." for s, c in rate.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, rate.values,
           color=["tab:pink" if s == "female" else "tab:blue" for s, _ in rate.index])
    for i, v in enumerate(rate.values):
        ax.text(i, v + 1, f"{v}%", ha="center")
    ax.set_title("Процент выживших по полу и классу билета")
    ax.set_ylabel("Выжившие, %")
    ax.set_ylim(0, 105)
    save_fig(fig, "titanic_task13.png")
    sex = (df.groupby("Sex")["Survived"].mean() * 100).round(1)
    cls = (df.groupby("Pclass")["Survived"].mean() * 100).round(1)
    print("\n[13d] Выводы:")
    print(f"Женщины выживали в {sex['female']}% случаев, мужчины — в {sex['male']}%.")
    print(f"По классам: 1 — {cls[1]}%, 2 — {cls[2]}%, 3 — {cls[3]}%.")
    print(f"Лучшая группа: {rate.idxmax()} ({rate.max()}%), "
          f"худшая: {rate.idxmin()} ({rate.min()}%).")
    print("Пол влияет сильнее класса, но оба фактора значимы: "
          "женщины высших классов почти все выжили, мужчины 3 класса — единицы.")


def pandas_14():
    df = TITANIC.copy()
    num_cols = ["Age", "Fare", "SibSp", "Parch"]
    print("[14a] Распределение ключевых столбцов:")
    print(df[num_cols].describe().round(2))
    for col in ("Sex", "Pclass", "Embarked", "Survived"):
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), num_cols):
        ax.hist(df[col].dropna(), bins=20, edgecolor="black")
        ax.set_title(f"Гистограмма: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Количество")
    save_fig(fig, "titanic_task14_hist.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), ("Sex", "Pclass", "Embarked", "Survived")):
        df[col].value_counts().sort_index().plot.bar(ax=ax, edgecolor="black")
        ax.set_title(f"Барчарт: {col}")
        ax.set_ylabel("Количество")
        ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "titanic_task14_bars.png")

    print("\n[14c] Выбросы (правило 1.5·IQR):")
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int((df[col].lt(low) | df[col].gt(high)).sum())
        print(f"{col}: границы [{low:.2f}; {high:.2f}], выбросов: {n_out}, "
              f"максимум {df[col].max()}")
    print("\n[14d] Выводы о качестве и структуре данных:")
    miss = (df.isnull().mean() * 100).round(1)
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов; "
          f"дубликатов: {int(df.duplicated().sum())}.")
    print("Пропуски, %:", {k: v for k, v in miss.items() if v > 0})
    print("Fare сильно скошен вправо (дорогие билеты 1 класса — выбросы), "
          "SibSp и Parch в основном 0–2; Cabin почти не пригоден из-за пропусков.")


def pandas_15():
    df = TITANIC.copy()
    print("[15a] Пропуски:")
    print(df.isnull().sum())
    df["Age"] = df["Age"].fillna(df["Age"].median())
    print("\n[15b] Age заполнен медианой.")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("[15c] Sex: male -> 0, female -> 1:")
    print(df["Sex"].head())
    many_na = df.columns[df.isnull().mean() > 0.5].tolist()
    df = df.drop(columns=many_na + ["Ticket"])
    print(f"\n[15d] Удалены столбцы: {many_na + ['Ticket']}")
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    print("Пропуски после обработки:", int(df.isnull().sum().sum()))
    df.to_csv(out_path("titanic_task15_clean.csv"), index=False)
    saved("titanic_task15_clean.csv")


def pandas_16():
    df = VGSALES.dropna(subset=["Global_Sales"])
    by_platform = (df.groupby("Platform")["Global_Sales"].sum()
                   .sort_values(ascending=False))
    print("[1d] Топ-3 платформы:")
    print(by_platform.head(3))
    print("Худшие 3 платформы:")
    print(by_platform.tail(3))
    fig, ax = plt.subplots(figsize=(9, 9))
    by_platform.sort_values().plot.barh(ax=ax)
    ax.set_title("Мировые продажи по платформам")
    ax.set_xlabel("Продажи, млн копий")
    ax.set_ylabel("Платформа")
    save_fig(fig, "vg_task1_platforms.png")
    print("\n[1g] Суммарные продажи по платформам:")
    print_table(by_platform.round(2))


def pandas_17():
    df = VGSALES
    by_genre = (df.groupby("Genre")["Global_Sales"].sum()
                .sort_values(ascending=False))
    top5 = by_genre.head(5)
    print("[2c] Топ-5 жанров:")
    print(top5.round(2))
    fig, ax = plt.subplots(figsize=(8, 5))
    by_genre.plot.bar(ax=ax)
    ax.set_title("Популярность жанров (мировые продажи)")
    ax.set_xlabel("Жанр")
    ax.set_ylabel("Продажи, млн копий")
    save_fig(fig, "vg_task2_genres.pdf")
    print("\n[2f] Суммарные продажи по жанрам:")
    print_table(by_genre.round(2))
    top5.to_csv(out_path("vg_task2_top5_genres.csv"), header=True)
    saved("vg_task2_top5_genres.csv")


def pandas_18():
    df = VGSALES.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    by_year = df.groupby("Year")["Global_Sales"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_year.index, by_year.values, marker="o")
    ax.set_title("Мировые продажи по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Продажи, млн копий")
    ax.grid(alpha=0.3)
    save_fig(fig, "vg_task3_years.svg")
    print(f"[3d] Максимум продаж: {by_year.idxmax()} год ({by_year.max():.2f} млн)")
    print(f"Минимум продаж: {by_year.idxmin()} год ({by_year.min():.2f} млн)")
    print("\n[3f] Суммарные продажи по годам:")
    print_table(by_year.round(2))
    by_year.to_frame("Global_Sales").to_excel(out_path("vg_task3_years.xlsx"))
    saved("vg_task3_years.xlsx")


def pandas_19():
    df = VGSALES
    regions = ["NA_Sales", "EU_Sales", "JP_Sales"]
    totals = df[regions].sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Северная Америка", "Европа", "Япония"], totals.values,
           color=["tab:blue", "tab:green", "tab:red"])
    ax.set_title("Продажи игр по регионам")
    ax.set_ylabel("Продажи, млн копий")
    save_fig(fig, "vg_task4_regions.jpg")
    print(f"[4c] Наибольшие продажи: {totals.idxmax()} ({totals.max():.2f} млн)")
    print(f"Наименьшие продажи: {totals.idxmin()} ({totals.min():.2f} млн)")
    print("\n[4d] Игры, популярные в конкретном регионе "
          "(доля региона > 50% мировых продаж, Global_Sales >= 1):")
    big = df[df["Global_Sales"] >= 1]
    for reg in regions:
        local = big[big[reg] / big["Global_Sales"] > 0.5]
        print(f"\n{reg}: таких игр {len(local)}, топ-5:")
        print(local.nlargest(5, reg)[["Name", "Platform", reg, "Global_Sales"]]
              .to_string(index=False))
    print("\n[4f] Суммарные продажи по регионам:")
    print(totals.round(2))
    df[["Name", "Platform"] + regions].to_json(
        out_path("vg_task4_regions.json"), orient="records", force_ascii=False,
        indent=2)
    saved("vg_task4_regions.json")


def pandas_20():
    df = VGSALES.sort_values("Global_Sales", ascending=False)
    top10 = df.head(10)[["Name", "Genre", "Platform", "Global_Sales"]]
    print("[5c] Топ-10 игр:")
    print(top10.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, _, _ = ax.pie(top10["Global_Sales"], autopct="%1.1f%%", startangle=90)
    ax.legend(wedges, top10["Name"] + " (" + top10["Platform"] + ")",
              loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Доли игр из топ-10 в их суммарных продажах")
    save_fig(fig, "vg_task5_top10_pie.png")
    per_game = (VGSALES.groupby("Name")["Global_Sales"].sum()
                .sort_values(ascending=False))
    print("\n[5f] Суммарные продажи для каждой игры (по всем платформам):")
    print_table(per_game.round(2))
    top10.to_html(out_path("vg_task5_top10.html"), index=False)
    saved("vg_task5_top10.html")


def pandas_21():
    df = VGSALES
    by_pub = (df.groupby("Publisher")["Global_Sales"].sum()
              .sort_values(ascending=False))
    top5 = by_pub.head(5)
    print("[6c] Топ-5 издателей:")
    print(top5.round(2))
    fig, ax = plt.subplots(figsize=(8, 5))
    top5.sort_values().plot.barh(ax=ax)
    ax.set_title("Топ-5 издателей по мировым продажам")
    ax.set_xlabel("Продажи, млн копий")
    ax.set_ylabel("Издатель")
    save_fig(fig, "vg_task6_publishers.pdf")
    print("\n[6f] Суммарные продажи по издателям:")
    print_table(by_pub.round(2))
    top5.to_csv(out_path("vg_task6_top5_publishers.csv"), header=True)
    saved("vg_task6_top5_publishers.csv")


def pandas_22():
    df = VGSALES
    sales = pd.pivot_table(df, index="Genre", columns="Platform",
                           values="Global_Sales", aggfunc="sum", fill_value=0)
    counts = pd.pivot_table(df, index="Genre", columns="Platform",
                            values="Name", aggfunc="count", fill_value=0)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    draw_heatmap(axes[0], sales, "Продажи по жанрам и платформам",
                 cbar_label="млн копий")
    draw_heatmap(axes[1], counts, "Количество игр по жанрам и платформам",
                 cmap="Blues", cbar_label="игр")
    save_fig(fig, "vg_task7_heatmaps.png")
    diversity = df.groupby("Platform")["Genre"].nunique()
    best = diversity[diversity == diversity.max()].index.tolist()
    worst = diversity[diversity == diversity.min()].index.tolist()
    print(f"[7c] Наибольшее разнообразие жанров: {best} ({diversity.max()} жанров)")
    print(f"Наименьшее разнообразие: {worst} ({diversity.min()} жанр(а))")
    combo = (df.groupby(["Genre", "Platform"])["Global_Sales"].sum()
             .sort_values(ascending=False))
    print("\n[7e] Суммарные продажи по комбинациям жанр + платформа:")
    print_table(combo.round(2))
    with pd.ExcelWriter(out_path("vg_task7_genre_platform.xlsx")) as writer:
        sales.to_excel(writer, sheet_name="Продажи")
        counts.to_excel(writer, sheet_name="Количество игр")
    saved("vg_task7_genre_platform.xlsx")


def pandas_23():
    df = VGSALES.dropna(subset=["Year"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["Year"], df["Global_Sales"], s=8, alpha=0.4, rasterized=True)
    ax.set_title("Мировые продажи и год выпуска")
    ax.set_xlabel("Год выпуска")
    ax.set_ylabel("Мировые продажи, млн копий")
    save_fig(fig, "vg_task8_scatter.svg")
    corr = df["Year"].corr(df["Global_Sales"])
    print(f"[8c, 8e] Коэффициент корреляции Пирсона (Year, Global_Sales): {corr:.4f}")
    df[["Year", "Global_Sales"]].to_csv(out_path("vg_task8_year_sales.csv"),
                                        index=False)
    saved("vg_task8_year_sales.csv")


def pandas_24():
    df = VGSALES.copy()
    df["NA-EU"] = df["NA_Sales"] - df["EU_Sales"]
    df["NA-JP"] = df["NA_Sales"] - df["JP_Sales"]
    df["EU-JP"] = df["EU_Sales"] - df["JP_Sales"]
    reg = df[["NA_Sales", "EU_Sales", "JP_Sales"]]
    df["Max_diff"] = reg.max(axis=1) - reg.min(axis=1)
    diff_cols = ["Name", "Platform", "NA-EU", "NA-JP", "EU-JP", "Max_diff"]
    ranked = df.sort_values("Max_diff", ascending=False)
    top10 = ranked.head(10)
    print("[9c] Топ-10 игр с наибольшими региональными различиями:")
    print(top10[diff_cols].round(2).to_string(index=False))
    labels = top10["Name"].str.slice(0, 22) + " (" + top10["Platform"] + ")"
    x = np.arange(len(top10))
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, col in enumerate(["NA-EU", "NA-JP", "EU-JP"]):
        ax.bar(x + (k - 1) * 0.27, top10[col], width=0.27, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Региональные различия продаж: топ-10 игр")
    ax.set_ylabel("Разность продаж, млн копий")
    ax.legend()
    save_fig(fig, "vg_task9_diffs.jpg")
    print("\n[9f] Разность в продажах для каждой игры:")
    print_table(ranked[diff_cols].round(2))
    ranked[diff_cols].to_json(out_path("vg_task9_diffs.json"), orient="records",
                              force_ascii=False, indent=2)
    saved("vg_task9_diffs.json")


def pandas_25():
    df = VGSALES.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    grouped = df.groupby(["Genre", "Year"])["Global_Sales"].sum()
    table = grouped.unstack("Genre").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    table.plot(ax=ax)
    ax.set_title("Изменение популярности жанров по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Продажи, млн копий")
    ax.legend(ncol=2, fontsize=8)
    save_fig(fig, "vg_task10_genres_years.png")
    old = df[(df["Year"] >= 2000) & (df["Year"] <= 2009)]
    new = df[df["Year"] >= 2010]
    share_old = old.groupby("Genre")["Global_Sales"].sum() / old["Global_Sales"].sum()
    share_new = new.groupby("Genre")["Global_Sales"].sum() / new["Global_Sales"].sum()
    change = ((share_new - share_old) * 100).sort_values(ascending=False).round(2)
    print("[10d] Изменение доли жанра в продажах: 2010+ против 2000–2009, п.п.:")
    print(change)
    print("Стали популярнее:", change[change > 0].index.tolist())
    print("\n[10f] Суммарные продажи по жанрам и годам:")
    print_table(grouped.round(2))
    table.to_excel(out_path("vg_task10_genres_years.xlsx"))
    saved("vg_task10_genres_years.xlsx")


def pandas_26():
    df = VGSALES.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    by_year = df.groupby("Year")["Global_Sales"].sum()
    slope, intercept = np.polyfit(by_year.index, by_year.values, 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(by_year.index, by_year.values, label="Продажи за год")
    ax.plot(by_year.index, slope * by_year.index + intercept, color="red",
            label="Линейная регрессия")
    ax.set_title("Зависимость мировых продаж от года выпуска")
    ax.set_xlabel("Год выпуска")
    ax.set_ylabel("Мировые продажи, млн копий")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "vg_task11_trend.png")
    print(f"[11e] Уравнение регрессии: Global_Sales = {slope:.3f} * Year "
          f"+ ({intercept:.2f})")
    print(f"Угловой коэффициент: {slope:.3f}, свободный член: {intercept:.2f}")
    by_year.to_frame("Global_Sales").to_csv(out_path("vg_task11_year_sales.csv"))
    saved("vg_task11_year_sales.csv")


def pandas_27():
    df = VGSALES
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["NA_Sales"], df["EU_Sales"], s=8, alpha=0.4,
               label="Европа (по оси Y) / Сев. Америка (X)", rasterized=True)
    ax.scatter(df["NA_Sales"], df["JP_Sales"], s=8, alpha=0.4,
               label="Япония (по оси Y) / Сев. Америка (X)", rasterized=True)
    ax.set_title("Региональные продажи: NA, EU, JP")
    ax.set_xlabel("Продажи в Северной Америке (NA_Sales), млн")
    ax.set_ylabel("Продажи в Европе / Японии, млн")
    ax.legend()
    save_fig(fig, "vg_task12_regions_scatter.svg")
    coords = df[["Name", "NA_Sales", "EU_Sales", "JP_Sales"]]
    print("[12e] Координаты точек (X = NA_Sales; Y = EU_Sales и JP_Sales):")
    print_table(coords)
    coords.to_json(out_path("vg_task12_regions.json"), orient="records",
                   force_ascii=False, indent=2)
    saved("vg_task12_regions.json")


def pandas_28():
    df = VGSALES
    sales = pd.pivot_table(df, index="Genre", columns="Platform",
                           values="Global_Sales", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(16, 7))
    draw_heatmap(ax, sales, "Популярность жанров на платформах (млн копий)",
                 annotate=True, fmt="{:.0f}", cbar_label="Продажи, млн копий")
    ax.set_xlabel("Платформа")
    ax.set_ylabel("Жанр")
    save_fig(fig, "vg_task13_heatmap.png")
    print("[13e] Значения тепловой карты (млн копий):")
    print(sales.round(1).to_string())
    sales.to_excel(out_path("vg_task13_genre_platform.xlsx"))
    saved("vg_task13_genre_platform.xlsx")


def pandas_29():
    df = VGSALES[VGSALES["Year"] > 2010]
    print(f"[14b] Игр после 2010 года: {len(df)}")
    by_genre = (df.groupby("Genre")["Global_Sales"].sum()
                .sort_values(ascending=False))
    top5 = by_genre.head(5)
    print("\n[14d] Топ-5 жанров:")
    print(top5.round(2))
    fig, ax = plt.subplots(figsize=(8, 5))
    by_genre.plot.bar(ax=ax)
    ax.set_title("Популярность жанров (игры после 2010 года)")
    ax.set_xlabel("Жанр")
    ax.set_ylabel("Продажи, млн копий")
    save_fig(fig, "vg_task14_genres_after2010.pdf")
    print("\n[14g] Суммарные продажи по жанрам:")
    print_table(by_genre.round(2))
    top5.to_csv(out_path("vg_task14_top5_genres.csv"), header=True)
    saved("vg_task14_top5_genres.csv")


def pandas_30():
    df = VGSALES.select_dtypes(include="number")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="Коэффициент корреляции")
    ax.set_title("Корреляционная матрица")
    save_fig(fig, "vg_task15_corr.png")
    print("[15e] Корреляционная матрица:")
    print(corr.round(3))
    corr.to_excel(out_path("vg_task15_corr.xlsx"))
    saved("vg_task15_corr.xlsx")


PYTHON_TASKS = {i: f for i, f in enumerate([
    task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9,
    task_10, task_11, task_12, task_13, task_14, task_15, task_16, task_17,
    task_18, task_19, task_20, task_21, task_22, task_23, task_24, task_25,
    task_26, task_27, task_28, task_29, task_30, task_31, task_32, task_33,
    task_34, task_35, task_36,
], 1)}

NUMPY_TASKS = {i: f for i, f in enumerate([
    numpy_1, numpy_2, numpy_3, numpy_4, numpy_5, numpy_6, numpy_7, numpy_8,
    numpy_9, numpy_10, numpy_11, numpy_12, numpy_13, numpy_14, numpy_15,
    numpy_16, numpy_17, numpy_18, numpy_19, numpy_20, numpy_21, numpy_22,
    numpy_23, numpy_24, numpy_25, numpy_26, numpy_27, numpy_28, numpy_29,
], 1)}

PANDAS_TASKS = {i: f for i, f in enumerate([
    pandas_1, pandas_2, pandas_3, pandas_4, pandas_5, pandas_6, pandas_7,
    pandas_8, pandas_9, pandas_10, pandas_11, pandas_12, pandas_13, pandas_14,
    pandas_15, pandas_16, pandas_17, pandas_18, pandas_19, pandas_20,
    pandas_21, pandas_22, pandas_23, pandas_24, pandas_25, pandas_26,
    pandas_27, pandas_28, pandas_29, pandas_30,
], 1)}

SECTIONS = {
    "python": ("Python", PYTHON_TASKS),
    "numpy": ("NumPy", NUMPY_TASKS),
    "pandas": ("Pandas", PANDAS_TASKS),
}


def parse_choice(raw):
    m = re.fullmatch(r"(?:(python|numpy|pandas)[\s_\-]*)?(?:(ii|i)[\s_\-]+)?(\d+)",
                     raw.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    section = (m.group(1) or "python").lower()
    part = (m.group(2) or "").lower()
    num = int(m.group(3))
    if part and section != "pandas":
        return None
    if part == "ii":
        num += 15
    return section, num


def main():
    print(f"Разделы: Python (1–{len(PYTHON_TASKS)}), NumPy (1–{len(NUMPY_TASKS)}), "
          f"Pandas (1–{len(PANDAS_TASKS)}: 1–15 Titanic, 16–30 Video Game Sales).")
    while True:
        raw = input("\nЗадание (например, 7, Python_7, NumPy_3, Pandas_4; "
                    "exit — выход): ")
        if raw.strip().lower() in ("exit", "0"):
            break
        choice = parse_choice(raw)
        if choice is None:
            print("Неверный формат. Пример: Python_1, NumPy_1 или Pandas_1.")
            continue
        section, num = choice
        title, tasks = SECTIONS[section]
        if num not in tasks:
            print(f"В разделе {title} нет задания {num}.")
            continue
        if section == "pandas" and not PANDAS_READY:
            print("Не все файлы датасетов найдены в папке data/ "
                  "(train.csv, test.csv, gender_submission.csv, vgsales.csv).")
            continue
        print(f"--- {title}_{num} ---")
        tasks[num]()


if __name__ == "__main__":
    main()