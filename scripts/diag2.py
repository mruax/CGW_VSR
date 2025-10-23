"""
Детальная диагностика парсинга - показывает ЧТО именно считается dEStep
"""
import sys
import re

if len(sys.argv) < 2:
    print("Использование: python detailed_diagnostic.py ваш_лог.log")
    sys.exit(1)

log_file = sys.argv[1]

print("=" * 100)
print("ДЕТАЛЬНАЯ ДИАГНОСТИКА - ЧТО ПАРСИТСЯ КАК dEStep")
print("=" * 100)

ENERGY_UNITS = {'eV': 1e-6, 'keV': 1e-3, 'MeV': 1.0, 'GeV': 1e3, 'TeV': 1e6, 'meV': 1e-9}
LENGTH_UNITS = {'fm': 1e-12, 'nm': 1e-6, 'um': 1e-3, 'mm': 1.0, 'cm': 10.0, 'm': 1e3, 'km': 1e6}

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print("\n1. Ищем строки с данными шагов...")
in_table = False
samples = []

for i, line in enumerate(lines):
    # Пропускаем заголовки
    if 'Step#' in line and any(x in line for x in ['KineE', 'dE']):
        in_table = True
        print(f"\nЗаголовок найден на строке {i}:")
        print(f"  {line.strip()}")
        continue

    if not in_table:
        continue

    # Пропускаем пустые и разделители
    line_stripped = line.strip()
    if not line_stripped or line_stripped.startswith('---') or line_stripped.startswith('==='):
        continue

    # Удаляем префикс потока
    if 'G4WT' in line or 'G4MT' in line:
        parts = line.split('>', 1)
        if len(parts) > 1:
            line_stripped = parts[1].strip()

    tokens = line_stripped.split()

    # Проверяем, что это строка данных (начинается с числа)
    if len(tokens) < 10:
        continue

    try:
        step_num = int(tokens[0])
    except:
        continue

    # Собираем пары (значение, единица)
    pairs = []
    j = 1
    while j < len(tokens):
        try:
            val = float(tokens[j])
            unit = ""
            if j + 1 < len(tokens):
                next_tok = tokens[j + 1]
                if next_tok in ENERGY_UNITS or next_tok in LENGTH_UNITS:
                    unit = next_tok
                    j += 2
                else:
                    j += 1
            else:
                j += 1
            pairs.append((val, unit))
        except:
            j += 1

    if len(pairs) >= 7 and len(samples) < 10:
        samples.append({
            'line_num': i,
            'step': step_num,
            'pairs': pairs,
            'raw': line.strip()[:100]
        })

print(f"\n2. Найдено {len(samples)} примеров строк данных. Анализируем первые 10:")
print("=" * 100)

total_de_old_method = 0
total_de_new_method = 0

for idx, sample in enumerate(samples):
    print(f"\nПример {idx + 1} (строка {sample['line_num']}):")
    print(f"Сырая строка: {sample['raw']}")
    print(f"\nРазобранные пары (значение, единица):")

    for i, (val, unit) in enumerate(sample['pairs'][:9]):
        field = "?"
        if i == 0:
            field = "X"
        elif i == 1:
            field = "Y"
        elif i == 2:
            field = "Z"
        elif i == 3:
            field = "KineE"
        elif i == 4:
            field = "dEStep"
        elif i == 5:
            field = "StepLeng"
        elif i == 6:
            field = "TrakLeng"

        print(f"  [{i}] {field:10s}: {val:12.6f} {unit:5s}", end="")

        # Проверяем единицы
        if i <= 2:  # X, Y, Z
            if unit and unit not in LENGTH_UNITS:
                print(" ❌ Не длина!", end="")
        elif i == 3 or i == 4:  # KineE, dEStep
            if unit and unit not in ENERGY_UNITS:
                print(" ❌ Не энергия!", end="")

        print()

    # Старый метод (просто 4-е число как dE)
    if len(sample['pairs']) >= 5:
        de_old = sample['pairs'][4][0]  # 4-я пара
        unit_old = sample['pairs'][4][1]
        # Конвертируем в MeV
        de_old_mev = de_old * ENERGY_UNITS.get(unit_old, 1.0)
        total_de_old_method += de_old_mev

        print(f"\n  СТАРЫЙ МЕТОД: dE = пара[4] = {de_old} {unit_old} = {de_old_mev:.6f} MeV")

    # Новый метод (проверяем единицы)
    # Ищем первую энергию после 3 длин
    energy_pairs = [p for p in sample['pairs'][3:5] if p[1] in ENERGY_UNITS]
    if len(energy_pairs) >= 2:
        de_new = energy_pairs[1][0]  # Вторая энергия это dE
        unit_new = energy_pairs[1][1]
        de_new_mev = de_new * ENERGY_UNITS.get(unit_new, 1.0)
        total_de_new_method += de_new_mev

        print(f"  НОВЫЙ МЕТОД: dE = вторая энергия = {de_new} {unit_new} = {de_new_mev:.6f} MeV")

print("\n" + "=" * 100)
print("3. ИТОГОВАЯ СТАТИСТИКА (по 10 примерам):")
print(f"  Сумма dE (старый метод): {total_de_old_method:.6f} MeV")
print(f"  Сумма dE (новый метод): {total_de_new_method:.6f} MeV")
print(f"  Средний dE (старый): {total_de_old_method / len(samples):.6f} MeV")
print(f"  Средний dE (новый): {total_de_new_method / len(samples):.6f} MeV")

print("\n4. ВЫВОД:")
if total_de_old_method > 1.0:
    print("  ❌ Старый метод даёт огромные значения - парсит неправильное поле!")
else:
    print("  ✅ Старый метод работает нормально")

if total_de_new_method < 0.01:
    print("  ✅ Новый метод даёт разумные значения")
else:
    print("  ⚠️  Новый метод тоже даёт большие значения")

print("\n5. РЕКОМЕНДАЦИЯ:")
print("  Посмотрите на примеры выше и проверьте:")
print("  - Какое поле имеет единицу 'eV' или 'keV' (это должно быть dEStep)")
print("  - Какое поле имеет единицу 'MeV' (это должно быть KineE)")
print("  - Правильно ли определены позиции полей")

print("=" * 100)