"""
Диагностический скрипт для проверки парсинга dEStep
Запуск: python diagnostic_check.py ваш_лог.log
"""

import sys
import re

if len(sys.argv) < 2:
    print("Использование: python diagnostic_check.py ваш_лог.log")
    sys.exit(1)

log_file = sys.argv[1]

print("=" * 80)
print("ДИАГНОСТИКА ПАРСИНГА dEStep")
print("=" * 80)

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Ищем заголовок таблицы шагов
print("\n1. Поиск заголовка таблицы шагов:")
for i, line in enumerate(lines):
    if 'Step#' in line and 'dE' in line:
        print(f"\nСтрока {i}: {line.strip()}")

        # Показываем следующие 5 строк данных
        print("\nПримеры строк данных:")
        for j in range(i + 1, min(i + 6, len(lines))):
            if lines[j].strip() and not lines[j].startswith('---'):
                print(f"  Строка {j}: {lines[j].strip()}")
        break

# Ищем секцию с итоговой энергией
print("\n2. Итоговая сводка:")
for i, line in enumerate(lines):
    if 'Energy deposit' in line:
        print(f"  {line.strip()}")
    if 'Energy leakage' in line:
        print(f"  {line.strip()}")

# Анализируем числа в строках данных
print("\n3. Анализ чисел в строках данных:")
in_table = False
de_values = []

for i, line in enumerate(lines):
    if 'Step#' in line and 'dE' in line:
        in_table = True
        continue

    if in_table and line.strip() and not line.startswith('---'):
        # Пытаемся найти все числа с единицами
        numbers = re.findall(r'([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)', line)
        if len(numbers) >= 5:
            # 4-е и 5-е числа обычно KineE и dEStep
            if len(numbers) >= 5:
                kine_e = f"{numbers[3][0]} {numbers[3][1]}"
                de_step = f"{numbers[4][0]} {numbers[4][1]}"
                de_values.append((kine_e, de_step))

                if len(de_values) <= 5:
                    print(f"  KineE: {kine_e:>15s}, dEStep: {de_step:>15s}")

    if len(de_values) > 20:
        break

print("\n4. Рекомендации:")
print("  - Проверьте, что dEStep действительно в единицах eV/keV/MeV")
print("  - Убедитесь, что парсер правильно определяет позицию dEStep")
print("  - Для большинства шагов dEStep должен быть 0 или очень маленьким")
print("\n" + "=" * 80)