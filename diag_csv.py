"""
Проверка содержимого CSV - что реально распарсилось
"""
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Использование: python check_csv.py parsed_data.csv")
    sys.exit(1)

csv_file = sys.argv[1]

print("=" * 100)
print("АНАЛИЗ РАСПАРСЕННЫХ ДАННЫХ")
print("=" * 100)

df = pd.read_csv(csv_file)

print(f"\n1. ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"  Всего строк: {len(df)}")
print(f"  Колонки: {list(df.columns)}")

print(f"\n2. СТАТИСТИКА ПО dE_step:")
print(f"  Min:    {df['de_step'].min():.6f} MeV")
print(f"  Max:    {df['de_step'].max():.6f} MeV")
print(f"  Mean:   {df['de_step'].mean():.6f} MeV")
print(f"  Median: {df['de_step'].median():.6f} MeV")
print(f"  Sum:    {df['de_step'].sum():.6f} MeV")

print(f"\n3. СТАТИСТИКА ПО kine_e:")
print(f"  Min:    {df['kine_e'].min():.6f} MeV")
print(f"  Max:    {df['kine_e'].max():.6f} MeV")
print(f"  Mean:   {df['kine_e'].mean():.6f} MeV")
print(f"  Median: {df['kine_e'].median():.6f} MeV")

print(f"\n4. РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ de_step:")
print(df['de_step'].value_counts().head(20))

print(f"\n5. ПЕРВЫЕ 10 СТРОК (для проверки):")
print(df[['step_num', 'x', 'y', 'z', 'kine_e', 'de_step', 'process']].head(10).to_string())

print(f"\n6. ПОДОЗРИТЕЛЬНЫЕ СТРОКИ (de_step > 0.1 MeV):")
suspicious = df[df['de_step'] > 0.1]
if len(suspicious) > 0:
    print(f"  Найдено {len(suspicious)} подозрительных строк:")
    print(suspicious[['step_num', 'kine_e', 'de_step', 'process', 'particle']].head(20).to_string())
else:
    print("  Нет подозрительных строк ✅")

print(f"\n7. СТРОКИ С de_step == kine_e (признак ошибки парсинга):")
same_values = df[abs(df['de_step'] - df['kine_e']) < 0.001]
if len(same_values) > 0:
    print(f"  ❌ Найдено {len(same_values)} строк где de_step ≈ kine_e!")
    print("  ЭТО ОШИБКА ПАРСИНГА - dE не должен равняться KineE!")
    print(same_values[['step_num', 'kine_e', 'de_step', 'process']].head(10).to_string())
else:
    print("  ✅ Нет строк с de_step == kine_e")

print("\n" + "=" * 100)
print("ВЫВОД:")
if df['de_step'].mean() > 0.01:
    print("  ❌ Средний dE слишком большой - парсинг неправильный!")
    print("  dEStep в большинстве случаев должен быть 0 или очень маленьким")
else:
    print("  ✅ Средний dE выглядит разумно")

if len(same_values) > 0:
    print("  ❌ dEStep равен KineE - парсер путает поля!")
else:
    print("  ✅ dEStep и KineE различаются")

print("=" * 100)