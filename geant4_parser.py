"""
Geant4 Log Parser - Доработанная версия
Программа для парсинга, анализа и визуализации логов симуляции Geant4
с корректным подсчетом энергии и процессов
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class StepData:
    """Класс для хранения данных одного шага"""
    thread: str
    step_num: int
    x: float
    y: float
    z: float
    kine_e: float
    de_step: float
    step_leng: float
    trak_leng: float
    volume: str
    process: str
    track_id: int
    parent_id: int
    particle: str

    # Единицы измерения
    coord_unit: str = "mm"
    energy_unit: str = "MeV"
    length_unit: str = "mm"


class Geant4LogParser:
    """Парсер логов Geant4"""

    # Регулярные выражения для парсинга
    THREAD_PATTERN = r'(G4WT\d+)\s*>'
    TRACK_INFO_PATTERN = r'Track ID\s*=\s*(\d+).*?Parent ID\s*=\s*(\d+)'
    PARTICLE_PATTERN = r'Particle\s*=\s*(\w+)'

    # Паттерны для итоговой сводки
    ENERGY_DEPOSIT_PATTERN = r'Energy deposit[:\s]+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)'
    ENERGY_LEAKAGE_PATTERN = r'Energy leakage[:\s]+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)'
    PROCESS_FREQ_PATTERN = r'(\w+)\s*=\s*(\d+)'

    # Конверсия единиц в MeV и mm
    ENERGY_UNITS = {'eV': 1e-6, 'keV': 1e-3, 'MeV': 1.0, 'GeV': 1e3, 'TeV': 1e6, 'meV': 1e-9}
    LENGTH_UNITS = {'fm': 1e-12, 'nm': 1e-6, 'um': 1e-3, 'mm': 1.0, 'cm': 10.0, 'm': 1e3, 'km': 1e6}

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.steps: List[StepData] = []
        self.summary: Dict = {}
        self.current_thread = ""
        self.current_track_id = 0
        self.current_parent_id = 0
        self.current_particle = ""

        # Для отслеживания энергии треков
        self.track_initial_energy: Dict[int, float] = {}
        self.track_final_energy: Dict[int, float] = {}
        self.track_particles: Dict[int, str] = {}

    def convert_energy(self, value: float, unit: str) -> float:
        """Конвертация энергии в MeV"""
        return value * self.ENERGY_UNITS.get(unit, 1.0)

    def convert_length(self, value: float, unit: str) -> float:
        """Конвертация длины в mm"""
        return value * self.LENGTH_UNITS.get(unit, 1.0)

    def parse_log(self, debug: bool = False) -> None:
        """Основной метод парсинга лога"""
        print(f"Парсинг файла: {self.log_file}")

        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        in_summary = False
        in_step_table = False
        process_frequencies = {}
        step_header_found = False
        debug_lines_sample = []

        for i, line in enumerate(lines):
            # Сохраняем примеры строк для отладки
            if debug and len(debug_lines_sample) < 100 and 'Step#' not in line and len(line.strip()) > 20:
                debug_lines_sample.append((i, line.strip()))

            # Поиск префикса потока
            thread_match = re.search(self.THREAD_PATTERN, line)
            if thread_match:
                self.current_thread = thread_match.group(1)

            # Поиск информации о треке
            track_match = re.search(self.TRACK_INFO_PATTERN, line)
            if track_match:
                self.current_track_id = int(track_match.group(1))
                self.current_parent_id = int(track_match.group(2))

            # Поиск типа частицы
            particle_match = re.search(self.PARTICLE_PATTERN, line)
            if particle_match:
                self.current_particle = particle_match.group(1)
                if self.current_track_id not in self.track_particles:
                    self.track_particles[self.current_track_id] = self.current_particle

            # Определение начала таблицы шагов
            if 'Step#' in line and any(x in line for x in ['KineE', 'dE', 'StepLen']):
                step_header_found = True
                in_step_table = True
                if debug:
                    print(f"\n[DEBUG] Найден заголовок таблицы шагов на строке {i}:")
                    print(f"  {line.strip()}")
                continue

            # Парсинг строк данных шагов
            if in_step_table and step_header_found:
                step_data = self._parse_step_line_simple(line, i, debug)
                if step_data:
                    self.steps.append(step_data)

                    # Отслеживание начальной и конечной энергии трека
                    track_id = step_data.track_id
                    if track_id not in self.track_initial_energy:
                        self.track_initial_energy[track_id] = step_data.kine_e
                    self.track_final_energy[track_id] = step_data.kine_e

                elif line.strip() == '' or line.startswith('---') or line.startswith('==='):
                    in_step_table = False

            # Поиск итоговой сводки - Energy deposit
            if 'Energy deposit' in line or 'Total energy deposit' in line:
                energy_match = re.search(self.ENERGY_DEPOSIT_PATTERN, line)
                if energy_match:
                    energy_val = float(energy_match.group(1))
                    energy_unit = energy_match.group(2)
                    self.summary['energy_deposit'] = self.convert_energy(energy_val, energy_unit)
                    if debug:
                        print(f"\n[DEBUG] Найдена Energy deposit: {energy_val} {energy_unit} = {self.summary['energy_deposit']} MeV")

            # Поиск Energy leakage
            if 'Energy leakage' in line:
                leakage_match = re.search(self.ENERGY_LEAKAGE_PATTERN, line)
                if leakage_match:
                    leakage_val = float(leakage_match.group(1))
                    leakage_unit = leakage_match.group(2)
                    self.summary['energy_leakage'] = self.convert_energy(leakage_val, leakage_unit)
                    if debug:
                        print(f"[DEBUG] Найдена Energy leakage: {leakage_val} {leakage_unit} = {self.summary['energy_leakage']} MeV")

            # Поиск Process calls frequency
            if 'Process calls frequency' in line or 'Process frequency' in line:
                in_summary = True
                continue

            if in_summary:
                # Парсинг строк с процессами: "CoulombScat= 1211" или "Transportation= 988"
                proc_matches = re.findall(self.PROCESS_FREQ_PATTERN, line)
                for proc_match in proc_matches:
                    process_name = proc_match[0]
                    process_count = int(proc_match[1])
                    process_frequencies[process_name] = process_count

        self.summary['process_frequencies'] = process_frequencies
        print(f"Извлечено шагов: {len(self.steps)}")
        print(f"Уникальных треков: {len(self.track_initial_energy)}")

        if debug and len(self.steps) > 0:
            print(f"\n[DEBUG] Первые 3 шага:")
            for i, step in enumerate(self.steps[:3]):
                print(f"  Шаг {i}: particle={step.particle}, track={step.track_id}, KineE={step.kine_e} MeV, dE={step.de_step} MeV, process={step.process}")

        if debug and len(self.steps) == 0:
            print(f"\n[DEBUG] Шаги не найдены! Примеры строк из файла:")
            for line_num, line_text in debug_lines_sample[:10]:
                print(f"  Строка {line_num}: {line_text[:100]}")

    def _parse_step_line_simple(self, line: str, line_num: int, debug: bool = False) -> Optional[StepData]:
        """Упрощенный парсинг строки шага по позициям/токенам"""
        line = line.strip()

        # Пропускаем пустые строки, разделители и заголовки
        if not line or line.startswith('---') or line.startswith('===') or 'Step#' in line:
            return None

        # Удаляем префикс потока, если есть
        if 'G4WT' in line or 'G4MT' in line:
            parts = line.split('>', 1)
            if len(parts) > 1:
                line = parts[1].strip()

        # Разбиваем строку на токены
        tokens = line.split()

        # Минимальное количество токенов для валидной строки шага
        if len(tokens) < 10:
            return None

        try:
            # Пытаемся найти Step# (должен быть числом)
            step_idx = -1
            for i, token in enumerate(tokens):
                if token.isdigit():
                    step_idx = i
                    break

            if step_idx == -1:
                return None

            step_num = int(tokens[step_idx])

            # Парсим координаты и другие числовые значения
            # Формат: Step# X unit Y unit Z unit KineE unit dE unit StepLen unit TrakLen unit Volume Process
            # Собираем пары (значение, единица)
            pairs = []
            i = step_idx + 1

            while i < len(tokens):
                try:
                    val = float(tokens[i])
                    unit = ""
                    # Проверяем следующий токен на единицу измерения
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if next_token in self.ENERGY_UNITS or next_token in self.LENGTH_UNITS:
                            unit = next_token
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                    pairs.append((val, unit))
                except (ValueError, IndexError):
                    i += 1
                    continue

            # Нужно минимум 7 пар: X, Y, Z, KineE, dE, StepLen, TrakLen
            if len(pairs) < 7:
                return None

            # Определяем поля по единицам измерения
            # Первые 3 должны быть длины (X, Y, Z)
            # Затем энергия (KineE)
            # Затем энергия (dE)
            # Затем 2 длины (StepLen, TrakLen)

            x_val, x_unit = pairs[0]
            y_val, y_unit = pairs[1]
            z_val, z_unit = pairs[2]

            # KineE - первое значение с единицей энергии после координат
            kine_e_val, kine_e_unit = pairs[3]
            de_val, de_unit = pairs[4]
            step_leng_val, step_leng_unit = pairs[5]
            trak_leng_val, trak_leng_unit = pairs[6]

            # Проверяем корректность единиц
            # X, Y, Z должны быть длины
            if x_unit and x_unit not in self.LENGTH_UNITS:
                return None
            if y_unit and y_unit not in self.LENGTH_UNITS:
                return None
            if z_unit and z_unit not in self.LENGTH_UNITS:
                return None

            # KineE и dE должны быть энергии
            if kine_e_unit and kine_e_unit not in self.ENERGY_UNITS:
                return None
            if de_unit and de_unit not in self.ENERGY_UNITS:
                return None

            # Устанавливаем единицы по умолчанию
            x_unit = x_unit if x_unit else "mm"
            y_unit = y_unit if y_unit else "mm"
            z_unit = z_unit if z_unit else "mm"
            kine_e_unit = kine_e_unit if kine_e_unit else "MeV"
            de_unit = de_unit if de_unit else "eV"
            step_leng_unit = step_leng_unit if step_leng_unit else "mm"
            trak_leng_unit = trak_leng_unit if trak_leng_unit else "mm"

            # Конвертация
            x = self.convert_length(x_val, x_unit)
            y = self.convert_length(y_val, y_unit)
            z = self.convert_length(z_val, z_unit)
            kine_e = self.convert_energy(kine_e_val, kine_e_unit)
            de_step = self.convert_energy(de_val, de_unit)
            step_leng = self.convert_length(step_leng_val, step_leng_unit)
            trak_leng = self.convert_length(trak_leng_val, trak_leng_unit)

            # Ищем Volume и Process (обычно последние 2 токена)
            volume = tokens[-2] if len(tokens) >= 2 else "Unknown"
            process = tokens[-1] if len(tokens) >= 1 else "Unknown"

            # Проверка на разумность значений
            if abs(de_step) > 1e6:  # Слишком большая потеря энергии
                if debug:
                    print(f"[WARNING] Строка {line_num}: подозрительно большое dE = {de_step} MeV")
                    print(f"  Исходная строка: {line[:100]}")
                return None

            return StepData(
                thread=self.current_thread,
                step_num=step_num,
                x=x, y=y, z=z,
                kine_e=kine_e,
                de_step=de_step,
                step_leng=step_leng,
                trak_leng=trak_leng,
                volume=volume,
                process=process,
                track_id=self.current_track_id,
                parent_id=self.current_parent_id,
                particle=self.current_particle
            )

        except (ValueError, IndexError) as e:
            if debug:
                print(f"[DEBUG] Не удалось распарсить строку {line_num}: {e}")
                print(f"  {line[:100]}")
            return None

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация данных в DataFrame"""
        if not self.steps:
            return pd.DataFrame()

        data = []
        for step in self.steps:
            data.append({
                'thread': step.thread,
                'step_num': step.step_num,
                'x': step.x,
                'y': step.y,
                'z': step.z,
                'kine_e': step.kine_e,
                'de_step': step.de_step,
                'step_leng': step.step_leng,
                'trak_leng': step.trak_leng,
                'volume': step.volume,
                'process': step.process,
                'track_id': step.track_id,
                'parent_id': step.parent_id,
                'particle': step.particle
            })

        return pd.DataFrame(data)


class Geant4Analyzer:
    """Класс для анализа и визуализации данных"""

    def __init__(self, df: pd.DataFrame, summary: Dict, output_dir: str, input_file: str):
        self.df = df
        self.summary = summary
        self.input_file = Path(input_file).name

        # Создаем выходную директорию на основе имени входного файла
        if output_dir == 'output':
            base_name = Path(input_file).stem
            self.output_dir = Path(f'output_{base_name}')
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_data(self) -> Dict[str, pd.DataFrame]:
        """Агрегация данных по различным параметрам"""
        results = {}

        # Агрегация по типам частиц
        if 'particle' in self.df.columns:
            results['by_particle'] = self.df.groupby('particle').agg({
                'de_step': ['sum', 'mean', 'std', 'count'],
                'kine_e': ['min', 'max', 'mean'],
                'step_leng': ['sum', 'mean']
            }).round(6)

        # Агрегация по процессам
        if 'process' in self.df.columns:
            results['by_process'] = self.df.groupby('process').agg({
                'de_step': ['sum', 'count'],
                'kine_e': 'mean'
            }).round(6)

        # Агрегация по потокам
        if 'thread' in self.df.columns:
            results['by_thread'] = self.df.groupby('thread').agg({
                'de_step': 'sum',
                'step_num': 'count'
            }).round(6)

        # Агрегация по трекам
        if 'track_id' in self.df.columns:
            results['by_track'] = self.df.groupby('track_id').agg({
                'de_step': 'sum',
                'trak_leng': 'max',
                'particle': 'first'
            }).round(6)

        return results

    def calculate_energy_balance(self, parser: Geant4LogParser) -> Dict:
        """
        Расчет энергетического баланса:
        1. Сумма начальных энергий всех треков
        2. Сумма конечных энергий всех треков
        3. Потерянная энергия = Начальная - Конечная
        4. Сумма dEStep из всех шагов (только положительные значения - реальные потери)

        ВАЖНО: В зависимости от verbose level, могут быть записаны не все шаги!
        """
        energy_balance = {}

        # Суммируем начальные энергии первичных частиц (parentID == 0)
        initial_energy_primary = 0
        for track_id, initial_e in parser.track_initial_energy.items():
            # Находим первый шаг этого трека, чтобы проверить parentID
            track_steps = self.df[self.df['track_id'] == track_id]
            if len(track_steps) > 0:
                parent_id = track_steps.iloc[0]['parent_id']
                if parent_id == 0:  # Первичная частица
                    initial_energy_primary += initial_e

        # Суммируем начальные энергии всех треков
        total_initial_energy = sum(parser.track_initial_energy.values())

        # Суммируем конечные энергии всех треков
        total_final_energy = sum(parser.track_final_energy.values())

        # Потерянная энергия
        energy_lost = total_initial_energy - total_final_energy

        # Сумма dEStep - ВАЖНО: берем только положительные значения
        # В Geant4 dEStep может быть отрицательным при создании частиц
        # Для расчета потерь энергии нужны только положительные значения
        total_de_step_all = self.df['de_step'].sum()
        total_de_step_positive = self.df[self.df['de_step'] > 0]['de_step'].sum()

        # Количество шагов с положительной и отрицательной dE
        n_positive = len(self.df[self.df['de_step'] > 0])
        n_negative = len(self.df[self.df['de_step'] < 0])
        n_zero = len(self.df[self.df['de_step'] == 0])

        # Определяем тип лога: полный или неполный (пропущены шаги)
        # Если средний dE > 0.001 MeV, скорее всего это неполный лог
        mean_de = total_de_step_positive / n_positive if n_positive > 0 else 0
        log_type = "incomplete" if mean_de > 0.001 else "complete"

        # Для неполного лога: оцениваем коэффициент пропуска шагов
        # Если dEStep очень большой, значит в одном записанном шаге - несколько реальных
        skip_factor = 1.0
        if log_type == "incomplete" and mean_de > 0.001:
            # Примерная оценка: нормальный dE для eIoni ~0.0001 MeV
            skip_factor = mean_de / 0.0001

        energy_balance['initial_energy_primary'] = initial_energy_primary
        energy_balance['total_initial_energy'] = total_initial_energy
        energy_balance['total_final_energy'] = total_final_energy
        energy_balance['energy_lost'] = energy_lost
        energy_balance['total_de_step_all'] = total_de_step_all
        energy_balance['total_de_step_positive'] = total_de_step_positive
        energy_balance['n_positive_de'] = n_positive
        energy_balance['n_negative_de'] = n_negative
        energy_balance['n_zero_de'] = n_zero
        energy_balance['mean_de'] = mean_de
        energy_balance['log_type'] = log_type
        energy_balance['estimated_skip_factor'] = skip_factor

        return energy_balance

    def verify_results(self, parser: Geant4LogParser) -> Dict[str, any]:
        """Сверка результатов парсинга с итоговой сводкой"""
        verification = {}

        # Проверка на аномальные значения
        print("\n[ПРОВЕРКА ДАННЫХ]")
        if len(self.df) > 0:
            de_stats = self.df['de_step'].describe()
            print(f"Статистика dE:")
            print(f"  Min:    {de_stats['min']:.6f} MeV")
            print(f"  Max:    {de_stats['max']:.6f} MeV")
            print(f"  Mean:   {de_stats['mean']:.6f} MeV")
            print(f"  Median: {self.df['de_step'].median():.6f} MeV")

            # Статистика по знаку dE
            n_positive = len(self.df[self.df['de_step'] > 0])
            n_negative = len(self.df[self.df['de_step'] < 0])
            n_zero = len(self.df[self.df['de_step'] == 0])
            print(f"\nРаспределение по знаку dE:")
            print(f"  Положительные (потери): {n_positive}")
            print(f"  Отрицательные (прирост): {n_negative}")
            print(f"  Нулевые: {n_zero}")

            # Проверка на подозрительные значения
            large_de = self.df[self.df['de_step'].abs() > 1000]
            if len(large_de) > 0:
                print(f"\n[ВНИМАНИЕ] Найдено {len(large_de)} шагов с |dE| > 1000 MeV!")
                print("Это может указывать на ошибку парсинга.")
                print("Примеры:")
                print(large_de[['particle', 'kine_e', 'de_step', 'process']].head())

        # Расчет энергетического баланса
        energy_balance = self.calculate_energy_balance(parser)

        # Сверка с итоговой сводкой
        energy_deposit_summary = self.summary.get('energy_deposit', 0)
        energy_leakage_summary = self.summary.get('energy_leakage', 0)

        # Используем положительные dEStep для сравнения с Energy deposit
        total_de_parsed_positive = energy_balance['total_de_step_positive']
        total_de_parsed_all = energy_balance['total_de_step_all']

        # Метод 2: Разница начальной и конечной энергии
        energy_deposited_calculated = energy_balance['energy_lost']

        verification['energy_balance'] = energy_balance
        verification['total_de_parsed_all'] = total_de_parsed_all
        verification['total_de_parsed_positive'] = total_de_parsed_positive
        verification['energy_deposited_calculated'] = energy_deposited_calculated
        verification['energy_deposit_summary'] = energy_deposit_summary
        verification['energy_leakage_summary'] = energy_leakage_summary

        # Разница между расчетами и сводкой (используем положительные dE)
        verification['absolute_difference_de'] = abs(total_de_parsed_positive - energy_deposit_summary)
        verification['absolute_difference_calc'] = abs(energy_deposited_calculated - energy_deposit_summary)

        if energy_deposit_summary > 0:
            verification['relative_difference_de'] = (
                abs(total_de_parsed_positive - energy_deposit_summary) / energy_deposit_summary * 100
            )
            verification['relative_difference_calc'] = (
                abs(energy_deposited_calculated - energy_deposit_summary) / energy_deposit_summary * 100
            )
        else:
            verification['relative_difference_de'] = 0
            verification['relative_difference_calc'] = 0

        # Сверка частот процессов
        process_counts_parsed = self.df['process'].value_counts().to_dict()
        process_freq_summary = self.summary.get('process_frequencies', {})

        verification['process_comparison'] = {}
        all_processes = set(list(process_counts_parsed.keys()) + list(process_freq_summary.keys()))

        # Исключаем мусорные "процессы" из итоговой статистики
        excluded_keywords = ['sumtot', 'counter', 'Range', 'simulation', 'end', 'rms', 'true', 'N=']

        for process in all_processes:
            # Пропускаем мусорные процессы
            if any(keyword in process for keyword in excluded_keywords):
                continue

            parsed = process_counts_parsed.get(process, 0)
            summary = process_freq_summary.get(process, 0)
            difference = parsed - summary

            verification['process_comparison'][process] = {
                'parsed': parsed,
                'summary': summary,
                'difference': difference,
                'abs_difference': abs(difference)
            }

        return verification

    def create_visualizations(self, save_formats: List[str] = ['png', 'svg']) -> None:
        """Создание всех визуализаций"""
        print("Создание визуализаций...")

        # 1. Гистограммы энергии по типам частиц
        self._plot_energy_distributions(save_formats)

        # 2. Boxplot/Violin plot потерь энергии
        self._plot_energy_loss_distribution(save_formats)

        # 3. Диаграмма частоты процессов
        self._plot_process_frequencies(save_formats)

        # 4. Heatmap координат
        self._plot_coordinate_heatmaps(save_formats)

        print(f"Визуализации сохранены в папку {self.output_dir}/")

    def _plot_energy_distributions(self, formats: List[str]) -> None:
        """Гистограммы распределения кинетической энергии"""
        particles = self.df['particle'].unique()

        for particle in particles:
            particle_data = self.df[self.df['particle'] == particle]['kine_e']

            if len(particle_data) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(particle_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

            # Статистика
            min_e = particle_data.min()
            max_e = particle_data.max()
            mean_e = particle_data.mean()

            ax.axvline(mean_e, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_e:.3f} MeV')

            ax.set_xlabel('Кинетическая энергия (MeV)', fontsize=12)
            ax.set_ylabel('Частота', fontsize=12)
            ax.set_title(f'Распределение энергии: {particle}\nMin: {min_e:.3f}, Max: {max_e:.3f}, Mean: {mean_e:.3f} MeV',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(False)

            for fmt in formats:
                plt.savefig(self.output_dir / f'energy_dist_{particle}.{fmt}',
                           dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_energy_loss_distribution(self, formats: List[str]) -> None:
        """Boxplot/Violin plot потерь энергии"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Boxplot
        particles = self.df['particle'].unique()
        data_for_box = [self.df[self.df['particle'] == p]['de_step'].values
                        for p in particles]

        bp = ax1.boxplot(data_for_box, labels=particles, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax1.set_ylabel('Потеря энергии (MeV)', fontsize=12)
        ax1.set_xlabel('Тип частицы', fontsize=12)
        ax1.set_title('Boxplot потерь энергии по типам частиц', fontsize=14, fontweight='bold')
        ax1.grid(False)

        # Violin plot
        data_for_violin = []
        labels_for_violin = []
        for p in particles:
            particle_de = self.df[self.df['particle'] == p]['de_step'].values
            if len(particle_de) > 0:
                data_for_violin.append(particle_de)
                labels_for_violin.append(p)

        if data_for_violin:
            parts = ax2.violinplot(data_for_violin, positions=range(len(labels_for_violin)),
                                   showmeans=True, showmedians=True)
            ax2.set_xticks(range(len(labels_for_violin)))
            ax2.set_xticklabels(labels_for_violin)
            ax2.set_ylabel('Потеря энергии (MeV)', fontsize=12)
            ax2.set_xlabel('Тип частицы', fontsize=12)
            ax2.set_title('Violin plot потерь энергии по типам частиц', fontsize=14, fontweight='bold')
            ax2.grid(False)

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(self.output_dir / f'energy_loss_distribution.{fmt}',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_process_frequencies(self, formats: List[str]) -> None:
        """Диаграмма частоты процессов"""
        process_counts = self.df['process'].value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(process_counts)))
        bars = ax.barh(range(len(process_counts)), process_counts.values, color=colors)

        ax.set_yticks(range(len(process_counts)))
        ax.set_yticklabels(process_counts.index, fontsize=10)
        ax.set_xlabel('Частота', fontsize=12)
        ax.set_title('Частота процессов (отсортировано по убыванию)', fontsize=14, fontweight='bold')
        ax.grid(False)

        # Добавляем значения на барах
        for i, (bar, value) in enumerate(zip(bars, process_counts.values)):
            ax.text(value, i, f' {value}', va='center', fontsize=9)

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(self.output_dir / f'process_frequencies.{fmt}',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_coordinate_heatmaps(self, formats: List[str]) -> None:
        """Heatmap плотности распределения координат"""
        if len(self.df) < 10:
            print("Недостаточно данных для построения heatmap координат")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # X-Y плоскость
        if len(self.df['x'].unique()) > 1 and len(self.df['y'].unique()) > 1:
            h, xedges, yedges = np.histogram2d(self.df['x'], self.df['y'], bins=50)
            im1 = axes[0, 0].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            axes[0, 0].set_xlabel('X (mm)', fontsize=12)
            axes[0, 0].set_ylabel('Y (mm)', fontsize=12)
            axes[0, 0].set_title('Плотность распределения: X-Y', fontsize=13, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0], label='Количество точек')

        # X-Z плоскость
        if len(self.df['x'].unique()) > 1 and len(self.df['z'].unique()) > 1:
            h, xedges, zedges = np.histogram2d(self.df['x'], self.df['z'], bins=50)
            im2 = axes[0, 1].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                                   extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])
            axes[0, 1].set_xlabel('X (mm)', fontsize=12)
            axes[0, 1].set_ylabel('Z (mm)', fontsize=12)
            axes[0, 1].set_title('Плотность распределения: X-Z', fontsize=13, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1], label='Количество точек')

        # Y-Z плоскость
        if len(self.df['y'].unique()) > 1 and len(self.df['z'].unique()) > 1:
            h, yedges, zedges = np.histogram2d(self.df['y'], self.df['z'], bins=50)
            im3 = axes[1, 0].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                                   extent=[yedges[0], yedges[-1], zedges[0], zedges[-1]])
            axes[1, 0].set_xlabel('Y (mm)', fontsize=12)
            axes[1, 0].set_ylabel('Z (mm)', fontsize=12)
            axes[1, 0].set_title('Плотность распределения: Y-Z', fontsize=13, fontweight='bold')
            plt.colorbar(im3, ax=axes[1, 0], label='Количество точек')

        # 3D scatter (проекция)
        scatter = axes[1, 1].scatter(self.df['x'], self.df['y'],
                                    c=self.df['z'], cmap='viridis',
                                    alpha=0.5, s=10)
        axes[1, 1].set_xlabel('X (mm)', fontsize=12)
        axes[1, 1].set_ylabel('Y (mm)', fontsize=12)
        axes[1, 1].set_title('Проекция координат (цвет = Z)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1], label='Z (mm)')

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(self.output_dir / f'coordinate_heatmaps.{fmt}',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def export_data(self, formats: List[str] = ['csv', 'xlsx']) -> None:
        """Экспорт данных в различные форматы"""
        print("Экспорт данных...")

        for fmt in formats:
            if fmt == 'csv':
                self.df.to_csv(self.output_dir / 'parsed_data.csv', index=False)
            elif fmt == 'xlsx':
                with pd.ExcelWriter(self.output_dir / 'parsed_data.xlsx', engine='openpyxl') as writer:
                    self.df.to_excel(writer, sheet_name='Steps', index=False)

                    # Агрегированные данные
                    agg_data = self.aggregate_data()
                    for name, df_agg in agg_data.items():
                        df_agg.to_excel(writer, sheet_name=name[:31])
            elif fmt == 'dat':
                # Простой текстовый формат
                with open(self.output_dir / 'parsed_data.dat', 'w') as f:
                    f.write(self.df.to_string())

        print(f"Данные экспортированы в форматы: {', '.join(formats)}")

    def generate_report(self, parser: Geant4LogParser) -> str:
        """Генерация текстового отчета"""
        verification = self.verify_results(parser)
        agg_data = self.aggregate_data()

        report = []
        report.append("=" * 100)
        report.append("ОТЧЕТ ПО АНАЛИЗУ ЛОГОВ GEANT4")
        report.append("=" * 100)
        report.append("")

        # Общая статистика
        report.append("1. ОБЩАЯ СТАТИСТИКА")
        report.append("-" * 100)
        report.append(f"Всего шагов: {len(self.df)}")
        report.append(f"Уникальных частиц: {self.df['particle'].nunique()}")
        report.append(f"Уникальных процессов: {self.df['process'].nunique()}")
        report.append(f"Уникальных треков: {self.df['track_id'].nunique()}")
        report.append("")

        # Агрегация по частицам
        if 'by_particle' in agg_data:
            report.append("2. СТАТИСТИКА ПО ТИПАМ ЧАСТИЦ")
            report.append("-" * 100)
            report.append(agg_data['by_particle'].to_string())
            report.append("")

        # Энергетический баланс
        report.append("3. ЭНЕРГЕТИЧЕСКИЙ БАЛАНС")
        report.append("-" * 100)
        energy_balance = verification['energy_balance']
        report.append(f"Начальная энергия первичных частиц: {energy_balance['initial_energy_primary']:.6f} MeV")
        report.append(f"Суммарная начальная энергия всех треков: {energy_balance['total_initial_energy']:.6f} MeV")
        report.append(f"Суммарная конечная энергия всех треков: {energy_balance['total_final_energy']:.6f} MeV")
        report.append(f"Потерянная энергия (Начальная - Конечная): {energy_balance['energy_lost']:.6f} MeV")
        report.append("")
        report.append(f"Анализ dEStep:")
        report.append(f"  Шагов с положительным dE (потери): {energy_balance['n_positive_de']}")
        report.append(f"  Шагов с отрицательным dE (прирост): {energy_balance['n_negative_de']}")
        report.append(f"  Шагов с нулевым dE: {energy_balance['n_zero_de']}")
        report.append(f"  Средний dE на шаг: {energy_balance['mean_de']:.6f} MeV")
        report.append(f"  Сумма всех dEStep: {energy_balance['total_de_step_all']:.6f} MeV")
        report.append(f"  Сумма положительных dEStep: {energy_balance['total_de_step_positive']:.6f} MeV")
        report.append("")

        # Тип лога
        log_type = energy_balance['log_type']
        report.append(f"Тип лога: {'НЕПОЛНЫЙ (пропущены шаги)' if log_type == 'incomplete' else 'ПОЛНЫЙ'}")
        if log_type == "incomplete":
            report.append(f"  ⚠️  ВНИМАНИЕ: Средний dE ({energy_balance['mean_de']:.6f} MeV) слишком велик!")
            report.append(f"  Это означает, что в логе записаны не все шаги.")
            report.append(f"  Оценка пропуска: ~{energy_balance['estimated_skip_factor']:.0f}x шагов")
            report.append(f"  Для неполных логов используйте ТОЛЬКО Метод 2 (энергетический баланс)!")
        else:
            report.append(f"  ✅ Лог содержит все или большинство шагов")
        report.append("")

        # Сверка результатов
        report.append("4. СВЕРКА С ИТОГОВОЙ СВОДКОЙ")
        report.append("-" * 100)
        report.append(f"Energy deposit (из сводки): {verification['energy_deposit_summary']:.6f} MeV")
        report.append(f"Energy leakage (из сводки): {verification.get('energy_leakage_summary', 0):.6f} MeV")
        report.append(f"Сумма (E_deposit + E_leakage): {verification['energy_deposit_summary'] + verification.get('energy_leakage_summary', 0):.6f} MeV")
        report.append("")

        log_type = energy_balance['log_type']

        if log_type == "incomplete":
            report.append("⚠️  ВНИМАНИЕ: Лог неполный (пропущены шаги)!")
            report.append("Метод 1 (сумма dEStep) НЕ РАБОТАЕТ для неполных логов.")
            report.append("Используйте ТОЛЬКО Метод 2 (энергетический баланс).")
            report.append("")

        report.append("Метод 1 (Сумма положительных dEStep - только потери энергии):")
        report.append(f"  Рассчитано: {verification['total_de_parsed_positive']:.6f} MeV")
        report.append(f"  Абсолютная разница: {verification['absolute_difference_de']:.6f} MeV")
        report.append(f"  Относительная разница: {verification['relative_difference_de']:.4f}%")
        if log_type == "incomplete":
            report.append(f"  ❌ НЕ ПРИМЕНИМО для этого лога (шаги пропущены)")
        report.append("")

        report.append("Метод 2 (Начальная - Конечная энергия):")
        report.append(f"  Рассчитано: {verification['energy_deposited_calculated']:.6f} MeV")
        report.append(f"  Абсолютная разница: {verification['absolute_difference_calc']:.6f} MeV")
        report.append(f"  Относительная разница: {verification['relative_difference_calc']:.4f}%")
        report.append("")

        # Проверка энергетического баланса первичных частиц
        primary_balance = energy_balance['initial_energy_primary']
        total_accounted = verification['energy_deposit_summary'] + verification.get('energy_leakage_summary', 0)
        balance_diff = abs(primary_balance - total_accounted)
        balance_rel = (balance_diff / primary_balance * 100) if primary_balance > 0 else 0

        report.append("Проверка энергетического баланса первичных частиц:")
        report.append(f"  Начальная энергия первичных: {primary_balance:.6f} MeV")
        report.append(f"  E_deposit + E_leakage:       {total_accounted:.6f} MeV")
        report.append(f"  Разница:                     {balance_diff:.6f} MeV ({balance_rel:.4f}%)")
        if balance_rel < 1:
            report.append(f"  ✅ ОТЛИЧНО! Энергетический баланс сошёлся!")
        elif balance_rel < 5:
            report.append(f"  ✅ ХОРОШО! Небольшая погрешность энергетического баланса")
        else:
            report.append(f"  ⚠️  Есть расхождение в энергетическом балансе")
        report.append("")

        report.append("Интерпретация:")
        if log_type == "incomplete":
            if balance_rel < 5:
                report.append("  ✅ Энергетический баланс первичных частиц сошёлся - симуляция корректна!")
                report.append("  Для неполных логов это главный критерий правильности.")
            else:
                report.append("  ⚠️  Есть расхождение в энергетическом балансе - проверьте лог")
        else:
            if verification['relative_difference_de'] < 5:
                report.append("  ✅ ОТЛИЧНО! Метод 1 показывает отличное совпадение (<5%)")
            elif verification['relative_difference_de'] < 20:
                report.append("  ✅ ХОРОШО! Метод 1 показывает хорошее совпадение (<20%)")
            elif verification['relative_difference_de'] < 30:
                report.append("  ⚠️  ПРИЕМЛЕМО. Метод 1 в пределах допустимого (<30%)")
            else:
                report.append("  ❌ ВНИМАНИЕ! Метод 1 показывает большое расхождение (>30%)")
        report.append("")

        # Объяснение расхождений
        report.append("5. ВОЗМОЖНЫЕ ПРИЧИНЫ РАСХОЖДЕНИЙ")
        report.append("-" * 100)
        report.append("a) Неполное логирование:")
        report.append("   - Не все шаги могут быть записаны в лог (зависит от verbose level)")
        report.append("   - Некоторые процессы могут не записывать dEStep явно")
        report.append("")
        report.append("b) Вторичные частицы:")
        report.append("   - Энергия может передаваться вторичным частицам (электроны, фотоны и т.д.)")
        report.append("   - Эти вторичные частицы могут депонировать энергию в других местах")
        report.append("")
        report.append("c) Энергетические пороги:")
        report.append("   - Частицы с энергией ниже порога могут не отслеживаться явно")
        report.append("   - Их энергия депонируется локально без явного логирования")
        report.append("")
        report.append("d) Границы детектора:")
        report.append("   - Частицы, покидающие объем (OutOfWorld), уносят энергию")
        report.append("   - Эта энергия учитывается в Energy leakage, а не в Energy deposit")
        report.append("")
        report.append("e) Округления и конверсия единиц:")
        report.append("   - Погрешности при конверсии между eV, keV, MeV")
        report.append("   - Накопление ошибок округления при большом количестве шагов")
        report.append("")
        report.append("f) Знак dEStep в Geant4:")
        report.append("   - dEStep может быть отрицательным при создании частиц")
        report.append("   - Для расчета Energy deposit используем только положительные значения")
        report.append("")

        # Сравнение процессов
        if verification['process_comparison']:
            report.append("6. СРАВНЕНИЕ ЧАСТОТ ПРОЦЕССОВ")
            report.append("-" * 100)
            report.append(f"{'Процесс':<30s} | {'Парсинг':>10s} | {'Сводка':>10s} | {'Разница':>10s}")
            report.append("-" * 100)

            sorted_processes = sorted(verification['process_comparison'].items(),
                                    key=lambda x: x[1]['abs_difference'], reverse=True)

            for process, data in sorted_processes:
                diff_str = f"{data['difference']:+d}"  # + или - перед числом
                report.append(f"{process:<30s} | {data['parsed']:>10d} | "
                            f"{data['summary']:>10d} | {diff_str:>10s}")

            report.append("")
            report.append("Анализ расхождений в процессах:")
            report.append("  Положительная разница: парсинг насчитал больше вызовов, чем в сводке")
            report.append("  Отрицательная разница: в сводке указано больше вызовов, чем найдено при парсинге")
            report.append("  Возможные причины:")
            report.append("    - Неполное логирование шагов (не все процессы записываются)")
            report.append("    - Разные способы подсчета (например, Transportation может считаться иначе)")
            report.append("    - Некоторые процессы могут выполняться без явного шага")
            report.append("    - initStep и OutOfWorld - служебные процессы Geant4")
            report.append("")

        report.append("=" * 100)

        return "\n".join(report)

    def save_report(self, parser: Geant4LogParser, filename: str = "analysis_report.txt") -> None:
        """Сохранение отчета в файл"""
        report = self.generate_report(parser)
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчет сохранен: {self.output_dir / filename}")


def main():
    """Основная функция с CLI"""
    parser = argparse.ArgumentParser(
        description='Парсер и анализатор логов Geant4 (улучшенная версия)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python geant4_parser_improved.py -i simulation.log
  python geant4_parser_improved.py -i simulation.log -o results --export csv xlsx --plot png svg
  python geant4_parser_improved.py -i simulation.log --no-viz --debug
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Входной файл лога Geant4')
    parser.add_argument('-o', '--output', default='output',
                       help='Папка для выходных файлов (по умолчанию: output)')
    parser.add_argument('--export', nargs='+', default=['csv', 'xlsx'],
                       choices=['csv', 'xlsx', 'dat'],
                       help='Форматы экспорта данных')
    parser.add_argument('--plot', nargs='+', default=['png', 'svg'],
                       choices=['png', 'svg'],
                       help='Форматы сохранения графиков')
    parser.add_argument('--no-viz', action='store_true',
                       help='Не создавать визуализации')
    parser.add_argument('--debug', action='store_true',
                       help='Режим отладки с детальным выводом')

    args = parser.parse_args()

    # Парсинг лога
    print("=" * 100)
    print("GEANT4 LOG PARSER - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("=" * 100)

    parser_obj = Geant4LogParser(args.input)
    parser_obj.parse_log(debug=args.debug)

    if not parser_obj.steps:
        print("ПРЕДУПРЕЖДЕНИЕ: Не найдено данных о шагах в логе!")
        print("Возможно, формат лога не соответствует ожидаемому.")
        print("\nПопробуйте запустить с флагом --debug для диагностики:")
        print(f"  python geant4_parser_improved.py -i {args.input} --debug")
        return

    # Конвертация в DataFrame
    df = parser_obj.to_dataframe()
    print(f"\nСоздан DataFrame с {len(df)} записями")
    print(f"Колонки: {', '.join(df.columns)}")

    # Создание анализатора с указанием имени входного файла
    analyzer = Geant4Analyzer(df, parser_obj.summary, args.output, args.input)

    # Агрегация данных
    print("\nАгрегация данных...")
    agg_results = analyzer.aggregate_data()

    # Экспорт данных
    analyzer.export_data(formats=args.export)

    # Визуализация
    if not args.no_viz:
        analyzer.create_visualizations(save_formats=args.plot)

    # Генерация отчета
    analyzer.save_report(parser_obj)

    # Вывод основной статистики
    print("\n" + "=" * 100)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 100)
    verification = analyzer.verify_results(parser_obj)

    energy_balance = verification['energy_balance']
    print(f"\nЭнергетический баланс:")
    print(f"  Начальная энергия первичных частиц: {energy_balance['initial_energy_primary']:.6f} MeV")
    print(f"  Потерянная энергия (расчет): {energy_balance['energy_lost']:.6f} MeV")
    print(f"  Сумма положительных dEStep: {energy_balance['total_de_step_positive']:.6f} MeV")
    print(f"  Сумма всех dEStep: {energy_balance['total_de_step_all']:.6f} MeV")

    print(f"\nСверка с итоговой сводкой:")
    print(f"  Energy deposit (сводка): {verification['energy_deposit_summary']:.6f} MeV")
    print(f"  Относительная разница (метод dEStep): {verification['relative_difference_de']:.4f}%")
    print(f"  Относительная разница (метод баланса): {verification['relative_difference_calc']:.4f}%")

    print(f"\nСверка процессов:")
    total_diff = sum(abs(v['difference']) for v in verification['process_comparison'].values())
    print(f"  Суммарная абсолютная разница: {total_diff}")

    print("=" * 100)
    print(f"\nВсе результаты сохранены в папке: {analyzer.output_dir}")


if __name__ == "__main__":
    main()