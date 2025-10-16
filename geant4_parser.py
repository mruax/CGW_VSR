"""
Geant4 Log Parser
Программа для парсинга, анализа и визуализации логов симуляции Geant4
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
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

    # Паттерн для строки шага (типичный формат verbose stepping)
    STEP_PATTERN = r'(\d+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)\s+(\S+)\s+(\w+)'

    # Паттерны для итоговой сводки
    ENERGY_DEPOSIT_PATTERN = r'Energy deposit[:\s]+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\w+)'
    PROCESS_FREQ_PATTERN = r'(\w+)\s*:\s*(\d+)'

    # Конверсия единиц в MeV и mm
    ENERGY_UNITS = {'eV': 1e-6, 'keV': 1e-3, 'MeV': 1.0, 'GeV': 1e3, 'TeV': 1e6}
    LENGTH_UNITS = {'fm': 1e-12, 'nm': 1e-6, 'um': 1e-3, 'mm': 1.0, 'cm': 10.0, 'm': 1e3, 'km': 1e6}

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.steps: List[StepData] = []
        self.summary: Dict = {}
        self.current_thread = ""
        self.current_track_id = 0
        self.current_parent_id = 0
        self.current_particle = ""

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

            # Определение начала таблицы шагов
            if 'Step#' in line and any(x in line for x in ['KineE', 'dE', 'StepLen']):
                step_header_found = True
                in_step_table = True
                if debug:
                    print(f"\n[DEBUG] Найден заголовок таблицы шагов на строке {i}:")
                    print(f"  {line.strip()}")
                continue

            # Парсинг строк данных шагов (только если нашли заголовок)
            if in_step_table and step_header_found:
                # Простой парсинг: разделение по пробелам
                step_data = self._parse_step_line_simple(line, i, debug)
                if step_data:
                    self.steps.append(step_data)
                elif line.strip() == '' or line.startswith('---') or line.startswith('==='):
                    in_step_table = False  # Конец таблицы

            # Поиск итоговой сводки
            if 'Energy deposit' in line or 'Total energy deposit' in line:
                energy_match = re.search(self.ENERGY_DEPOSIT_PATTERN, line)
                if energy_match:
                    energy_val = float(energy_match.group(1))
                    energy_unit = energy_match.group(2)
                    self.summary['energy_deposit'] = self.convert_energy(energy_val, energy_unit)
                    if debug:
                        print(f"\n[DEBUG] Найдена Energy deposit: {energy_val} {energy_unit} = {self.summary['energy_deposit']} MeV")

            if 'Process calls frequency' in line or 'Process frequency' in line:
                in_summary = True
                continue

            if in_summary:
                proc_match = re.search(self.PROCESS_FREQ_PATTERN, line)
                if proc_match:
                    process_frequencies[proc_match.group(1)] = int(proc_match.group(2))

        self.summary['process_frequencies'] = process_frequencies
        print(f"Извлечено шагов: {len(self.steps)}")

        if debug and len(self.steps) > 0:
            print(f"\n[DEBUG] Первые 3 шага:")
            for i, step in enumerate(self.steps[:3]):
                print(f"  Шаг {i}: particle={step.particle}, KineE={step.kine_e} MeV, dE={step.de_step} MeV, process={step.process}")

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
        if 'G4WT' in line:
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
            # Формат обычно: Step# X unit Y unit Z unit KineE unit dE unit StepLen unit TrakLen unit Volume Process
            values = []
            units = []

            i = step_idx + 1
            while i < len(tokens) and len(values) < 7:  # Нужно 7 значений: X,Y,Z, KineE, dE, StepLen, TrakLen
                try:
                    val = float(tokens[i])
                    values.append(val)
                    # Следующий токен может быть единицей измерения
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        # Проверяем, является ли следующий токен единицей измерения
                        if next_token in self.ENERGY_UNITS or next_token in self.LENGTH_UNITS:
                            units.append(next_token)
                            i += 2
                        else:
                            units.append("")
                            i += 1
                    else:
                        units.append("")
                        i += 1
                except (ValueError, IndexError):
                    i += 1
                    continue

            # Проверяем, что получили достаточно значений
            if len(values) < 7:
                return None

            # Извлекаем значения
            x_val, y_val, z_val, kine_e_val, de_val, step_leng_val, trak_leng_val = values[:7]

            # Определяем единицы (по умолчанию mm для длин, MeV для энергий)
            x_unit = units[0] if len(units) > 0 and units[0] in self.LENGTH_UNITS else "mm"
            y_unit = units[1] if len(units) > 1 and units[1] in self.LENGTH_UNITS else "mm"
            z_unit = units[2] if len(units) > 2 and units[2] in self.LENGTH_UNITS else "mm"
            kine_e_unit = units[3] if len(units) > 3 and units[3] in self.ENERGY_UNITS else "MeV"
            de_unit = units[4] if len(units) > 4 and units[4] in self.ENERGY_UNITS else "MeV"
            step_leng_unit = units[5] if len(units) > 5 and units[5] in self.LENGTH_UNITS else "mm"
            trak_leng_unit = units[6] if len(units) > 6 and units[6] in self.LENGTH_UNITS else "mm"

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

    def _parse_step_line(self, match) -> Optional[StepData]:
        """Парсинг одной строки шага"""
        groups = match.groups()

        # Извлечение данных с учетом возможных вариантов формата
        step_num = int(groups[0])

        # X, Y, Z с единицами
        x_val = float(groups[1])
        x_unit = groups[2] if len(groups) > 2 else "mm"
        y_val = float(groups[3])
        y_unit = groups[4] if len(groups) > 4 else "mm"
        z_val = float(groups[5])
        z_unit = groups[6] if len(groups) > 6 else "mm"

        # Конвертация координат в mm
        x = self.convert_length(x_val, x_unit)
        y = self.convert_length(y_val, y_unit)
        z = self.convert_length(z_val, z_unit)

        # Кинетическая энергия
        kine_e_val = float(groups[7])
        kine_e_unit = groups[8] if len(groups) > 8 else "MeV"
        kine_e = self.convert_energy(kine_e_val, kine_e_unit)

        # Потеря энергии
        de_val = float(groups[9])
        de_unit = groups[10] if len(groups) > 10 else "MeV"
        de_step = self.convert_energy(de_val, de_unit)

        # Длины
        step_leng_val = float(groups[11])
        step_leng_unit = groups[12] if len(groups) > 12 else "mm"
        step_leng = self.convert_length(step_leng_val, step_leng_unit)

        trak_leng_val = float(groups[13])
        trak_leng_unit = groups[14] if len(groups) > 14 else "mm"
        trak_leng = self.convert_length(trak_leng_val, trak_leng_unit)

        volume = groups[15] if len(groups) > 15 else "Unknown"
        process = groups[16] if len(groups) > 16 else "Unknown"

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
    """Анализатор данных Geant4"""

    def __init__(self, df: pd.DataFrame, summary: Dict, output_base: str = "output", input_filename: str = ""):
        self.df = df
        self.summary = summary
        # Создаем подпапку с именем входного файла
        if input_filename:
            file_stem = Path(input_filename).stem  # Имя файла без расширения
            self.output_dir = Path(output_base) / file_stem
        else:
            self.output_dir = Path(output_base)
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

    def verify_results(self) -> Dict[str, any]:
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

            # Проверка на подозрительные значения
            large_de = self.df[self.df['de_step'].abs() > 1000]
            if len(large_de) > 0:
                print(f"\n[ВНИМАНИЕ] Найдено {len(large_de)} шагов с |dE| > 1000 MeV!")
                print("Это может указывать на ошибку парсинга.")
                print("Примеры:")
                print(large_de[['particle', 'kine_e', 'de_step', 'process']].head())

        # Сверка суммы dE
        total_de_parsed = self.df['de_step'].sum()
        energy_deposit_summary = self.summary.get('energy_deposit', 0)

        verification['total_de_parsed'] = total_de_parsed
        verification['energy_deposit_summary'] = energy_deposit_summary
        verification['absolute_difference'] = abs(total_de_parsed - energy_deposit_summary)

        if energy_deposit_summary > 0:
            verification['relative_difference'] = (
                abs(total_de_parsed - energy_deposit_summary) / energy_deposit_summary * 100
            )
        else:
            verification['relative_difference'] = 0

        # Сверка частот процессов
        process_counts_parsed = self.df['process'].value_counts().to_dict()
        process_freq_summary = self.summary.get('process_frequencies', {})

        verification['process_comparison'] = {}
        for process in set(list(process_counts_parsed.keys()) + list(process_freq_summary.keys())):
            parsed = process_counts_parsed.get(process, 0)
            summary = process_freq_summary.get(process, 0)
            verification['process_comparison'][process] = {
                'parsed': parsed,
                'summary': summary,
                'difference': abs(parsed - summary)
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

        print("Визуализации сохранены в папку output/")

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
        ax1.tick_params(axis='x', rotation=45)

        # Violin plot
        df_plot = self.df[['particle', 'de_step']].copy()
        sns.violinplot(data=df_plot, x='particle', y='de_step', ax=ax2, palette='Set2')
        ax2.set_ylabel('Потеря энергии (MeV)', fontsize=12)
        ax2.set_xlabel('Тип частицы', fontsize=12)
        ax2.set_title('Violin plot потерь энергии по типам частиц', fontsize=14, fontweight='bold')
        ax2.grid(False)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(self.output_dir / f'energy_loss_distribution.{fmt}',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_process_frequencies(self, formats: List[str]) -> None:
        """Диаграмма частоты процессов"""
        process_counts = self.df['process'].value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(process_counts)))
        bars = ax.barh(range(len(process_counts)), process_counts.values, color=colors)

        ax.set_yticks(range(len(process_counts)))
        ax.set_yticklabels(process_counts.index)
        ax.set_xlabel('Частота', fontsize=12)
        ax.set_ylabel('Процесс', fontsize=12)
        ax.set_title('Частота процессов (отсортировано по убыванию)',
                    fontsize=14, fontweight='bold')
        ax.grid(False)

        # Добавление значений на столбцы
        for i, (bar, count) in enumerate(zip(bars, process_counts.values)):
            ax.text(count + max(process_counts.values) * 0.01, i,
                   f'{int(count)}', va='center', fontsize=9)

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(self.output_dir / f'process_frequencies.{fmt}',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_coordinate_heatmaps(self, formats: List[str]) -> None:
        """Heatmap координат X/Y/Z"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # XY heatmap
        if len(self.df) > 0:
            h1, xedges, yedges = np.histogram2d(self.df['x'], self.df['y'], bins=50)
            im1 = axes[0, 0].imshow(h1.T, origin='lower', aspect='auto',
                                   cmap='YlOrRd', interpolation='bilinear',
                                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            axes[0, 0].set_xlabel('X (mm)', fontsize=12)
            axes[0, 0].set_ylabel('Y (mm)', fontsize=12)
            axes[0, 0].set_title('Плотность распределения: X-Y', fontsize=13, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0], label='Количество точек')

            # XZ heatmap
            h2, xedges, zedges = np.histogram2d(self.df['x'], self.df['z'], bins=50)
            im2 = axes[0, 1].imshow(h2.T, origin='lower', aspect='auto',
                                   cmap='YlOrRd', interpolation='bilinear',
                                   extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])
            axes[0, 1].set_xlabel('X (mm)', fontsize=12)
            axes[0, 1].set_ylabel('Z (mm)', fontsize=12)
            axes[0, 1].set_title('Плотность распределения: X-Z', fontsize=13, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1], label='Количество точек')

            # YZ heatmap
            h3, yedges, zedges = np.histogram2d(self.df['y'], self.df['z'], bins=50)
            im3 = axes[1, 0].imshow(h3.T, origin='lower', aspect='auto',
                                   cmap='YlOrRd', interpolation='bilinear',
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

    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        verification = self.verify_results()
        agg_data = self.aggregate_data()

        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ПО АНАЛИЗУ ЛОГОВ GEANT4")
        report.append("=" * 80)
        report.append("")

        # Общая статистика
        report.append("1. ОБЩАЯ СТАТИСТИКА")
        report.append("-" * 80)
        report.append(f"Всего шагов: {len(self.df)}")
        report.append(f"Уникальных частиц: {self.df['particle'].nunique()}")
        report.append(f"Уникальных процессов: {self.df['process'].nunique()}")
        report.append(f"Уникальных треков: {self.df['track_id'].nunique()}")
        report.append("")

        # Агрегация по частицам
        if 'by_particle' in agg_data:
            report.append("2. СТАТИСТИКА ПО ТИПАМ ЧАСТИЦ")
            report.append("-" * 80)
            report.append(agg_data['by_particle'].to_string())
            report.append("")

        # Сверка результатов
        report.append("3. СВЕРКА РЕЗУЛЬТАТОВ")
        report.append("-" * 80)
        report.append(f"Сумма dE (парсинг): {verification['total_de_parsed']:.6f} MeV")
        report.append(f"Energy deposit (сводка): {verification['energy_deposit_summary']:.6f} MeV")
        report.append(f"Абсолютная разница: {verification['absolute_difference']:.6f} MeV")
        report.append(f"Относительная разница: {verification['relative_difference']:.4f}%")
        report.append("")

        # Объяснение расхождений
        report.append("4. ВОЗМОЖНЫЕ ПРИЧИНЫ РАСХОЖДЕНИЙ")
        report.append("-" * 80)
        report.append("- Неполный парсинг некоторых строк лога")
        report.append("- Различия в учете граничных условий")
        report.append("- Округления при конвертации единиц измерения")
        report.append("- Частичное логирование шагов в verbose режиме")
        report.append("- Энергия, потерянная в процессах без явного логирования шагов")
        report.append("")

        # Сравнение процессов
        if verification['process_comparison']:
            report.append("5. СРАВНЕНИЕ ЧАСТОТ ПРОЦЕССОВ")
            report.append("-" * 80)
            for process, data in sorted(verification['process_comparison'].items(),
                                       key=lambda x: x[1]['parsed'], reverse=True):
                report.append(f"{process:30s} | Парсинг: {data['parsed']:8d} | "
                            f"Сводка: {data['summary']:8d} | Разница: {data['difference']:8d}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, filename: str = "analysis_report.txt") -> None:
        """Сохранение отчета в файл"""
        report = self.generate_report()
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчет сохранен: {self.output_dir / filename}")


def main():
    """Основная функция с CLI"""
    parser = argparse.ArgumentParser(
        description='Парсер и анализатор логов Geant4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python geant4_parser.py -i simulation.log
  python geant4_parser.py -i simulation.log -o results --export csv xlsx --plot png svg
  python geant4_parser.py -i simulation.log --no-viz
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
    parser.add_argument('--workers', type=int, default=1,
                       help='Количество потоков для обработки (резерв для будущего)')
    parser.add_argument('--debug', action='store_true',
                       help='Режим отладки с детальным выводом')

    args = parser.parse_args()

    # Парсинг лога
    print("=" * 80)
    print("GEANT4 LOG PARSER")
    print("=" * 80)

    parser_obj = Geant4LogParser(args.input)
    parser_obj.parse_log(debug=args.debug)

    if not parser_obj.steps:
        print("ПРЕДУПРЕЖДЕНИЕ: Не найдено данных о шагах в логе!")
        print("Возможно, формат лога не соответствует ожидаемому.")
        print("\nПопробуйте запустить с флагом --debug для диагностики:")
        print(f"  python geant4_parser.py -i {args.input} --debug")
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
    analyzer.save_report()

    # Вывод основной статистики
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    verification = analyzer.verify_results()
    print(f"Сумма dE (парсинг): {verification['total_de_parsed']:.6f} MeV")
    print(f"Energy deposit (сводка): {verification['energy_deposit_summary']:.6f} MeV")
    print(f"Относительная разница: {verification['relative_difference']:.4f}%")
    print("=" * 80)
    print(f"\nВсе результаты сохранены в папке: {analyzer.output_dir}")


if __name__ == "__main__":
    main()