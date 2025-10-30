"""
Geant4 Log Parser - GUI Interface
Графический интерфейс для парсинга, анализа и визуализации логов симуляции Geant4
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import subprocess
import os

import datetime
import sys

warnings.filterwarnings('ignore')


def setup_log(log_filename="geant4_log"):
    log_dir = Path("output") / f"{log_filename}_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.txt"
    sys.stdout = open(log_path, "w", encoding="utf-8")
    sys.stderr = sys.stdout
    return log_path


# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 9


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

    @property
    def is_primary(self) -> bool:
        """Проверка, является ли частица первичной (Parent ID = 0)"""
        return self.parent_id == 0


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
        self.track_parent_ids: Dict[int, int] = {}

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
                self.track_parent_ids[self.current_track_id] = self.current_parent_id

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
                step_data = self._parse_step_line_improved(line, i, debug)
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
                        print(
                            f"\n[DEBUG] Найдена Energy deposit: {energy_val} {energy_unit} = {self.summary['energy_deposit']} MeV")

            # Поиск Energy leakage
            if 'Energy leakage' in line:
                leakage_match = re.search(self.ENERGY_LEAKAGE_PATTERN, line)
                if leakage_match:
                    leakage_val = float(leakage_match.group(1))
                    leakage_unit = leakage_match.group(2)
                    self.summary['energy_leakage'] = self.convert_energy(leakage_val, leakage_unit)
                    if debug:
                        print(
                            f"[DEBUG] Найдена Energy leakage: {leakage_val} {leakage_unit} = {self.summary['energy_leakage']} MeV")

            # Поиск частот процессов
            if in_summary and '=' in line and any(c.isdigit() for c in line):
                proc_match = re.search(self.PROCESS_FREQ_PATTERN, line)
                if proc_match:
                    process_name = proc_match.group(1)
                    frequency = int(proc_match.group(2))
                    process_frequencies[process_name] = frequency

            # Определение начала сводной секции
            if 'Number of process calls' in line or 'processes count' in line:
                in_summary = True

        # Сохранение частот процессов
        if process_frequencies:
            self.summary['process_frequencies'] = process_frequencies

        print(f"Парсинг завершен: найдено {len(self.steps)} шагов")
        if self.summary:
            print(f"Найдена итоговая сводка с {len(self.summary)} параметрами")

        # Подсчет первичных и вторичных частиц
        if self.steps:
            primary_steps = sum(1 for s in self.steps if s.is_primary)
            secondary_steps = len(self.steps) - primary_steps
            print(f"  - Шаги первичных частиц (Parent ID = 0): {primary_steps}")
            print(f"  - Шаги вторичных частиц (Parent ID > 0): {secondary_steps}")
            print(f"  - Уникальных треков: {len(set(s.track_id for s in self.steps))}")
            print(f"  - Уникальных потоков: {len(set(s.thread for s in self.steps))}")
            print(f"  - Типы частиц: {', '.join(set(s.particle for s in self.steps))}")

    def _parse_step_line_improved(self, line: str, line_num: int, debug: bool = False) -> Optional[StepData]:
        """
        УЛУЧШЕННЫЙ парсинг строки шага с правильной обработкой dEStep

        Формат строки:
        G4WT26 >     1     2.7 mm   2.694 mm   3.605 mm   733.5 keV      223.2 keV  488.6 um   488.6 um       MAPI       eIoni
        """
        if not line.strip() or 'Step#' in line:
            return None

        # Проверка наличия префикса потока
        if not re.search(self.THREAD_PATTERN, line):
            return None

        parts = line.split()
        if len(parts) < 8:
            return None

        try:
            # Находим номер шага
            step_num_idx = None
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:  # пропускаем G4WT
                    step_num_idx = i
                    break

            if step_num_idx is None:
                return None

            step_num = int(parts[step_num_idx])

            # === ПАРСИНГ КООРДИНАТ ===
            coords = []
            coord_units = []
            i = step_num_idx + 1
            while len(coords) < 3 and i < len(parts) - 1:
                try:
                    val = float(parts[i])
                    unit = parts[i + 1] if i + 1 < len(parts) else 'mm'
                    # Проверяем, что единица - это единица длины
                    if unit in self.LENGTH_UNITS or unit in ['mm', 'um', 'fm', 'cm', 'm', 'nm']:
                        coords.append(self.convert_length(val, unit))
                        coord_units.append(unit)
                        i += 2
                    else:
                        i += 1
                except (ValueError, IndexError):
                    i += 1

            if len(coords) < 3:
                return None

            x, y, z = coords[0], coords[1], coords[2]

            # === ПАРСИНГ ЭНЕРГИЙ ===
            # Находим все пары число+единица энергии
            energy_values = []
            i = step_num_idx + 7  # начинаем после координат

            while i < min(len(parts) - 1, step_num_idx + 25):
                try:
                    val = float(parts[i])
                    unit = parts[i + 1] if i + 1 < len(parts) else ''

                    # Проверяем, является ли следующий элемент единицей энергии
                    if unit in self.ENERGY_UNITS or unit in ['eV', 'keV', 'MeV', 'GeV', 'TeV', 'meV']:
                        energy_mev = self.convert_energy(val, unit)
                        energy_values.append((i, energy_mev, val, unit))
                        i += 2  # пропускаем значение и единицу
                    else:
                        i += 1
                except (ValueError, IndexError):
                    i += 1

            # Первое значение энергии - это KineE
            # Второе значение энергии - это dEStep
            kine_e = energy_values[0][1] if len(energy_values) >= 1 else 0.0
            de_step = energy_values[1][1] if len(energy_values) >= 2 else 0.0

            if debug and de_step > 0:
                print(
                    f"[DEBUG] Найдено dEStep на строке {line_num}: {energy_values[1][2]} {energy_values[1][3]} = {de_step:.6f} MeV")

            # === ПАРСИНГ ДЛИН ===
            # Находим все пары число+единица длины (после энергий)
            length_values = []
            start_idx = energy_values[-1][0] + 2 if energy_values else step_num_idx + 10

            for i in range(start_idx, min(len(parts) - 1, len(parts) - 3)):
                try:
                    val = float(parts[i])
                    unit = parts[i + 1] if i + 1 < len(parts) else ''

                    if unit in self.LENGTH_UNITS:
                        length_mm = self.convert_length(val, unit)
                        length_values.append(length_mm)

                        if len(length_values) >= 2:
                            break
                except (ValueError, IndexError):
                    continue

            step_leng = length_values[0] if len(length_values) >= 1 else 0.0
            trak_leng = length_values[1] if len(length_values) >= 2 else 0.0

            # === ПАРСИНГ VOLUME И PROCESS ===
            volume = "Unknown"
            process = "Unknown"

            # Process обычно последний элемент
            if len(parts) > 0:
                process = parts[-1]

            # Volume обычно предпоследний
            if len(parts) > 1:
                volume = parts[-2]

            # Создаем объект StepData
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
                particle=self.current_particle,
                coord_unit="mm",
                energy_unit="MeV",
                length_unit="mm"
            )

        except Exception as e:
            if debug:
                print(f"[DEBUG] Ошибка парсинга строки {line_num}: {e}")
                print(f"  Строка: {line.strip()}")
            return None

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация списка шагов в pandas DataFrame"""
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
                'particle': step.particle,
                'is_primary': step.is_primary
            })

        return pd.DataFrame(data)


class Geant4Analyzer:
    """Анализатор данных Geant4"""

    def __init__(self, df: pd.DataFrame, summary: Dict, output_dir: str = 'output', input_filename: str = ''):
        self.df_all = df
        self.df = df  # для обратной совместимости
        self.summary = summary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.input_filename = Path(input_filename).stem if input_filename else 'analysis'

        # Разделение на первичные и вторичные частицы
        self.df_primary = df[df['is_primary'] == True].copy()
        self.df_secondary = df[df['is_primary'] == False].copy()

        print(f"\nСтатистика данных:")
        print(f"  Всего шагов: {len(df)}")
        print(f"  Первичные частицы (Parent ID = 0): {len(self.df_primary)}")
        print(f"  Вторичные частицы (Parent ID > 0): {len(self.df_secondary)}")

    def aggregate_data(self) -> Dict:
        """Агрегация данных по процессам и частицам"""
        results = {}

        # Агрегация по процессам
        process_stats = self.df.groupby('process').agg({
            'de_step': ['sum', 'mean', 'count'],
            'step_leng': ['sum', 'mean'],
            'kine_e': 'mean'
        }).round(6)
        results['process_stats'] = process_stats

        # Агрегация по частицам
        particle_stats = self.df.groupby('particle').agg({
            'de_step': ['sum', 'mean', 'count'],
            'kine_e': ['mean', 'max'],
            'step_leng': 'sum'
        }).round(6)
        results['particle_stats'] = particle_stats

        # Агрегация первичных частиц
        if len(self.df_primary) > 0:
            primary_stats = self.df_primary.groupby('particle').agg({
                'de_step': ['sum', 'mean', 'count'],
                'kine_e': ['mean', 'max'],
                'track_id': 'nunique'
            }).round(6)
            results['primary_stats'] = primary_stats

        # Агрегация вторичных частиц
        if len(self.df_secondary) > 0:
            secondary_stats = self.df_secondary.groupby('particle').agg({
                'de_step': ['sum', 'mean', 'count'],
                'kine_e': ['mean', 'max'],
                'track_id': 'nunique'
            }).round(6)
            results['secondary_stats'] = secondary_stats

        return results

    def export_data(self, formats: List[str] = ['xlsx']) -> None:
        """Экспорт данных в различные форматы с разбиением >1 млн строк"""
        print("\nЭкспорт данных...")

        MAX_ROWS = 1_000_000
        datasets = [
            (self.df_primary, 'steps_primary'),
            (self.df_secondary, 'steps_secondary'),
            (self.df_all, 'steps_all')
        ]

        for df, name in datasets:
            if len(df) == 0:
                continue

            for fmt in formats:
                sub_dir = self.output_dir / "steps_parts" / fmt
                sub_dir.mkdir(exist_ok=True, parents=True)

                if len(df) > MAX_ROWS:
                    num_parts = (len(df) // MAX_ROWS) + int(len(df) % MAX_ROWS > 0)
                    print(f"  ⚠ {name}: {len(df):,} строк → разбивка на {num_parts} файлов по {MAX_ROWS:,} строк")

                    for i in range(num_parts):
                        start = i * MAX_ROWS
                        end = start + MAX_ROWS
                        df_part = df.iloc[start:end]
                        filepath = sub_dir / f"{name}_part{i + 1}.{fmt}"

                        if fmt == 'csv':
                            df_part.to_csv(filepath, index=False, encoding='utf-8')
                        elif fmt == 'xlsx':
                            df_part.to_excel(filepath, index=False, engine='openpyxl')

                        print(f"    • Сохранено: {filepath.name} ({len(df_part):,} строк)")

                else:
                    filepath = sub_dir / f"{name}.{fmt}"
                    if fmt == 'csv':
                        df.to_csv(filepath, index=False, encoding='utf-8')
                    elif fmt == 'xlsx':
                        df.to_excel(filepath, index=False, engine='openpyxl')

                    print(f"  ✓ Сохранено: {filepath.name} ({len(df):,} строк)")

        print(f"Данные сохранены в подпапках: {self.output_dir / 'steps_parts'}")

    def create_visualizations(self, save_formats: List[str] = ['png']) -> Dict:
        """Создание всех визуализаций"""
        print("\nСоздание визуализаций...")

        figures = {}

        # 1. Анализ процессов
        fig_processes = self._create_process_analysis()
        figures['processes'] = fig_processes

        # 2. Анализ первичных частиц
        if len(self.df_primary) > 0:
            fig_primary = self._create_primary_analysis()
            figures['primary'] = fig_primary

        # 3. Анализ вторичных частиц
        if len(self.df_secondary) > 0:
            fig_secondary = self._create_secondary_analysis()
            figures['secondary'] = fig_secondary

        # Сохранение фигур
        for name, fig in figures.items():
            for fmt in save_formats:
                filename = self.output_dir / f'{name}_analysis.{fmt}'
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  Сохранено: {filename}")

        return figures

    def _create_process_analysis(self) -> Figure:
        """Создание графиков анализа процессов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ процессов', fontsize=14, fontweight='bold')

        # 1. Частота процессов
        process_counts = self.df['process'].value_counts().head(15)
        axes[0, 0].barh(range(len(process_counts)), process_counts.values)
        axes[0, 0].set_yticks(range(len(process_counts)))
        axes[0, 0].set_yticklabels(process_counts.index)
        axes[0, 0].set_xlabel('Количество шагов')
        axes[0, 0].set_title('Топ-15 процессов по частоте')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Энергетические потери по процессам
        process_energy = self.df.groupby('process')['de_step'].sum().sort_values(ascending=False).head(15)
        axes[0, 1].barh(range(len(process_energy)), process_energy.values)
        axes[0, 1].set_yticks(range(len(process_energy)))
        axes[0, 1].set_yticklabels(process_energy.index)
        axes[0, 1].set_xlabel('Суммарные потери энергии (MeV)')
        axes[0, 1].set_title('Топ-15 процессов по энергопотерям')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Сравнение первичных и вторичных по процессам
        if len(self.df_primary) > 0 and len(self.df_secondary) > 0:
            primary_proc = self.df_primary['process'].value_counts().head(10)
            secondary_proc = self.df_secondary['process'].value_counts().head(10)

            all_processes = list(set(primary_proc.index) | set(secondary_proc.index))
            x = np.arange(len(all_processes))
            width = 0.35

            primary_vals = [primary_proc.get(p, 0) for p in all_processes]
            secondary_vals = [secondary_proc.get(p, 0) for p in all_processes]

            axes[1, 0].bar(x - width / 2, primary_vals, width, label='Первичные', alpha=0.8)
            axes[1, 0].bar(x + width / 2, secondary_vals, width, label='Вторичные', alpha=0.8)
            axes[1, 0].set_xlabel('Процессы')
            axes[1, 0].set_ylabel('Количество шагов')
            axes[1, 0].set_title('Сравнение процессов: первичные vs вторичные')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(all_processes, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Средняя длина шага по процессам
        process_step_len = self.df.groupby('process')['step_leng'].mean().sort_values(ascending=False).head(15)
        axes[1, 1].barh(range(len(process_step_len)), process_step_len.values)
        axes[1, 1].set_yticks(range(len(process_step_len)))
        axes[1, 1].set_yticklabels(process_step_len.index)
        axes[1, 1].set_xlabel('Средняя длина шага (mm)')
        axes[1, 1].set_title('Средняя длина шага по процессам')
        axes[1, 1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_primary_analysis(self) -> Figure:
        """Создание графиков анализа первичных частиц"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ первичных частиц (Parent ID = 0)', fontsize=14, fontweight='bold')

        # 1. Распределение по типам частиц
        particle_counts = self.df_primary['particle'].value_counts()
        axes[0, 0].pie(particle_counts.values, labels=particle_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Распределение первичных частиц')

        # 2. Энергопотери первичных частиц
        particle_energy = self.df_primary.groupby('particle')['de_step'].sum().sort_values(ascending=False)
        axes[0, 1].bar(range(len(particle_energy)), particle_energy.values)
        axes[0, 1].set_xticks(range(len(particle_energy)))
        axes[0, 1].set_xticklabels(particle_energy.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Суммарные потери энергии (MeV)')
        axes[0, 1].set_title('Энергопотери первичных частиц')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Траектории первичных частиц (XY проекция)
        for particle in self.df_primary['particle'].unique():
            df_part = self.df_primary[self.df_primary['particle'] == particle]
            axes[1, 0].plot(df_part['x'], df_part['y'], 'o-', label=particle, alpha=0.6, markersize=3)
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        axes[1, 0].set_title('Траектории первичных частиц (XY)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Энергия vs пройденный путь
        for particle in self.df_primary['particle'].unique():
            df_part = self.df_primary[self.df_primary['particle'] == particle]
            axes[1, 1].plot(df_part['trak_leng'], df_part['kine_e'], 'o-', label=particle, alpha=0.6, markersize=3)
        axes[1, 1].set_xlabel('Пройденный путь (mm)')
        axes[1, 1].set_ylabel('Кинетическая энергия (MeV)')
        axes[1, 1].set_title('Зависимость энергии от пути')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_secondary_analysis(self) -> Figure:
        """Создание графиков анализа вторичных частиц"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ вторичных частиц (Parent ID > 0)', fontsize=14, fontweight='bold')

        # 1. Распределение по типам вторичных частиц
        particle_counts = self.df_secondary['particle'].value_counts().head(10)
        axes[0, 0].barh(range(len(particle_counts)), particle_counts.values)
        axes[0, 0].set_yticks(range(len(particle_counts)))
        axes[0, 0].set_yticklabels(particle_counts.index)
        axes[0, 0].set_xlabel('Количество шагов')
        axes[0, 0].set_title('Топ-10 вторичных частиц')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Энергопотери вторичных частиц
        particle_energy = self.df_secondary.groupby('particle')['de_step'].sum().sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(particle_energy)), particle_energy.values)
        axes[0, 1].set_yticks(range(len(particle_energy)))
        axes[0, 1].set_yticklabels(particle_energy.index)
        axes[0, 1].set_xlabel('Суммарные потери энергии (MeV)')
        axes[0, 1].set_title('Энергопотери вторичных частиц')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Распределение начальной энергии вторичных частиц
        secondary_tracks = self.df_secondary.groupby('track_id')['kine_e'].first()
        axes[1, 0].hist(secondary_tracks.values, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Начальная кинетическая энергия (MeV)')
        axes[1, 0].set_ylabel('Количество частиц')
        axes[1, 0].set_title('Распределение начальной энергии вторичных частиц')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)

        # 4. Средняя длина пути по типам вторичных частиц
        particle_path = self.df_secondary.groupby('particle')['trak_leng'].mean().sort_values(ascending=False).head(10)
        axes[1, 1].barh(range(len(particle_path)), particle_path.values)
        axes[1, 1].set_yticks(range(len(particle_path)))
        axes[1, 1].set_yticklabels(particle_path.index)
        axes[1, 1].set_xlabel('Средняя длина пути (mm)')
        axes[1, 1].set_title('Средний путь вторичных частиц')
        axes[1, 1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def verify_results(self, parser: Geant4LogParser) -> Dict:
        """Верификация результатов"""
        verification = {}

        # Анализ первичных частиц
        if len(self.df_primary) > 0:
            primary_tracks = self.df_primary['track_id'].unique()
            primary_initial = sum(parser.track_initial_energy.get(tid, 0) for tid in primary_tracks)
            primary_final = sum(parser.track_final_energy.get(tid, 0) for tid in primary_tracks)
            primary_lost = primary_initial - primary_final
            primary_de_step_positive = self.df_primary[self.df_primary['de_step'] > 0]['de_step'].sum()

            verification['primary'] = {
                'energy_balance': {
                    'initial_energy': primary_initial,
                    'final_energy': primary_final,
                    'energy_lost': primary_lost,
                    'total_de_step_positive': primary_de_step_positive
                }
            }

        # Анализ вторичных частиц
        if len(self.df_secondary) > 0:
            secondary_tracks = self.df_secondary['track_id'].unique()
            secondary_initial = sum(parser.track_initial_energy.get(tid, 0) for tid in secondary_tracks)
            secondary_final = sum(parser.track_final_energy.get(tid, 0) for tid in secondary_tracks)
            secondary_lost = secondary_initial - secondary_final
            secondary_de_step_positive = self.df_secondary[self.df_secondary['de_step'] > 0]['de_step'].sum()

            verification['secondary'] = {
                'energy_balance': {
                    'initial_energy': secondary_initial,
                    'final_energy': secondary_final,
                    'energy_lost': secondary_lost,
                    'total_de_step_positive': secondary_de_step_positive
                }
            }

        # Общая верификация
        all_tracks = self.df['track_id'].unique()
        total_initial = sum(parser.track_initial_energy.get(tid, 0) for tid in all_tracks)
        total_final = sum(parser.track_final_energy.get(tid, 0) for tid in all_tracks)
        total_lost = total_initial - total_final
        total_de_step_positive = self.df[self.df['de_step'] > 0]['de_step'].sum()

        verification['combined'] = {
            'energy_balance': {
                'initial_energy': total_initial,
                'final_energy': total_final,
                'energy_lost': total_lost,
                'total_de_step_positive': total_de_step_positive
            }
        }

        # Сверка с итоговой сводкой
        if 'energy_deposit' in self.summary and 'energy_leakage' in self.summary:
            energy_deposit_summary = self.summary['energy_deposit']
            energy_leakage_summary = self.summary['energy_leakage']
            total_summary = energy_deposit_summary + energy_leakage_summary

            verification['combined']['energy_deposit_summary'] = energy_deposit_summary
            verification['combined']['energy_leakage_summary'] = energy_leakage_summary
            verification['combined']['total_summary'] = total_summary

            # Метод 1: сумма положительных dEStep
            method1_diff = abs(total_de_step_positive - total_summary)
            method1_rel_diff = (method1_diff / total_summary * 100) if total_summary > 0 else 0

            verification['combined']['method1_absolute_difference'] = method1_diff
            verification['combined']['method1_relative_difference'] = method1_rel_diff

            # Метод 2: начальная - конечная энергия
            method2_diff = abs(total_lost - total_summary)
            method2_rel_diff = (method2_diff / total_summary * 100) if total_summary > 0 else 0

            verification['combined']['method2_absolute_difference'] = method2_diff
            verification['combined']['method2_relative_difference'] = method2_rel_diff

            # Проверка энергетического баланса первичных частиц
            if 'primary' in verification:
                primary_initial_energy = verification['primary']['energy_balance']['initial_energy']
                balance_diff = abs(primary_initial_energy - total_summary)
                balance_rel = (balance_diff / primary_initial_energy * 100) if primary_initial_energy > 0 else 0

                verification['combined']['primary_initial_energy'] = primary_initial_energy
                verification['combined']['balance_check_difference'] = balance_diff
                verification['combined']['balance_check_relative'] = balance_rel

        return verification

    def generate_report(self, parser: Geant4LogParser) -> str:
        """Генерация текстового отчета"""
        report = []
        report.append("=" * 100)
        report.append("ОТЧЕТ ОБ АНАЛИЗЕ ЛОГОВ GEANT4")
        report.append("=" * 100)
        report.append("")

        # Основная информация
        report.append(f"Входной файл: {parser.log_file}")
        report.append(f"Всего шагов: {len(self.df)}")
        report.append(f"Первичные частицы: {len(self.df_primary)} шагов")
        report.append(f"Вторичные частицы: {len(self.df_secondary)} шагов")
        report.append("")

        # Верификация
        verification = self.verify_results(parser)

        if 'primary' in verification:
            report.append("-" * 100)
            report.append("ПЕРВИЧНЫЕ ЧАСТИЦЫ (Parent ID = 0)")
            report.append("-" * 100)
            prim_balance = verification['primary']['energy_balance']
            report.append(f"Начальная энергия: {prim_balance['initial_energy']:.6f} MeV")
            report.append(f"Конечная энергия: {prim_balance['final_energy']:.6f} MeV")
            report.append(f"Потерянная энергия: {prim_balance['energy_lost']:.6f} MeV")
            report.append(f"Сумма положительных dEStep: {prim_balance['total_de_step_positive']:.6f} MeV")
            report.append("")

        if 'secondary' in verification:
            report.append("-" * 100)
            report.append("ВТОРИЧНЫЕ ЧАСТИЦЫ (Parent ID > 0)")
            report.append("-" * 100)
            sec_balance = verification['secondary']['energy_balance']
            report.append(f"Начальная энергия: {sec_balance['initial_energy']:.6f} MeV")
            report.append(f"Конечная энергия: {sec_balance['final_energy']:.6f} MeV")
            report.append(f"Потерянная энергия: {sec_balance['energy_lost']:.6f} MeV")
            report.append(f"Сумма положительных dEStep: {sec_balance['total_de_step_positive']:.6f} MeV")
            report.append("")

        if 'combined' in verification and 'energy_deposit_summary' in verification['combined']:
            report.append("-" * 100)
            report.append("ЭНЕРГЕТИЧЕСКИЙ БАЛАНС")
            report.append("-" * 100)
            comb = verification['combined']
            report.append(f"Energy deposit (из сводки): {comb['energy_deposit_summary']:.6f} MeV")
            report.append(f"Energy leakage (из сводки): {comb['energy_leakage_summary']:.6f} MeV")
            report.append(f"Сумма: {comb['total_summary']:.6f} MeV")
            report.append("")
            report.append(f"Начальная энергия первичных: {comb['primary_initial_energy']:.6f} MeV")
            report.append(
                f"Разница с E_deposit + E_leakage: {comb['balance_check_difference']:.6f} MeV ({comb['balance_check_relative']:.4f}%)")
            report.append("")

        report.append("=" * 100)
        return "\n".join(report)

    def save_report(self, parser: Geant4LogParser, filename: str = "analysis_report.txt") -> None:
        """Сохранение отчета в файл"""
        report = self.generate_report(parser)
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчет сохранен: {self.output_dir / filename}")


class Geant4GUI:
    """Графический интерфейс для анализа логов Geant4"""

    def __init__(self, root):
        self.root = root
        self.root.title("Geant4 Log Analyzer")
        self.root.geometry("1400x900")

        self.parser = None
        self.analyzer = None
        self.df = None
        self.figures = {}
        self.output_dir = None
        self.file_paths = {}

        self.create_widgets()

    def create_widgets(self):
        """Создание виджетов интерфейса"""

        # Верхняя панель - загрузка файла
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X, side=tk.TOP)

        # Первая строка - загрузка файла
        file_frame = ttk.Frame(top_frame)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(file_frame, text="Файл лога:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(file_frame, text="Файл не выбран", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.load_btn = ttk.Button(file_frame, text="Загрузить лог-файл", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(file_frame, text="Анализировать", command=self.analyze_file, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        # Вторая строка - форматы экспорта
        format_frame = ttk.LabelFrame(top_frame, text="Форматы экспорта", padding="5")
        format_frame.pack(fill=tk.X, pady=(5, 0))

        # Левая часть - графики
        plot_frame = ttk.Frame(format_frame)
        plot_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(plot_frame, text="Графики:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.format_png = tk.BooleanVar(value=True)
        self.format_svg = tk.BooleanVar(value=False)
        ttk.Checkbutton(plot_frame, text="PNG", variable=self.format_png).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(plot_frame, text="SVG", variable=self.format_svg).pack(side=tk.LEFT, padx=5)

        # Правая часть - данные
        data_frame = ttk.Frame(format_frame)
        data_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(data_frame, text="Данные:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.format_xlsx = tk.BooleanVar(value=True)
        self.format_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_frame, text="XLSX", variable=self.format_xlsx).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(data_frame, text="CSV", variable=self.format_csv).pack(side=tk.LEFT, padx=5)

        # Создание notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Вкладка 1: Обзор и результаты
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="📊 Обзор")
        self.create_overview_tab()

        # Вкладка 2: Анализ процессов
        self.processes_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.processes_tab, text="⚙️ Процессы")
        self.create_processes_tab()

        # Вкладка 3: Первичные частицы
        self.primary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.primary_tab, text="🔵 Первичные частицы")
        self.create_primary_tab()

        # Вкладка 4: Вторичные частицы
        self.secondary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.secondary_tab, text="🔴 Вторичные частицы")
        self.create_secondary_tab()

        # Вкладка 5: Файлы и экспорт
        self.files_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.files_tab, text="📁 Файлы")
        self.create_files_tab()

        # Строка состояния
        self.status_bar = ttk.Label(self.root, text="Готов к работе", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def create_overview_tab(self):
        """Создание вкладки обзора"""
        # Текстовое поле для отчета
        self.overview_text = scrolledtext.ScrolledText(self.overview_tab, wrap=tk.WORD, font=('Courier', 10))
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.overview_text.insert('1.0', "Загрузите лог-файл и нажмите 'Анализировать' для начала работы...")
        self.overview_text.config(state=tk.DISABLED)

    def create_processes_tab(self):
        """Создание вкладки анализа процессов"""
        # Фрейм для графика
        self.processes_canvas_frame = ttk.Frame(self.processes_tab)
        self.processes_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        label = ttk.Label(self.processes_canvas_frame, text="Графики появятся после анализа",
                          font=('Arial', 12), foreground="gray")
        label.pack(expand=True)

    def create_primary_tab(self):
        """Создание вкладки первичных частиц"""
        # Фрейм для графика
        self.primary_canvas_frame = ttk.Frame(self.primary_tab)
        self.primary_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        label = ttk.Label(self.primary_canvas_frame, text="Графики появятся после анализа",
                          font=('Arial', 12), foreground="gray")
        label.pack(expand=True)

    def create_secondary_tab(self):
        """Создание вкладки вторичных частиц"""
        # Фрейм для графика
        self.secondary_canvas_frame = ttk.Frame(self.secondary_tab)
        self.secondary_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        label = ttk.Label(self.secondary_canvas_frame, text="Графики появятся после анализа",
                          font=('Arial', 12), foreground="gray")
        label.pack(expand=True)

    def create_files_tab(self):
        """Создание вкладки для работы с файлами"""
        # Фрейм для открытия файлов
        files_frame = ttk.LabelFrame(self.files_tab, text="Сгенерированные файлы", padding="10")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Список файлов
        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox для файлов
        self.files_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Courier', 10))
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.files_listbox.yview)

        # Кнопки действий
        buttons_frame = ttk.Frame(files_frame)
        buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(buttons_frame, text="Открыть выбранный файл",
                   command=self.open_selected_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Открыть папку с результатами",
                   command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Обновить список",
                   command=self.refresh_files_list).pack(side=tk.LEFT, padx=5)

    def load_file(self):
        """Загрузка лог-файла"""
        filename = filedialog.askopenfilename(
            title="Выберите лог-файл Geant4",
            filetypes=[("Text files", "*.txt *.log"), ("All files", "*.*")]
        )

        if filename:
            self.log_file = filename
            self.file_label.config(text=os.path.basename(filename), foreground="black")
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Файл загружен: {filename}")

    def analyze_file(self):
        """Анализ файла"""
        if not hasattr(self, 'log_file'):
            # messagebox.showerror("Ошибка", "Сначала загрузите лог-файл")
            return

        # Получение выбранных форматов
        plot_formats = []
        if self.format_png.get():
            plot_formats.append('png')
        if self.format_svg.get():
            plot_formats.append('svg')

        data_formats = []
        if self.format_xlsx.get():
            data_formats.append('xlsx')
        if self.format_csv.get():
            data_formats.append('csv')

        # Проверка что хотя бы один формат выбран
        if not plot_formats:
            # messagebox.showwarning("Предупреждение", "Выберите хотя бы один формат для графиков (PNG или SVG)")
            return
        if not data_formats:
            # messagebox.showwarning("Предупреждение", "Выберите хотя бы один формат для данных (XLSX или CSV)")
            return

        try:
            self.status_bar.config(text="Идет анализ...")
            self.root.update()

            # Создание выходной директории
            self.output_dir = Path('output')
            self.output_dir.mkdir(exist_ok=True)

            # Парсинг лога
            self.parser = Geant4LogParser(self.log_file)
            self.parser.parse_log(debug=False)

            if not self.parser.steps:
                # messagebox.showerror("Ошибка", "Не найдено данных о шагах в логе!")
                self.status_bar.config(text="Ошибка: данные не найдены")
                return

            # Конвертация в DataFrame
            self.df = self.parser.to_dataframe()

            # Создание анализатора
            self.analyzer = Geant4Analyzer(self.df, self.parser.summary, str(self.output_dir), self.log_file)

            # Агрегация данных
            self.analyzer.aggregate_data()

            # Экспорт данных с выбранными форматами
            self.analyzer.export_data(formats=data_formats)

            # Визуализация с выбранными форматами
            self.figures = self.analyzer.create_visualizations(save_formats=plot_formats)

            # Генерация отчета
            self.analyzer.save_report(self.parser)

            # Обновление интерфейса
            self.update_overview()
            self.update_processes_plot()
            self.update_primary_plot()
            self.update_secondary_plot()
            self.refresh_files_list()

            self.status_bar.config(text="Анализ завершен успешно!")
            # messagebox.showinfo("Успех", f"Анализ завершен!\n\nРезультаты сохранены в папке: {self.output_dir}")

        except Exception as e:
            # messagebox.showerror("Ошибка", f"Произошла ошибка при анализе:\n{str(e)}")
            self.status_bar.config(text="Ошибка анализа")
            import traceback
            traceback.print_exc()

    def update_overview(self):
        """Обновление вкладки обзора"""
        if self.analyzer is None:
            return

        report = self.analyzer.generate_report(self.parser)

        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete('1.0', tk.END)
        self.overview_text.insert('1.0', report)
        self.overview_text.config(state=tk.DISABLED)

    def update_processes_plot(self):
        """Обновление графиков процессов"""
        if 'processes' not in self.figures:
            return

        # Очистка фрейма
        for widget in self.processes_canvas_frame.winfo_children():
            widget.destroy()

        # Создание canvas для matplotlib
        canvas = FigureCanvasTkAgg(self.figures['processes'], master=self.processes_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_primary_plot(self):
        """Обновление графиков первичных частиц"""
        if 'primary' not in self.figures:
            label = ttk.Label(self.primary_canvas_frame, text="Нет данных о первичных частицах",
                              font=('Arial', 12), foreground="gray")
            label.pack(expand=True)
            return

        # Очистка фрейма
        for widget in self.primary_canvas_frame.winfo_children():
            widget.destroy()

        # Создание canvas для matplotlib
        canvas = FigureCanvasTkAgg(self.figures['primary'], master=self.primary_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_secondary_plot(self):
        """Обновление графиков вторичных частиц"""
        if 'secondary' not in self.figures:
            label = ttk.Label(self.secondary_canvas_frame, text="Нет данных о вторичных частицах",
                              font=('Arial', 12), foreground="gray")
            label.pack(expand=True)
            return

        # Очистка фрейма
        for widget in self.secondary_canvas_frame.winfo_children():
            widget.destroy()

        # Создание canvas для matplotlib
        canvas = FigureCanvasTkAgg(self.figures['secondary'], master=self.secondary_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def refresh_files_list(self):
        """Обновление списка файлов"""
        if self.output_dir is None or not self.output_dir.exists():
            return

        self.files_listbox.delete(0, tk.END)

        # Получение списка всех файлов рекурсивно
        all_files = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                filepath = Path(root) / file
                relative_path = filepath.relative_to(self.output_dir)
                size = filepath.stat().st_size / 1024  # Размер в KB
                all_files.append((str(relative_path), size, filepath))

        # Сортировка по пути
        all_files.sort(key=lambda x: x[0])

        # Добавление в список
        for relative_path, size, filepath in all_files:
            if size < 1024:
                self.files_listbox.insert(tk.END, f"{relative_path} ({size:.1f} KB)")
            else:
                self.files_listbox.insert(tk.END, f"{relative_path} ({size / 1024:.1f} MB)")

        # Сохраняем полные пути для открытия
        self.file_paths = {str(f[0]): f[2] for f in all_files}

    def open_selected_file(self):
        """Открытие выбранного файла"""
        selection = self.files_listbox.curselection()
        if not selection:
            # messagebox.showwarning("Предупреждение", "Выберите файл из списка")
            return

        # Получаем относительный путь из строки (до первой скобки)
        selected_text = self.files_listbox.get(selection[0])
        relative_path = selected_text.split(' (')[0]

        # Получаем полный путь
        if not hasattr(self, 'file_paths') or relative_path not in self.file_paths:
            # messagebox.showerror("Ошибка", "Файл не найден")
            return

        filepath = self.file_paths[relative_path]

        if not filepath.exists():
            # messagebox.showerror("Ошибка", "Файл не найден")
            return

        # Открытие файла с помощью программы по умолчанию
        try:
            if sys.platform == 'win32':
                os.startfile(filepath)
            elif sys.platform == 'darwin':
                subprocess.run(['open', filepath])
            else:
                subprocess.run(['xdg-open', filepath])
        except Exception as e:
            pass
            # messagebox.showerror("Ошибка", f"Не удалось открыть файл: {str(e)}")

    def open_output_folder(self):
        """Открытие папки с результатами"""
        if self.output_dir is None or not self.output_dir.exists():
            # messagebox.showwarning("Предупреждение", "Папка с результатами не создана")
            return

        try:
            if sys.platform == 'win32':
                os.startfile(self.output_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', self.output_dir])
            else:
                subprocess.run(['xdg-open', self.output_dir])
        except Exception as e:
            pass
            # messagebox.showerror("Ошибка", f"Не удалось открыть папку: {str(e)}")


def main():
    """Запуск GUI приложения"""
    setup_log("geant4_parser")  # Лог перенаправлен в output/geant4_parser_log/...
    root = tk.Tk()
    app = Geant4GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
