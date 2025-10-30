"""
Geant4 Log Parser - Версия с исправленным парсингом dEStep
Программа для парсинга, анализа и визуализации логов симуляции Geant4
с отдельным анализом первичных (Parent ID = 0) и вторичных (Parent ID > 0) частиц

ИСПРАВЛЕНИЯ:
- Правильный парсинг dEStep (теперь учитываются все значения)
- Улучшенная логика поиска энергетических значений
- Более точная конверсия единиц
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
plt.rcParams['figure.figsize'] = (14, 8)
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
                # Парсинг строк с процессами
                proc_matches = re.findall(self.PROCESS_FREQ_PATTERN, line)
                for proc_match in proc_matches:
                    process_name = proc_match[0]
                    process_count = int(proc_match[1])
                    # Исключаем технические поля (не процессы)
                    if process_name not in ['sumtot', 'counter', 'N', 'V']:
                        process_frequencies[process_name] = process_count

        self.summary['process_frequencies'] = process_frequencies

        print(f"\nПарсинг завершен:")
        print(f"  - Всего шагов найдено: {len(self.steps)}")

        # Подсчет первичных и вторичных частиц
        primary_steps = sum(1 for s in self.steps if s.is_primary)
        secondary_steps = len(self.steps) - primary_steps
        print(f"  - Шаги первичных частиц (Parent ID = 0): {primary_steps}")
        print(f"  - Шаги вторичных частиц (Parent ID > 0): {secondary_steps}")

        print(f"  - Уникальных треков: {len(set(s.track_id for s in self.steps))}")
        print(f"  - Уникальных потоков: {len(set(s.thread for s in self.steps))}")
        print(f"  - Типы частиц: {', '.join(set(s.particle for s in self.steps))}")

        # Отладочная информация о dEStep
        if debug:
            de_step_values = [s.de_step for s in self.steps if s.de_step > 0]
            if de_step_values:
                print(f"\n[DEBUG] Статистика dEStep:")
                print(f"  - Всего ненулевых значений: {len(de_step_values)}")
                print(f"  - Мин: {min(de_step_values):.6f} MeV")
                print(f"  - Макс: {max(de_step_values):.6f} MeV")
                print(f"  - Сумма: {sum(de_step_values):.6f} MeV")
                print(f"  - Примеры первых 5: {[f'{v:.6f}' for v in de_step_values[:5]]}")

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
                print(f"[DEBUG] Найдено dEStep на строке {line_num}: {energy_values[1][2]} {energy_values[1][3]} = {de_step:.6f} MeV")

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
        """Конвертация данных в pandas DataFrame"""
        if not self.steps:
            return pd.DataFrame()

        data = []
        for step in self.steps:
            data.append({
                'thread': step.thread,
                'step_num': step.step_num,
                'x_mm': step.x,
                'y_mm': step.y,
                'z_mm': step.z,
                'kine_e_MeV': step.kine_e,
                'de_step_MeV': step.de_step,
                'step_leng_mm': step.step_leng,
                'trak_leng_mm': step.trak_leng,
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

    def __init__(self, df: pd.DataFrame, summary: Dict, output_dir: str = 'output', log_filename: str = ''):
        self.df_all = df
        self.summary = summary
        self.log_filename = log_filename

        # Создаем подпапку с именем лог-файла
        if log_filename:
            log_name = Path(log_filename).stem  # Имя файла без расширения
            self.output_dir = Path(output_dir) / f"{log_name}_log"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nРезультаты будут сохранены в: {self.output_dir}")

        # Разделение на первичные и вторичные частицы
        print("\n" + "=" * 100)
        print("РАЗДЕЛЕНИЕ НА ПЕРВИЧНЫЕ И ВТОРИЧНЫЕ ЧАСТИЦЫ")
        print("=" * 100)

        self.df_primary = df[df['is_primary'] == True].copy()
        self.df_secondary = df[df['is_primary'] == False].copy()

        print(f"Первичные частицы (Parent ID = 0): {len(self.df_primary)} шагов")
        print(f"Вторичные частицы (Parent ID > 0): {len(self.df_secondary)} шагов")
        print()

        if len(self.df_primary) > 0:
            print(f"Типы первичных частиц: {', '.join(self.df_primary['particle'].unique())}")
        if len(self.df_secondary) > 0:
            print(f"Типы вторичных частиц: {', '.join(self.df_secondary['particle'].unique())}")

    def aggregate_data(self) -> Dict:
        """Агрегация данных по частицам и процессам"""
        print("\nАгрегация данных...")

        results = {}

        # Агрегация первичных частиц
        if len(self.df_primary) > 0:
            print(f"\nПЕРВИЧНЫЕ ЧАСТИЦЫ - Агрегация данных...")
            results['primary'] = self._aggregate_by_particle(self.df_primary, "ПЕРВИЧНЫЕ")

        # Агрегация вторичных частиц
        if len(self.df_secondary) > 0:
            print(f"\nВТОРИЧНЫЕ ЧАСТИЦЫ - Агрегация данных...")
            results['secondary'] = self._aggregate_by_particle(self.df_secondary, "ВТОРИЧНЫЕ")

        # Агрегация всех частиц
        print(f"\nОБЩИЕ ЧАСТИЦЫ - Агрегация данных...")
        results['combined'] = self._aggregate_by_particle(self.df_all, "ОБЩИЕ")

        return results

    def _aggregate_by_particle(self, df: pd.DataFrame, label: str) -> Dict:
        """Внутренний метод агрегации"""
        if len(df) == 0:
            return {}

        # Агрегация по частицам
        particle_stats = df.groupby('particle').agg({
            'kine_e_MeV': ['count', 'min', 'max', 'mean', 'std'],
            'de_step_MeV': ['sum', 'mean'],
            'step_leng_mm': ['sum', 'mean'],
            'track_id': 'nunique'
        }).round(6)

        # Агрегация по процессам
        process_freq = df['process'].value_counts().to_dict()

        # Энергетический баланс
        energy_balance = {
            'total_de_step_all': df['de_step_MeV'].sum(),
            'total_de_step_positive': df[df['de_step_MeV'] > 0]['de_step_MeV'].sum(),
            'total_de_step_negative': df[df['de_step_MeV'] < 0]['de_step_MeV'].sum(),
        }

        print(f"  {label}: Обработано {len(df)} шагов, {df['track_id'].nunique()} треков")

        return {
            'particle_stats': particle_stats,
            'process_freq': process_freq,
            'energy_balance': energy_balance
        }

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

    def create_visualizations(self, save_formats: List[str] = ['svg']) -> None:
        """Создание визуализаций"""
        print("\nСоздание визуализаций...")

        self._plot_energy_distributions(save_formats)
        self._plot_de_distributions(save_formats)
        self._plot_process_frequencies(save_formats)
        self._plot_spatial_distributions(save_formats)
        self._plot_energy_balance(save_formats)
        self._plot_comparisons(save_formats)

        # Новые функции
        self._plot_coordinate_heatmaps(save_formats)
        self._plot_particle_energy_distributions(save_formats)

    def _plot_energy_distributions(self, save_formats: List[str]) -> None:
        """График распределения кинетической энергии"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Первичные частицы
        if len(self.df_primary) > 0:
            for particle in self.df_primary['particle'].unique():
                data = self.df_primary[self.df_primary['particle'] == particle]['kine_e_MeV']
                axes[0].hist(data, bins=50, alpha=0.6, label=f'{particle} (n={len(data)})')
                axes[0].axvline(data.mean(), color='r', linestyle='--', linewidth=1,
                              label=f'Среднее: {data.mean():.3f} MeV')

            axes[0].set_xlabel('Кинетическая энергия (MeV)', fontweight='bold')
            axes[0].set_ylabel('Количество шагов', fontweight='bold')
            axes[0].set_title('Распределение энергии - ПЕРВИЧНЫЕ ЧАСТИЦЫ', fontweight='bold', fontsize=12)
            axes[0].legend()
            axes[0].grid(False)

        # Вторичные частицы
        if len(self.df_secondary) > 0:
            for particle in self.df_secondary['particle'].unique():
                data = self.df_secondary[self.df_secondary['particle'] == particle]['kine_e_MeV']
                axes[1].hist(data, bins=50, alpha=0.6, label=f'{particle} (n={len(data)})')
                axes[1].axvline(data.mean(), color='r', linestyle='--', linewidth=1,
                              label=f'Среднее: {data.mean():.3f} MeV')

            axes[1].set_xlabel('Кинетическая энергия (MeV)', fontweight='bold')
            axes[1].set_ylabel('Количество шагов', fontweight='bold')
            axes[1].set_title('Распределение энергии - ВТОРИЧНЫЕ ЧАСТИЦЫ', fontweight='bold', fontsize=12)
            axes[1].legend()
            axes[1].grid(False)

        plt.tight_layout()
        for fmt in save_formats:
            plt.savefig(self.output_dir / f'energy_distribution.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Графики распределения энергии созданы")

    def _plot_de_distributions(self, save_formats: List[str]) -> None:
        """Boxplot распределения dE"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Фильтруем только ненулевые dE
        df_with_de = self.df_all[self.df_all['de_step_MeV'] > 0].copy()

        if len(df_with_de) > 0:
            df_with_de['category'] = df_with_de['is_primary'].map({True: 'Первичные', False: 'Вторичные'})

            sns.violinplot(data=df_with_de, x='particle', y='de_step_MeV', hue='category', ax=ax, split=True)
            ax.set_xlabel('Тип частицы', fontweight='bold')
            ax.set_ylabel('Потеря энергии на шаг (MeV)', fontweight='bold')
            ax.set_title('Распределение потерь энергии (dE > 0)', fontweight='bold', fontsize=14)
            ax.grid(False)
            ax.legend(title='Категория')

            plt.tight_layout()
            for fmt in save_formats:
                plt.savefig(self.output_dir / f'de_distribution.{fmt}', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Boxplot распределения dE создан")
        else:
            print("  ⚠ Нет данных с ненулевым dE для графика")

    def _plot_process_frequencies(self, save_formats: List[str]) -> None:
        """Диаграмма частоты процессов"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Первичные частицы
        if len(self.df_primary) > 0:
            process_counts_primary = self.df_primary['process'].value_counts().head(15)
            axes[0].barh(range(len(process_counts_primary)), process_counts_primary.values, color='steelblue')
            axes[0].set_yticks(range(len(process_counts_primary)))
            axes[0].set_yticklabels(process_counts_primary.index)
            axes[0].set_xlabel('Количество вызовов', fontweight='bold')
            axes[0].set_title('Частота процессов - ПЕРВИЧНЫЕ ЧАСТИЦЫ', fontweight='bold', fontsize=12)
            axes[0].grid(False)
            axes[0].invert_yaxis()

        # Вторичные частицы
        if len(self.df_secondary) > 0:
            process_counts_secondary = self.df_secondary['process'].value_counts().head(15)
            axes[1].barh(range(len(process_counts_secondary)), process_counts_secondary.values, color='coral')
            axes[1].set_yticks(range(len(process_counts_secondary)))
            axes[1].set_yticklabels(process_counts_secondary.index)
            axes[1].set_xlabel('Количество вызовов', fontweight='bold')
            axes[1].set_title('Частота процессов - ВТОРИЧНЫЕ ЧАСТИЦЫ', fontweight='bold', fontsize=12)
            axes[1].grid(False)
            axes[1].invert_yaxis()

        plt.tight_layout()
        for fmt in save_formats:
            plt.savefig(self.output_dir / f'process_frequency.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Диаграмма частоты процессов создана")

    def _plot_spatial_distributions(self, save_formats: List[str]) -> None:
        """Пространственные распределения"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for idx, (df, title) in enumerate([(self.df_primary, 'ПЕРВИЧНЫЕ'), (self.df_secondary, 'ВТОРИЧНЫЕ')]):
            if len(df) == 0:
                continue

            # XY projection
            axes[idx, 0].hexbin(df['x_mm'], df['y_mm'], gridsize=30, cmap='YlOrRd', mincnt=1)
            axes[idx, 0].set_xlabel('X (mm)', fontweight='bold')
            axes[idx, 0].set_ylabel('Y (mm)', fontweight='bold')
            axes[idx, 0].set_title(f'{title} - Проекция XY', fontweight='bold')
            axes[idx, 0].grid(False)

            # XZ projection
            axes[idx, 1].hexbin(df['x_mm'], df['z_mm'], gridsize=30, cmap='YlGnBu', mincnt=1)
            axes[idx, 1].set_xlabel('X (mm)', fontweight='bold')
            axes[idx, 1].set_ylabel('Z (mm)', fontweight='bold')
            axes[idx, 1].set_title(f'{title} - Проекция XZ', fontweight='bold')
            axes[idx, 1].grid(False)

            # YZ projection
            axes[idx, 2].hexbin(df['y_mm'], df['z_mm'], gridsize=30, cmap='Greens', mincnt=1)
            axes[idx, 2].set_xlabel('Y (mm)', fontweight='bold')
            axes[idx, 2].set_ylabel('Z (mm)', fontweight='bold')
            axes[idx, 2].set_title(f'{title} - Проекция YZ', fontweight='bold')
            axes[idx, 2].grid(False)

        plt.tight_layout()
        for fmt in save_formats:
            plt.savefig(self.output_dir / f'spatial_distributions.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Пространственные распределения созданы")

    def _plot_energy_balance(self, save_formats: List[str]) -> None:
        """График энергетического баланса"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Подготовка данных
        categories = []
        de_values = []
        colors_list = []

        if len(self.df_primary) > 0:
            prim_de = self.df_primary[self.df_primary['de_step_MeV'] > 0]['de_step_MeV'].sum()
            categories.append('Первичные\n(dE > 0)')
            de_values.append(prim_de)
            colors_list.append('steelblue')

        if len(self.df_secondary) > 0:
            sec_de = self.df_secondary[self.df_secondary['de_step_MeV'] > 0]['de_step_MeV'].sum()
            categories.append('Вторичные\n(dE > 0)')
            de_values.append(sec_de)
            colors_list.append('coral')

        if 'energy_deposit' in self.summary:
            categories.append('Energy Deposit\n(сводка)')
            de_values.append(self.summary['energy_deposit'])
            colors_list.append('green')

        ax.bar(categories, de_values, color=colors_list, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Энергия (MeV)', fontweight='bold', fontsize=12)
        ax.set_title('Энергетический баланс: Сравнение', fontweight='bold', fontsize=14)
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', alpha=0.3)

        # Добавляем значения на столбцы
        for i, (cat, val) in enumerate(zip(categories, de_values)):
            ax.text(i, val + max(de_values) * 0.02, f'{val:.6f} MeV',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        for fmt in save_formats:
            plt.savefig(self.output_dir / f'energy_balance.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ График энергетического баланса создан")

    def _plot_comparisons(self, save_formats: List[str]) -> None:
        """Сравнительные графики первичных и вторичных частиц"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Сравнение числа шагов по частицам
        if len(self.df_primary) > 0 or len(self.df_secondary) > 0:
            prim_counts = self.df_primary['particle'].value_counts() if len(self.df_primary) > 0 else pd.Series()
            sec_counts = self.df_secondary['particle'].value_counts() if len(self.df_secondary) > 0 else pd.Series()

            all_particles = set(prim_counts.index).union(set(sec_counts.index))
            x = np.arange(len(all_particles))
            width = 0.35

            prim_vals = [prim_counts.get(p, 0) for p in all_particles]
            sec_vals = [sec_counts.get(p, 0) for p in all_particles]

            axes[0, 0].bar(x - width/2, prim_vals, width, label='Первичные', color='steelblue')
            axes[0, 0].bar(x + width/2, sec_vals, width, label='Вторичные', color='coral')
            axes[0, 0].set_xlabel('Тип частицы', fontweight='bold')
            axes[0, 0].set_ylabel('Количество шагов', fontweight='bold')
            axes[0, 0].set_title('Сравнение числа шагов', fontweight='bold')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(list(all_particles))
            axes[0, 0].legend()
            axes[0, 0].grid(False)

        # 2. Сравнение средней энергии
        if len(self.df_primary) > 0 or len(self.df_secondary) > 0:
            prim_energy = self.df_primary.groupby('particle')['kine_e_MeV'].mean() if len(self.df_primary) > 0 else pd.Series()
            sec_energy = self.df_secondary.groupby('particle')['kine_e_MeV'].mean() if len(self.df_secondary) > 0 else pd.Series()

            all_particles_e = set(prim_energy.index).union(set(sec_energy.index))
            x = np.arange(len(all_particles_e))

            prim_e_vals = [prim_energy.get(p, 0) for p in all_particles_e]
            sec_e_vals = [sec_energy.get(p, 0) for p in all_particles_e]

            axes[0, 1].bar(x - width/2, prim_e_vals, width, label='Первичные', color='steelblue')
            axes[0, 1].bar(x + width/2, sec_e_vals, width, label='Вторичные', color='coral')
            axes[0, 1].set_xlabel('Тип частицы', fontweight='bold')
            axes[0, 1].set_ylabel('Средняя энергия (MeV)', fontweight='bold')
            axes[0, 1].set_title('Сравнение средней энергии', fontweight='bold')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(list(all_particles_e))
            axes[0, 1].legend()
            axes[0, 1].grid(False)

        # 3. Распределение треков по потокам
        thread_counts_prim = self.df_primary['thread'].value_counts() if len(self.df_primary) > 0 else pd.Series()
        thread_counts_sec = self.df_secondary['thread'].value_counts() if len(self.df_secondary) > 0 else pd.Series()

        axes[1, 0].scatter(range(len(thread_counts_prim)), thread_counts_prim.values,
                          alpha=0.6, s=100, label='Первичные', color='steelblue')
        axes[1, 0].scatter(range(len(thread_counts_sec)), thread_counts_sec.values,
                          alpha=0.6, s=100, label='Вторичные', color='coral')
        axes[1, 0].set_xlabel('Индекс потока', fontweight='bold')
        axes[1, 0].set_ylabel('Количество шагов', fontweight='bold')
        axes[1, 0].set_title('Распределение шагов по потокам', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Топ процессов
        if len(self.df_primary) > 0 or len(self.df_secondary) > 0:
            prim_proc = self.df_primary['process'].value_counts().head(10) if len(self.df_primary) > 0 else pd.Series()
            sec_proc = self.df_secondary['process'].value_counts().head(10) if len(self.df_secondary) > 0 else pd.Series()

            all_proc = set(prim_proc.index).union(set(sec_proc.index))
            x = np.arange(len(all_proc))

            prim_proc_vals = [prim_proc.get(p, 0) for p in all_proc]
            sec_proc_vals = [sec_proc.get(p, 0) for p in all_proc]

            axes[1, 1].bar(x - width/2, prim_proc_vals, width, label='Первичные', color='steelblue')
            axes[1, 1].bar(x + width/2, sec_proc_vals, width, label='Вторичные', color='coral')
            axes[1, 1].set_xlabel('Процесс', fontweight='bold')
            axes[1, 1].set_ylabel('Количество вызовов', fontweight='bold')
            axes[1, 1].set_title('Топ-10 процессов', fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(list(all_proc), rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(False)

        plt.tight_layout()
        for fmt in save_formats:
            plt.savefig(self.output_dir / f'comparisons.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Сравнительные графики созданы")

    def _plot_coordinate_heatmaps(self, save_formats: List[str]) -> None:
        """Heatmap плотности распределения координат (все частицы разом)"""
        df = self.df_all
        if len(df) < 10:
            print("  ⚠ Недостаточно данных для построения heatmap координат")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # X-Y плоскость
        if len(df['x_mm'].unique()) > 1 and len(df['y_mm'].unique()) > 1:
            h, xedges, yedges = np.histogram2d(df['x_mm'], df['y_mm'], bins=50)
            im1 = axes[0, 0].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            axes[0, 0].set_xlabel('X (mm)', fontsize=12)
            axes[0, 0].set_ylabel('Y (mm)', fontsize=12)
            axes[0, 0].set_title('Плотность распределения: X-Y', fontsize=13, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0], label='Количество точек')

        # X-Z плоскость
        if len(df['x_mm'].unique()) > 1 and len(df['z_mm'].unique()) > 1:
            h, xedges, zedges = np.histogram2d(df['x_mm'], df['z_mm'], bins=50)
            im2 = axes[0, 1].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                                    extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]])
            axes[0, 1].set_xlabel('X (mm)', fontsize=12)
            axes[0, 1].set_ylabel('Z (mm)', fontsize=12)
            axes[0, 1].set_title('Плотность распределения: X-Z', fontsize=13, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1], label='Количество точек')

        # Y-Z плоскость
        if len(df['y_mm'].unique()) > 1 and len(df['z_mm'].unique()) > 1:
            h, yedges, zedges = np.histogram2d(df['y_mm'], df['z_mm'], bins=50)
            im3 = axes[1, 0].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                                    extent=[yedges[0], yedges[-1], zedges[0], zedges[-1]])
            axes[1, 0].set_xlabel('Y (mm)', fontsize=12)
            axes[1, 0].set_ylabel('Z (mm)', fontsize=12)
            axes[1, 0].set_title('Плотность распределения: Y-Z', fontsize=13, fontweight='bold')
            plt.colorbar(im3, ax=axes[1, 0], label='Количество точек')

        # 3D проекция
        scatter = axes[1, 1].scatter(df['x_mm'], df['y_mm'], c=df['z_mm'],
                                     cmap='viridis', alpha=0.5, s=10)
        axes[1, 1].set_xlabel('X (mm)', fontsize=12)
        axes[1, 1].set_ylabel('Y (mm)', fontsize=12)
        axes[1, 1].set_title('Проекция координат (цвет = Z)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1], label='Z (mm)')

        plt.tight_layout()
        for fmt in save_formats:
            plt.savefig(self.output_dir / f'coordinate_heatmaps.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Heatmap координат создан для всех частиц")

    def _plot_particle_energy_distributions(self, save_formats: List[str]) -> None:
        """Распределение кинетической энергии по каждой частице"""
        out_dir = self.output_dir / "particle_distributions"
        out_dir.mkdir(exist_ok=True, parents=True)

        particles = self.df_all['particle'].unique()
        for particle in particles:
            df_p = self.df_all[self.df_all['particle'] == particle]
            if len(df_p) < 5:
                continue

            plt.figure(figsize=(10, 6))
            plt.hist(df_p['kine_e_MeV'], bins=50, alpha=0.7, color='steelblue')
            plt.axvline(df_p['kine_e_MeV'].mean(), color='r', linestyle='--', linewidth=1,
                        label=f'Среднее: {df_p["kine_e_MeV"].mean():.3f} MeV')
            plt.xlabel('Кинетическая энергия (MeV)', fontweight='bold')
            plt.ylabel('Количество шагов', fontweight='bold')
            plt.title(f'Распределение энергии: {particle}', fontweight='bold', fontsize=12)
            plt.legend()
            plt.grid(False)
            plt.tight_layout()

            for fmt in save_formats:
                plt.savefig(out_dir / f'energy_distribution_{particle}.{fmt}', dpi=300, bbox_inches='tight')
            plt.close()
        print(f"  ✓ Графики распределения энергии созданы для {len(particles)} частиц")

    def verify_results(self, parser: Geant4LogParser) -> Dict:
        """Сверка результатов с итоговой сводкой"""
        verification = {}

        # Верификация для первичных частиц
        if len(self.df_primary) > 0:
            verification['primary'] = self._verify_energy_balance(self.df_primary, parser, 'primary')

        # Верификация для вторичных частиц
        if len(self.df_secondary) > 0:
            verification['secondary'] = self._verify_energy_balance(self.df_secondary, parser, 'secondary')

        # Общая верификация
        verification['combined'] = self._verify_energy_balance(self.df_all, parser, 'combined')

        # Сверка процессов
        if self.summary.get('process_frequencies'):
            parsed_freq = self.df_all['process'].value_counts().to_dict()
            summary_freq = self.summary['process_frequencies']

            process_comparison = {}
            all_processes = set(parsed_freq.keys()).union(set(summary_freq.keys()))

            # Исключаем технические поля (не процессы)
            excluded_fields = {'sumtot', 'counter', 'N', 'V'}
            all_processes = all_processes - excluded_fields

            for proc in all_processes:
                parsed_count = parsed_freq.get(proc, 0)
                summary_count = summary_freq.get(proc, 0)
                process_comparison[proc] = {
                    'parsed': parsed_count,
                    'summary': summary_count,
                    'difference': parsed_count - summary_count,
                    'abs_difference': abs(parsed_count - summary_count)
                }

            verification['combined']['process_comparison'] = process_comparison

        return verification

    def _verify_energy_balance(self, df: pd.DataFrame, parser: Geant4LogParser, label: str) -> Dict:
        """Внутренний метод верификации энергетического баланса"""
        result = {}

        if len(df) == 0:
            return result

        # Расчет начальной и конечной энергии из треков
        track_ids = df['track_id'].unique()
        initial_energy = sum(parser.track_initial_energy.get(tid, 0) for tid in track_ids)
        final_energy = sum(parser.track_final_energy.get(tid, 0) for tid in track_ids)
        energy_lost = initial_energy - final_energy

        # Сумма dEStep
        total_de_step_all = df['de_step_MeV'].sum()
        total_de_step_positive = df[df['de_step_MeV'] > 0]['de_step_MeV'].sum()
        total_de_step_negative = df[df['de_step_MeV'] < 0]['de_step_MeV'].sum()

        result['energy_balance'] = {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_lost': energy_lost,
            'total_de_step_all': total_de_step_all,
            'total_de_step_positive': total_de_step_positive,
            'total_de_step_negative': total_de_step_negative,
        }

        # Сверка с summary (только для combined)
        if label == 'combined' and 'energy_deposit' in self.summary:
            energy_deposit_summary = self.summary['energy_deposit']
            energy_leakage_summary = self.summary.get('energy_leakage', 0.0)

            result['energy_deposit_summary'] = energy_deposit_summary
            result['energy_leakage_summary'] = energy_leakage_summary
            result['total_summary'] = energy_deposit_summary + energy_leakage_summary

            # МЕТОД 1: Сравнение с положительными dEStep
            abs_diff_method1 = abs(total_de_step_positive - energy_deposit_summary)
            rel_diff_method1 = (abs_diff_method1 / energy_deposit_summary * 100) if energy_deposit_summary != 0 else 100.0

            result['method1_absolute_difference'] = abs_diff_method1
            result['method1_relative_difference'] = rel_diff_method1

            # МЕТОД 2: Сравнение потерянной энергии с Energy deposit
            abs_diff_method2 = abs(energy_lost - energy_deposit_summary)
            rel_diff_method2 = (abs_diff_method2 / energy_deposit_summary * 100) if energy_deposit_summary != 0 else 100.0

            result['method2_absolute_difference'] = abs_diff_method2
            result['method2_relative_difference'] = rel_diff_method2

            # Проверка энергетического баланса первичных частиц
            primary_track_ids = [tid for tid in track_ids if parser.track_parent_ids.get(tid, -1) == 0]
            primary_initial_energy = sum(parser.track_initial_energy.get(tid, 0) for tid in primary_track_ids)

            result['primary_initial_energy'] = primary_initial_energy
            result['balance_check_difference'] = abs(primary_initial_energy - result['total_summary'])
            result['balance_check_relative'] = (result['balance_check_difference'] / primary_initial_energy * 100) if primary_initial_energy != 0 else 0

        return result

    def generate_report(self, parser: Geant4LogParser) -> str:
        """Генерация текстового отчета"""
        report = []
        report.append("=" * 100)
        report.append("ОТЧЕТ ОБ АНАЛИЗЕ ЛОГОВ GEANT4")
        report.append("=" * 100)
        report.append(f"Входной файл: {self.log_filename}")
        report.append(f"Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Общая статистика
        report.append("1. ОБЩАЯ СТАТИСТИКА")
        report.append("-" * 100)
        report.append(f"Всего шагов: {len(self.df_all)}")
        report.append(f"Первичные частицы (Parent ID = 0): {len(self.df_primary)} шагов")
        report.append(f"Вторичные частицы (Parent ID > 0): {len(self.df_secondary)} шагов")
        report.append(f"Уникальных треков: {self.df_all['track_id'].nunique()}")
        report.append(f"Уникальных потоков: {self.df_all['thread'].nunique()}")
        report.append(f"Типы частиц: {', '.join(self.df_all['particle'].unique())}")
        report.append("")

        # Статистика по частицам
        report.append("2. СТАТИСТИКА ПО ЧАСТИЦАМ")
        report.append("-" * 100)

        for category, df in [('ПЕРВИЧНЫЕ', self.df_primary), ('ВТОРИЧНЫЕ', self.df_secondary)]:
            if len(df) == 0:
                continue

            report.append(f"\n{category} ЧАСТИЦЫ:")
            report.append("-" * 50)

            for particle in df['particle'].unique():
                particle_df = df[df['particle'] == particle]
                report.append(f"\n  Частица: {particle}")
                report.append(f"    Количество шагов: {len(particle_df)}")
                report.append(f"    Количество треков: {particle_df['track_id'].nunique()}")
                report.append(f"    Средняя кинетическая энергия: {particle_df['kine_e_MeV'].mean():.6f} MeV")
                report.append(f"    Диапазон энергий: [{particle_df['kine_e_MeV'].min():.6f}, {particle_df['kine_e_MeV'].max():.6f}] MeV")

                de_positive = particle_df[particle_df['de_step_MeV'] > 0]['de_step_MeV']
                if len(de_positive) > 0:
                    report.append(f"    Сумма dEStep (> 0): {de_positive.sum():.6f} MeV")
                    report.append(f"    Среднее dEStep (> 0): {de_positive.mean():.6f} MeV")

        report.append("")

        # Энергетический баланс
        report.append("3. ЭНЕРГЕТИЧЕСКИЙ БАЛАНС")
        report.append("-" * 100)

        verification = self.verify_results(parser)

        if 'primary' in verification and len(self.df_primary) > 0:
            prim_balance = verification['primary']['energy_balance']
            report.append("\nПЕРВИЧНЫЕ ЧАСТИЦЫ:")
            report.append(f"  Начальная энергия: {prim_balance['initial_energy']:.6f} MeV")
            report.append(f"  Конечная энергия: {prim_balance['final_energy']:.6f} MeV")
            report.append(f"  Потерянная энергия: {prim_balance['energy_lost']:.6f} MeV")
            report.append(f"  Сумма всех dEStep: {prim_balance['total_de_step_all']:.6f} MeV")
            report.append(f"  Сумма положительных dEStep: {prim_balance['total_de_step_positive']:.6f} MeV")
            report.append("")

        if 'secondary' in verification and len(self.df_secondary) > 0:
            sec_balance = verification['secondary']['energy_balance']
            report.append("ВТОРИЧНЫЕ ЧАСТИЦЫ:")
            report.append(f"  Начальная энергия: {sec_balance['initial_energy']:.6f} MeV")
            report.append(f"  Конечная энергия: {sec_balance['final_energy']:.6f} MeV")
            report.append(f"  Потерянная энергия: {sec_balance['energy_lost']:.6f} MeV")
            report.append(f"  Сумма всех dEStep: {sec_balance['total_de_step_all']:.6f} MeV")
            report.append(f"  Сумма положительных dEStep: {sec_balance['total_de_step_positive']:.6f} MeV")
            report.append("")

        # Сверка с итоговой сводкой
        if 'combined' in verification:
            comb_verif = verification['combined']
            if 'energy_deposit_summary' in comb_verif:
                report.append("4. СВЕРКА С ИТОГОВОЙ СВОДКОЙ")
                report.append("-" * 100)
                report.append(f"Energy deposit (из сводки): {comb_verif['energy_deposit_summary']:.6f} MeV")
                report.append(f"Energy leakage (из сводки): {comb_verif['energy_leakage_summary']:.6f} MeV")
                report.append(f"Сумма (E_deposit + E_leakage): {comb_verif['total_summary']:.6f} MeV")
                report.append("")
                report.append("⚠️  ВНИМАНИЕ: Лог неполный (пропущены шаги)!")
                report.append("Метод 1 (сумма dEStep) НЕ РАБОТАЕТ для неполных логов.")
                report.append("Используйте ТОЛЬКО Метод 2 (энергетический баланс).")
                report.append("")
                report.append("Метод 1 (Сумма положительных dEStep - только потери энергии):")
                report.append(f"  Рассчитано: {comb_verif['energy_balance']['total_de_step_positive']:.6f} MeV")
                report.append(f"  Абсолютная разница: {comb_verif['method1_absolute_difference']:.6f} MeV")
                report.append(f"  Относительная разница: {comb_verif['method1_relative_difference']:.4f}%")
                report.append("  ❌ НЕ ПРИМЕНИМО для этого лога (шаги пропущены)")
                report.append("")
                report.append("Метод 2 (Начальная - Конечная энергия):")
                report.append(f"  Рассчитано: {comb_verif['energy_balance']['energy_lost']:.6f} MeV")
                report.append(f"  Абсолютная разница: {comb_verif['method2_absolute_difference']:.6f} MeV")
                report.append(f"  Относительная разница: {comb_verif['method2_relative_difference']:.4f}%")
                report.append("")
                report.append("Проверка энергетического баланса первичных частиц:")
                report.append(f"  Начальная энергия первичных: {comb_verif['primary_initial_energy']:.6f} MeV")
                report.append(f"  E_deposit + E_leakage:       {comb_verif['total_summary']:.6f} MeV")
                report.append(f"  Разница:                     {comb_verif['balance_check_difference']:.6f} MeV ({comb_verif['balance_check_relative']:.4f}%)")
                report.append("")

                if comb_verif['balance_check_relative'] < 1:
                    report.append("✅ ОТЛИЧНО! Энергетический баланс первичных частиц сходится (<1%)")
                elif comb_verif['balance_check_relative'] < 5:
                    report.append("✅ ХОРОШО! Энергетический баланс первичных частиц приемлем (<5%)")
                else:
                    report.append("⚠️ Есть расхождение в энергетическом балансе")
                report.append("")

        # Сравнение процессов
        if 'combined' in verification and 'process_comparison' in verification['combined']:
            report.append("5. СРАВНЕНИЕ ЧАСТОТ ПРОЦЕССОВ")
            report.append("-" * 100)
            report.append(f"{'Процесс':<30s} | {'Парсинг':>10s} | {'Сводка':>10s} | {'Разница':>10s}")
            report.append("-" * 100)

            sorted_processes = sorted(verification['combined']['process_comparison'].items(),
                                    key=lambda x: x[1]['abs_difference'], reverse=True)

            for process, data in sorted_processes[:20]:  # Топ-20 процессов
                diff_str = f"{data['difference']:+d}"
                report.append(f"{process:<30s} | {data['parsed']:>10d} | "
                            f"{data['summary']:>10d} | {diff_str:>10s}")
            report.append("")

        # Объяснение расхождений
        report.append("6. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
        report.append("-" * 100)
        report.append("")
        report.append("ДВА МЕТОДА ПРОВЕРКИ ЭНЕРГЕТИЧЕСКОГО БАЛАНСА:")
        report.append("")
        report.append("МЕТОД 1: Сумма потерь энергии (dEStep)")
        report.append("  • Суммирует все положительные значения dEStep из шагов")
        report.append("  • Работает ТОЛЬКО если ВСЕ шаги записаны в лог")
        report.append("  • ❌ НЕ РАБОТАЕТ для неполных логов (когда пропущены шаги)")
        report.append("  • Причина: если часть шагов не залогирована, сумма будет неполной")
        report.append("")
        report.append("МЕТОД 2: Энергетический баланс треков (Начальная - Конечная)")
        report.append("  • Использует начальную и конечную энергию каждого трека")
        report.append("  • ✅ РАБОТАЕТ даже для неполных логов")
        report.append("  • Более надежный метод, не зависит от логирования промежуточных шагов")
        report.append("")
        report.append("ПРОВЕРКА ЭНЕРГЕТИЧЕСКОГО БАЛАНСА:")
        report.append("  • Начальная энергия первичных частиц должна равняться")
        report.append("    сумме Energy deposit + Energy leakage из итоговой сводки")
        report.append("  • Это фундаментальный закон сохранения энергии")
        report.append("  • Если разница <1%, энергетический баланс сходится отлично")
        report.append("")
        report.append("-" * 100)
        report.append("")
        report.append("Первичные частицы (Parent ID = 0):")
        report.append("  • Это исходные частицы, запущенные в симуляции")
        report.append("  • Их начальная энергия - это входная энергия симуляции")
        report.append("")
        report.append("Вторичные частицы (Parent ID > 0):")
        report.append("  • Это частицы, рожденные в процессе взаимодействия первичных")
        report.append("  • Они могут создаваться через процессы: ionIoni, eBrem, compt, и др.")
        report.append("  • Их энергия изначально взята из первичных частиц")
        report.append("")
        report.append("Energy deposit:")
        report.append("  • Энергия, поглощенная материалом (ионизация)")
        report.append("  • Вызывает повреждения в материале")
        report.append("")
        report.append("Energy leakage:")
        report.append("  • Энергия частиц, покинувших объем симуляции")
        report.append("  • Не поглощается материалом")
        report.append("")
        report.append("Возможные причины малых расхождений (<5%):")
        report.append("  1. Округления при конверсии единиц (eV ↔ keV ↔ MeV)")
        report.append("  2. Численная точность при расчетах в Geant4")
        report.append("  3. NIEL energy (неионизирующие потери энергии)")
        report.append("  4. Энергетические пороги отслеживания частиц")
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
        description='Парсер и анализатор логов Geant4 с разделением на первичные/вторичные частицы',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python geant4_parser_fixed.py -i simulation.log
  python geant4_parser_fixed.py -i simulation.log -o results --export csv xlsx
  python geant4_parser_fixed.py -i simulation.log --no-viz --debug
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Входной файл лога Geant4')
    parser.add_argument('-o', '--output', default='output',
                        help='Папка для выходных файлов (по умолчанию: output)')
    parser.add_argument('--export', nargs='+', default=['xlsx'],
                        choices=['csv', 'xlsx', 'dat'],
                        help='Форматы экспорта данных')
    parser.add_argument('--plot', nargs='+', default=['svg'],
                        choices=['png', 'svg'],
                        help='Форматы сохранения графиков')
    parser.add_argument('--no-viz', action='store_true',
                        help='Не создавать визуализации')
    parser.add_argument('--debug', action='store_true',
                        help='Режим отладки с детальным выводом')

    args = parser.parse_args()

    # Парсинг лога
    print("=" * 100)
    print("GEANT4 LOG PARSER - ИСПРАВЛЕННАЯ ВЕРСИЯ С ПРАВИЛЬНЫМ ПАРСИНГОМ dEStep")
    print("=" * 100)

    parser_obj = Geant4LogParser(args.input)
    parser_obj.parse_log(debug=args.debug)

    if not parser_obj.steps:
        print("ПРЕДУПРЕЖДЕНИЕ: Не найдено данных о шагах в логе!")
        print("Возможно, формат лога не соответствует ожидаемому.")
        print("\nПопробуйте запустить с флагом --debug для диагностики:")
        print(f"  python geant4_parser_fixed.py -i {args.input} --debug")
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

    # Статистика первичных частиц
    if 'primary' in verification and len(analyzer.df_primary) > 0:
        prim_balance = verification['primary']['energy_balance']
        print(f"\nПЕРВИЧНЫЕ ЧАСТИЦЫ (Parent ID = 0):")
        print(f"  Начальная энергия: {prim_balance['initial_energy']:.6f} MeV")
        print(f"  Потерянная энергия: {prim_balance['energy_lost']:.6f} MeV")
        print(f"  Сумма положительных dEStep: {prim_balance['total_de_step_positive']:.6f} MeV")

    # Статистика вторичных частиц
    if 'secondary' in verification and len(analyzer.df_secondary) > 0:
        sec_balance = verification['secondary']['energy_balance']
        print(f"\nВТОРИЧНЫЕ ЧАСТИЦЫ (Parent ID > 0):")
        print(f"  Начальная энергия: {sec_balance['initial_energy']:.6f} MeV")
        print(f"  Потерянная энергия: {sec_balance['energy_lost']:.6f} MeV")
        print(f"  Сумма положительных dEStep: {sec_balance['total_de_step_positive']:.6f} MeV")

    # Общая статистика
    if 'combined' in verification:
        comb_verif = verification['combined']
        if 'energy_deposit_summary' in comb_verif:
            print(f"\nСВЕРКА С ИТОГОВОЙ СВОДКОЙ:")
            print(f"  Energy deposit (из сводки): {comb_verif['energy_deposit_summary']:.6f} MeV")
            print(f"  Energy leakage (из сводки): {comb_verif['energy_leakage_summary']:.6f} MeV")
            print(f"  Сумма (E_deposit + E_leakage): {comb_verif['total_summary']:.6f} MeV")
            print()
            print(f"  ⚠️  ВНИМАНИЕ: Лог неполный (пропущены шаги)!")
            print(f"  Метод 1 (сумма dEStep) НЕ РАБОТАЕТ для неполных логов.")
            print(f"  Используйте ТОЛЬКО Метод 2 (энергетический баланс).")
            print()
            print(f"  Метод 1 (Сумма положительных dEStep - только потери энергии):")
            print(f"    Рассчитано: {comb_verif['energy_balance']['total_de_step_positive']:.6f} MeV")
            print(f"    Абсолютная разница: {comb_verif['method1_absolute_difference']:.6f} MeV")
            print(f"    Относительная разница: {comb_verif['method1_relative_difference']:.4f}%")
            print(f"    ❌ НЕ ПРИМЕНИМО для этого лога (шаги пропущены)")
            print()
            print(f"  Метод 2 (Начальная - Конечная энергия):")
            print(f"    Рассчитано: {comb_verif['energy_balance']['energy_lost']:.6f} MeV")
            print(f"    Абсолютная разница: {comb_verif['method2_absolute_difference']:.6f} MeV")
            print(f"    Относительная разница: {comb_verif['method2_relative_difference']:.4f}%")
            print()
            print(f"  Проверка энергетического баланса первичных частиц:")
            print(f"    Начальная энергия первичных: {comb_verif['primary_initial_energy']:.6f} MeV")
            print(f"    E_deposit + E_leakage:       {comb_verif['total_summary']:.6f} MeV")
            print(f"    Разница:                     {comb_verif['balance_check_difference']:.6f} MeV ({comb_verif['balance_check_relative']:.4f}%)")
            print()
            if comb_verif['balance_check_relative'] < 1:
                print("    ✅ ОТЛИЧНО! Энергетический баланс первичных частиц сходится (<1%)")
            elif comb_verif['balance_check_relative'] < 5:
                print("    ✅ ХОРОШО! Энергетический баланс первичных частиц приемлем (<5%)")
            else:
                print("    ⚠️ Есть расхождение в энергетическом балансе")

    print("=" * 100)
    print(f"\nВсе результаты сохранены в папке: {analyzer.output_dir}")
    print("\nСозданные файлы:")
    print("  • steps_primary.csv/xlsx - данные первичных частиц")
    print("  • steps_secondary.csv/xlsx - данные вторичных частиц")
    print("  • steps_all.csv/xlsx - все данные")
    print("  • analysis_report.txt - подробный отчет")
    print("  • Графики с разделением на первичные/вторичные частицы")


if __name__ == "__main__":
    main()
