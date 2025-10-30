"""
Geant4 Log Parser - Версия с разделением на первичные и вторичные частицы
Программа для парсинга, анализа и визуализации логов симуляции Geant4
с отдельным анализом первичных (Parent ID = 0) и вторичных (Parent ID > 0) частиц
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
        self.track_parent_ids: Dict[int, int] = {}  # Добавлено для отслеживания Parent ID

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

        if debug and debug_lines_sample:
            print(f"\n[DEBUG] Примеры строк из лога:")
            for line_num, line_text in debug_lines_sample[:10]:
                print(f"  [{line_num}] {line_text[:100]}")

    def _parse_step_line_simple(self, line: str, line_num: int, debug: bool = False) -> Optional[StepData]:
        """Упрощенный парсинг строки шага с большей гибкостью"""
        if not line.strip() or 'Step#' in line:
            return None

        # Проверка наличия префикса потока
        if not re.search(self.THREAD_PATTERN, line):
            return None

        parts = line.split()
        if len(parts) < 8:
            return None

        try:
            # Пытаемся найти номер шага
            step_num_idx = None
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:  # пропускаем G4WT
                    step_num_idx = i
                    break

            if step_num_idx is None:
                return None

            step_num = int(parts[step_num_idx])

            # Ищем координаты (3 числа подряд с единицами mm, um и т.д.)
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

            # Ищем энергию KineE (после координат)
            kine_e = 0.0
            energy_unit = 'MeV'
            for i in range(step_num_idx + 7, min(len(parts) - 1, step_num_idx + 15)):
                try:
                    val = float(parts[i])
                    unit = parts[i + 1] if i + 1 < len(parts) else 'MeV'
                    if unit in self.ENERGY_UNITS or unit in ['eV', 'keV', 'MeV', 'GeV']:
                        kine_e = self.convert_energy(val, unit)
                        energy_unit = unit
                        break
                except (ValueError, IndexError):
                    continue

            # Ищем dEStep (обычно следующее значение энергии)
            de_step = 0.0
            for i in range(step_num_idx + 7, min(len(parts) - 1, step_num_idx + 20)):
                try:
                    val = float(parts[i])
                    unit = parts[i + 1] if i + 1 < len(parts) else 'eV'
                    if unit in self.ENERGY_UNITS and i > step_num_idx + 9:  # пропускаем KineE
                        de_step = self.convert_energy(val, unit)
                        break
                except (ValueError, IndexError):
                    continue

            # Ищем длины шагов
            step_leng = 0.0
            trak_leng = 0.0
            length_found = 0
            for i in range(step_num_idx + 10, min(len(parts) - 1, step_num_idx + 25)):
                try:
                    val = float(parts[i])
                    unit = parts[i + 1] if i + 1 < len(parts) else 'mm'
                    if unit in self.LENGTH_UNITS:
                        if length_found == 0:
                            step_leng = self.convert_length(val, unit)
                            length_found += 1
                        elif length_found == 1:
                            trak_leng = self.convert_length(val, unit)
                            break
                except (ValueError, IndexError):
                    continue

            # Ищем Volume и Process (обычно в конце строки)
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
                'is_primary': step.is_primary  # Добавлено новое поле
            })

        return pd.DataFrame(data)


class Geant4Analyzer:
    """Класс для анализа и визуализации данных"""

    def __init__(self, df: pd.DataFrame, summary: Dict, output_dir: str = 'output',
                 input_filename: str = 'simulation.log'):
        self.df = df
        self.summary = summary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.input_filename = input_filename

        # Разделяем данные на первичные и вторичные
        self.df_primary = df[df['is_primary'] == True].copy()
        self.df_secondary = df[df['is_primary'] == False].copy()

        print(f"\n{'='*100}")
        print("РАЗДЕЛЕНИЕ НА ПЕРВИЧНЫЕ И ВТОРИЧНЫЕ ЧАСТИЦЫ")
        print(f"{'='*100}")
        print(f"Первичные частицы (Parent ID = 0): {len(self.df_primary)} шагов")
        print(f"Вторичные частицы (Parent ID > 0): {len(self.df_secondary)} шагов")

        if len(self.df_primary) > 0:
            print(f"\nТипы первичных частиц: {', '.join(self.df_primary['particle'].unique())}")
        if len(self.df_secondary) > 0:
            print(f"Типы вторичных частиц: {', '.join(self.df_secondary['particle'].unique())}")

    def aggregate_data(self) -> Dict:
        """Агрегация данных отдельно для первичных и вторичных частиц"""
        results = {
            'primary': {},
            'secondary': {},
            'combined': {}
        }

        # Агрегация для первичных частиц
        if len(self.df_primary) > 0:
            results['primary'] = self._aggregate_dataframe(self.df_primary, "ПЕРВИЧНЫЕ")

        # Агрегация для вторичных частиц
        if len(self.df_secondary) > 0:
            results['secondary'] = self._aggregate_dataframe(self.df_secondary, "ВТОРИЧНЫЕ")

        # Общая агрегация
        results['combined'] = self._aggregate_dataframe(self.df, "ОБЩИЕ")

        return results

    def _aggregate_dataframe(self, df: pd.DataFrame, category: str) -> Dict:
        """Вспомогательная функция для агрегации DataFrame"""
        print(f"\n{category} ЧАСТИЦЫ - Агрегация данных...")

        agg = {}

        # По типам частиц
        agg['by_particle'] = df.groupby('particle').agg({
            'de_step_MeV': ['sum', 'mean', 'std', 'count'],
            'kine_e_MeV': ['mean', 'min', 'max'],
            'step_leng_mm': ['mean', 'sum'],
            'track_id': 'nunique'
        }).round(6)

        # По процессам
        agg['by_process'] = df.groupby('process').agg({
            'de_step_MeV': 'sum',
            'process': 'count'
        }).rename(columns={'process': 'count'}).round(6)

        # По трекам
        agg['by_track'] = df.groupby('track_id').agg({
            'de_step_MeV': 'sum',
            'kine_e_MeV': ['first', 'last'],
            'step_num': 'count',
            'particle': 'first',
            'parent_id': 'first'
        }).round(6)

        # По потокам
        agg['by_thread'] = df.groupby('thread').agg({
            'de_step_MeV': 'sum',
            'track_id': 'nunique',
            'step_num': 'count'
        }).round(6)

        # Общая статистика
        agg['overall'] = {
            'total_steps': len(df),
            'total_tracks': df['track_id'].nunique(),
            'total_de_MeV': df['de_step_MeV'].sum(),
            'total_de_positive_MeV': df[df['de_step_MeV'] > 0]['de_step_MeV'].sum(),
            'mean_kine_e_MeV': df['kine_e_MeV'].mean(),
            'total_distance_mm': df['step_leng_mm'].sum()
        }

        print(f"  {category}: Обработано {agg['overall']['total_steps']} шагов, "
              f"{agg['overall']['total_tracks']} треков")

        return agg

    def export_data(self, formats: List[str] = ['csv', 'xlsx']) -> None:
        """Экспорт данных в различные форматы с разделением на первичные/вторичные"""
        print("\nЭкспорт данных...")

        # Экспорт первичных частиц
        if len(self.df_primary) > 0:
            if 'csv' in formats:
                csv_path = self.output_dir / 'steps_primary.csv'
                self.df_primary.to_csv(csv_path, index=False)
                print(f"  Сохранено: {csv_path}")

            if 'xlsx' in formats:
                xlsx_path = self.output_dir / 'steps_primary.xlsx'
                self.df_primary.to_excel(xlsx_path, index=False, engine='openpyxl')
                print(f"  Сохранено: {xlsx_path}")

        # Экспорт вторичных частиц
        if len(self.df_secondary) > 0:
            if 'csv' in formats:
                csv_path = self.output_dir / 'steps_secondary.csv'
                self.df_secondary.to_csv(csv_path, index=False)
                print(f"  Сохранено: {csv_path}")

            if 'xlsx' in formats:
                xlsx_path = self.output_dir / 'steps_secondary.xlsx'
                self.df_secondary.to_excel(xlsx_path, index=False, engine='openpyxl')
                print(f"  Сохранено: {xlsx_path}")

        # Экспорт всех данных
        if 'csv' in formats:
            csv_path = self.output_dir / 'steps_all.csv'
            self.df.to_csv(csv_path, index=False)
            print(f"  Сохранено: {csv_path}")

        if 'xlsx' in formats:
            xlsx_path = self.output_dir / 'steps_all.xlsx'
            self.df.to_excel(xlsx_path, index=False, engine='openpyxl')
            print(f"  Сохранено: {xlsx_path}")

    def create_visualizations(self, save_formats: List[str] = ['png', 'svg']) -> None:
        """Создание всех визуализаций с разделением на первичные/вторичные"""
        print("\nСоздание визуализаций...")

        # 1. Распределение кинетической энергии
        self._plot_energy_distributions(save_formats)

        # 2. Распределение потерь энергии
        self._plot_de_distributions(save_formats)

        # 3. Частоты процессов
        self._plot_process_frequencies(save_formats)

        # 4. Пространственное распределение
        self._plot_spatial_distributions(save_formats)

        # 5. Энергетический баланс
        self._plot_energy_balance(save_formats)

        # 6. Сравнение первичных и вторичных
        self._plot_primary_vs_secondary(save_formats)

    def _plot_energy_distributions(self, save_formats: List[str]) -> None:
        """График распределения кинетической энергии для первичных и вторичных частиц"""
        # Получаем уникальные типы частиц из обеих категорий
        all_particles = set(self.df['particle'].unique())

        for particle in all_particles:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Распределение кинетической энергии - {particle}',
                        fontsize=14, fontweight='bold')

            # Первичные частицы
            df_prim_part = self.df_primary[self.df_primary['particle'] == particle]
            if len(df_prim_part) > 0:
                axes[0].hist(df_prim_part['kine_e_MeV'], bins=50,
                           color='blue', alpha=0.7, edgecolor='black')
                axes[0].set_xlabel('Кинетическая энергия (MeV)')
                axes[0].set_ylabel('Частота')
                axes[0].set_title(f'ПЕРВИЧНЫЕ частицы (Parent ID = 0)')

                stats_text = (f"Шагов: {len(df_prim_part)}\n"
                            f"Min: {df_prim_part['kine_e_MeV'].min():.6f} MeV\n"
                            f"Max: {df_prim_part['kine_e_MeV'].max():.6f} MeV\n"
                            f"Mean: {df_prim_part['kine_e_MeV'].mean():.6f} MeV")
                axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                axes[0].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                           transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title(f'ПЕРВИЧНЫЕ частицы (Parent ID = 0)')

            # Вторичные частицы
            df_sec_part = self.df_secondary[self.df_secondary['particle'] == particle]
            if len(df_sec_part) > 0:
                axes[1].hist(df_sec_part['kine_e_MeV'], bins=50,
                           color='red', alpha=0.7, edgecolor='black')
                axes[1].set_xlabel('Кинетическая энергия (MeV)')
                axes[1].set_ylabel('Частота')
                axes[1].set_title(f'ВТОРИЧНЫЕ частицы (Parent ID > 0)')

                stats_text = (f"Шагов: {len(df_sec_part)}\n"
                            f"Min: {df_sec_part['kine_e_MeV'].min():.6f} MeV\n"
                            f"Max: {df_sec_part['kine_e_MeV'].max():.6f} MeV\n"
                            f"Mean: {df_sec_part['kine_e_MeV'].mean():.6f} MeV")
                axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                axes[1].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title(f'ВТОРИЧНЫЕ частицы (Parent ID > 0)')

            plt.tight_layout()
            for fmt in save_formats:
                filepath = self.output_dir / f'energy_distribution_{particle}.{fmt}'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  ✓ Графики распределения энергии созданы")

    def _plot_de_distributions(self, save_formats: List[str]) -> None:
        """Boxplot распределения потерь энергии для первичных и вторичных"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Распределение потерь энергии на шаг (dE)',
                    fontsize=14, fontweight='bold')

        # Первичные частицы
        if len(self.df_primary) > 0:
            df_prim_nonzero = self.df_primary[self.df_primary['de_step_MeV'] > 0]
            if len(df_prim_nonzero) > 0:
                particles_prim = df_prim_nonzero['particle'].unique()
                data_prim = [df_prim_nonzero[df_prim_nonzero['particle'] == p]['de_step_MeV'].values
                           for p in particles_prim]

                bp = axes[0].boxplot(data_prim, labels=particles_prim, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')

                axes[0].set_ylabel('dE (MeV)')
                axes[0].set_title('ПЕРВИЧНЫЕ частицы (Parent ID = 0)')
                axes[0].tick_params(axis='x', rotation=45)
            else:
                axes[0].text(0.5, 0.5, 'Нет данных с dE > 0', ha='center', va='center',
                           transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title('ПЕРВИЧНЫЕ частицы (Parent ID = 0)')
        else:
            axes[0].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                       transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('ПЕРВИЧНЫЕ частицы (Parent ID = 0)')

        # Вторичные частицы
        if len(self.df_secondary) > 0:
            df_sec_nonzero = self.df_secondary[self.df_secondary['de_step_MeV'] > 0]
            if len(df_sec_nonzero) > 0:
                particles_sec = df_sec_nonzero['particle'].unique()
                data_sec = [df_sec_nonzero[df_sec_nonzero['particle'] == p]['de_step_MeV'].values
                          for p in particles_sec]

                bp = axes[1].boxplot(data_sec, labels=particles_sec, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightcoral')

                axes[1].set_ylabel('dE (MeV)')
                axes[1].set_title('ВТОРИЧНЫЕ частицы (Parent ID > 0)')
                axes[1].tick_params(axis='x', rotation=45)
            else:
                axes[1].text(0.5, 0.5, 'Нет данных с dE > 0', ha='center', va='center',
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title('ВТОРИЧНЫЕ частицы (Parent ID > 0)')
        else:
            axes[1].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                       transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('ВТОРИЧНЫЕ частицы (Parent ID > 0)')

        plt.tight_layout()
        for fmt in save_formats:
            filepath = self.output_dir / f'de_distribution_boxplot.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Boxplot распределения dE создан")

    def _plot_process_frequencies(self, save_formats: List[str]) -> None:
        """Диаграмма частоты процессов для первичных и вторичных частиц"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('Частота процессов', fontsize=14, fontweight='bold')

        # Первичные частицы
        if len(self.df_primary) > 0:
            process_counts_prim = self.df_primary['process'].value_counts().sort_values(ascending=True)
            if len(process_counts_prim) > 0:
                colors_prim = plt.cm.Blues(np.linspace(0.4, 0.8, len(process_counts_prim)))
                axes[0].barh(process_counts_prim.index, process_counts_prim.values, color=colors_prim)
                axes[0].set_xlabel('Количество вызовов')
                axes[0].set_title('ПЕРВИЧНЫЕ частицы (Parent ID = 0)')
                axes[0].grid(axis='x', alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, 'Нет данных о процессах', ha='center', va='center',
                           transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title('ПЕРВИЧНЫЕ частицы (Parent ID = 0)')
        else:
            axes[0].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                       transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('ПЕРВИЧНЫЕ частицы (Parent ID = 0)')

        # Вторичные частицы
        if len(self.df_secondary) > 0:
            process_counts_sec = self.df_secondary['process'].value_counts().sort_values(ascending=True)
            if len(process_counts_sec) > 0:
                colors_sec = plt.cm.Reds(np.linspace(0.4, 0.8, len(process_counts_sec)))
                axes[1].barh(process_counts_sec.index, process_counts_sec.values, color=colors_sec)
                axes[1].set_xlabel('Количество вызовов')
                axes[1].set_title('ВТОРИЧНЫЕ частицы (Parent ID > 0)')
                axes[1].grid(axis='x', alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Нет данных о процессах', ha='center', va='center',
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title('ВТОРИЧНЫЕ частицы (Parent ID > 0)')
        else:
            axes[1].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                       transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('ВТОРИЧНЫЕ частицы (Parent ID > 0)')

        plt.tight_layout()
        for fmt in save_formats:
            filepath = self.output_dir / f'process_frequencies.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Диаграмма частоты процессов создана")

    def _plot_spatial_distributions(self, save_formats: List[str]) -> None:
        """Heatmap пространственного распределения для первичных и вторичных"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Пространственное распределение частиц', fontsize=14, fontweight='bold')

        # Функция для создания 2D heatmap
        def plot_2d_hist(ax, x, y, title, cmap='Blues'):
            if len(x) > 0 and len(y) > 0:
                h, xedges, yedges = np.histogram2d(x, y, bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(h.T, extent=extent, origin='lower', cmap=cmap, aspect='auto')
                plt.colorbar(im, ax=ax, label='Количество шагов')
            ax.set_title(title)
            return ax

        # Первичные частицы - XY
        ax1 = fig.add_subplot(gs[0, 0])
        plot_2d_hist(ax1, self.df_primary['x_mm'], self.df_primary['y_mm'],
                    'ПЕРВИЧНЫЕ: XY проекция', cmap='Blues')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')

        # Вторичные частицы - XY
        ax2 = fig.add_subplot(gs[0, 1])
        plot_2d_hist(ax2, self.df_secondary['x_mm'], self.df_secondary['y_mm'],
                    'ВТОРИЧНЫЕ: XY проекция', cmap='Reds')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')

        # Первичные частицы - XZ
        ax3 = fig.add_subplot(gs[1, 0])
        plot_2d_hist(ax3, self.df_primary['x_mm'], self.df_primary['z_mm'],
                    'ПЕРВИЧНЫЕ: XZ проекция', cmap='Blues')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Z (mm)')

        # Вторичные частицы - XZ
        ax4 = fig.add_subplot(gs[1, 1])
        plot_2d_hist(ax4, self.df_secondary['x_mm'], self.df_secondary['z_mm'],
                    'ВТОРИЧНЫЕ: XZ проекция', cmap='Reds')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Z (mm)')

        # Первичные частицы - YZ
        ax5 = fig.add_subplot(gs[2, 0])
        plot_2d_hist(ax5, self.df_primary['y_mm'], self.df_primary['z_mm'],
                    'ПЕРВИЧНЫЕ: YZ проекция', cmap='Blues')
        ax5.set_xlabel('Y (mm)')
        ax5.set_ylabel('Z (mm)')

        # Вторичные частицы - YZ
        ax6 = fig.add_subplot(gs[2, 1])
        plot_2d_hist(ax6, self.df_secondary['y_mm'], self.df_secondary['z_mm'],
                    'ВТОРИЧНЫЕ: YZ проекция', cmap='Reds')
        ax6.set_xlabel('Y (mm)')
        ax6.set_ylabel('Z (mm)')

        for fmt in save_formats:
            filepath = self.output_dir / f'spatial_distributions.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Пространственные распределения созданы")

    def _plot_energy_balance(self, save_formats: List[str]) -> None:
        """График энергетического баланса отдельно для первичных и вторичных"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Энергетический баланс', fontsize=14, fontweight='bold')

        # Первичные частицы
        if len(self.df_primary) > 0:
            primary_de_by_particle = self.df_primary.groupby('particle')['de_step_MeV'].sum()
            # Фильтруем положительные значения для pie chart
            primary_de_positive = primary_de_by_particle[primary_de_by_particle > 0]

            if len(primary_de_positive) > 0 and primary_de_positive.sum() > 0:
                colors_prim = plt.cm.Blues(np.linspace(0.4, 0.8, len(primary_de_positive)))
                axes[0].pie(primary_de_positive, labels=primary_de_positive.index,
                           autopct='%1.1f%%', colors=colors_prim, startangle=90)
                axes[0].set_title(f'ПЕРВИЧНЫЕ частицы\nВсего: {primary_de_positive.sum():.6f} MeV')
            else:
                axes[0].text(0.5, 0.5, 'Нет положительных\nпотерь энергии',
                           ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title(f'ПЕРВИЧНЫЕ частицы\nВсего: 0.000000 MeV')
        else:
            axes[0].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                       transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('ПЕРВИЧНЫЕ частицы')

        # Вторичные частицы
        if len(self.df_secondary) > 0:
            secondary_de_by_particle = self.df_secondary.groupby('particle')['de_step_MeV'].sum()
            # Фильтруем положительные значения для pie chart
            secondary_de_positive = secondary_de_by_particle[secondary_de_by_particle > 0]

            if len(secondary_de_positive) > 0 and secondary_de_positive.sum() > 0:
                colors_sec = plt.cm.Reds(np.linspace(0.4, 0.8, len(secondary_de_positive)))
                axes[1].pie(secondary_de_positive, labels=secondary_de_positive.index,
                           autopct='%1.1f%%', colors=colors_sec, startangle=90)
                axes[1].set_title(f'ВТОРИЧНЫЕ частицы\nВсего: {secondary_de_positive.sum():.6f} MeV')
            else:
                axes[1].text(0.5, 0.5, 'Нет положительных\nпотерь энергии',
                           ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title(f'ВТОРИЧНЫЕ частицы\nВсего: 0.000000 MeV')
        else:
            axes[1].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                       transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('ВТОРИЧНЫЕ частицы')

        plt.tight_layout()
        for fmt in save_formats:
            filepath = self.output_dir / f'energy_balance_pie.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ График энергетического баланса создан")

    def _plot_primary_vs_secondary(self, save_formats: List[str]) -> None:
        """Сравнительные графики первичных и вторичных частиц"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Сравнение первичных и вторичных частиц',
                    fontsize=14, fontweight='bold')

        # 1. Общая статистика по энергии
        categories = ['Первичные', 'Вторичные']
        total_de = [
            self.df_primary['de_step_MeV'].sum() if len(self.df_primary) > 0 else 0,
            self.df_secondary['de_step_MeV'].sum() if len(self.df_secondary) > 0 else 0
        ]

        axes[0, 0].bar(categories, total_de, color=['blue', 'red'], alpha=0.7)
        axes[0, 0].set_ylabel('Суммарная потеря энергии (MeV)')
        axes[0, 0].set_title('Общая потеря энергии')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Количество шагов
        step_counts = [len(self.df_primary), len(self.df_secondary)]
        axes[0, 1].bar(categories, step_counts, color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_ylabel('Количество шагов')
        axes[0, 1].set_title('Количество шагов')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Средняя кинетическая энергия
        mean_ke = [
            self.df_primary['kine_e_MeV'].mean() if len(self.df_primary) > 0 else 0,
            self.df_secondary['kine_e_MeV'].mean() if len(self.df_secondary) > 0 else 0
        ]
        axes[1, 0].bar(categories, mean_ke, color=['blue', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Средняя кин. энергия (MeV)')
        axes[1, 0].set_title('Средняя кинетическая энергия')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Количество уникальных треков
        track_counts = [
            self.df_primary['track_id'].nunique() if len(self.df_primary) > 0 else 0,
            self.df_secondary['track_id'].nunique() if len(self.df_secondary) > 0 else 0
        ]
        axes[1, 1].bar(categories, track_counts, color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('Количество треков')
        axes[1, 1].set_title('Уникальные треки')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        for fmt in save_formats:
            filepath = self.output_dir / f'primary_vs_secondary_comparison.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Сравнительные графики созданы")

    def verify_results(self, parser: Geant4LogParser) -> Dict:
        """Верификация результатов отдельно для первичных и вторичных"""
        verification = {
            'primary': {},
            'secondary': {},
            'combined': {}
        }

        # Верификация первичных частиц
        if len(self.df_primary) > 0:
            verification['primary'] = self._verify_dataframe(self.df_primary, parser, "ПЕРВИЧНЫЕ")

        # Верификация вторичных частиц
        if len(self.df_secondary) > 0:
            verification['secondary'] = self._verify_dataframe(self.df_secondary, parser, "ВТОРИЧНЫЕ")

        # Общая верификация
        verification['combined'] = self._verify_dataframe(self.df, parser, "ОБЩИЕ")

        return verification

    def _verify_dataframe(self, df: pd.DataFrame, parser: Geant4LogParser, category: str) -> Dict:
        """Вспомогательная функция для верификации DataFrame"""
        result = {}

        # Энергетический баланс
        total_de_step_all = df['de_step_MeV'].sum()
        total_de_step_positive = df[df['de_step_MeV'] > 0]['de_step_MeV'].sum()

        # Начальная энергия треков
        unique_tracks = df['track_id'].unique()
        initial_energy = sum(parser.track_initial_energy.get(tid, 0) for tid in unique_tracks)
        final_energy = sum(parser.track_final_energy.get(tid, 0) for tid in unique_tracks)
        energy_lost = initial_energy - final_energy

        result['energy_balance'] = {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_lost': energy_lost,
            'total_de_step_all': total_de_step_all,
            'total_de_step_positive': total_de_step_positive
        }

        # Сверка с итоговой сводкой (только для общих данных)
        if category == "ОБЩИЕ" and 'energy_deposit' in parser.summary:
            energy_deposit_summary = parser.summary.get('energy_deposit', 0)
            result['energy_deposit_summary'] = energy_deposit_summary

            # Метод 1: dEStep
            abs_diff_de = abs(total_de_step_positive - energy_deposit_summary)
            rel_diff_de = (abs_diff_de / energy_deposit_summary * 100) if energy_deposit_summary > 0 else 0

            result['absolute_difference_de'] = abs_diff_de
            result['relative_difference_de'] = rel_diff_de

            # Метод 2: Баланс энергии
            abs_diff_calc = abs(energy_lost - energy_deposit_summary)
            rel_diff_calc = (abs_diff_calc / energy_deposit_summary * 100) if energy_deposit_summary > 0 else 0

            result['absolute_difference_calc'] = abs_diff_calc
            result['relative_difference_calc'] = rel_diff_calc

        # Сверка процессов
        process_counts = df['process'].value_counts().to_dict()
        summary_processes = parser.summary.get('process_frequencies', {})

        process_comparison = {}
        all_processes = set(process_counts.keys()) | set(summary_processes.keys())

        for process in all_processes:
            parsed = process_counts.get(process, 0)
            summary = summary_processes.get(process, 0)
            process_comparison[process] = {
                'parsed': parsed,
                'summary': summary,
                'difference': parsed - summary,
                'abs_difference': abs(parsed - summary)
            }

        result['process_comparison'] = process_comparison

        return result

    def generate_report(self, parser: Geant4LogParser) -> str:
        """Генерация текстового отчета с разделением на первичные/вторичные"""
        report = []
        report.append("=" * 100)
        report.append("ОТЧЕТ АНАЛИЗА ЛОГОВ GEANT4 - С РАЗДЕЛЕНИЕМ НА ПЕРВИЧНЫЕ И ВТОРИЧНЫЕ ЧАСТИЦЫ")
        report.append("=" * 100)
        report.append(f"Входной файл: {self.input_filename}")
        report.append(f"Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Общая статистика
        report.append("1. ОБЩАЯ СТАТИСТИКА")
        report.append("-" * 100)
        report.append(f"Всего шагов обработано: {len(self.df)}")
        report.append(f"  → Первичные частицы (Parent ID = 0): {len(self.df_primary)}")
        report.append(f"  → Вторичные частицы (Parent ID > 0): {len(self.df_secondary)}")
        report.append(f"Уникальных треков: {self.df['track_id'].nunique()}")
        report.append(f"  → Первичные треки: {self.df_primary['track_id'].nunique() if len(self.df_primary) > 0 else 0}")
        report.append(f"  → Вторичные треки: {self.df_secondary['track_id'].nunique() if len(self.df_secondary) > 0 else 0}")
        report.append(f"Уникальных потоков: {self.df['thread'].nunique()}")
        report.append("")

        # Типы частиц
        report.append("2. ТИПЫ ЧАСТИЦ")
        report.append("-" * 100)

        if len(self.df_primary) > 0:
            report.append("ПЕРВИЧНЫЕ частицы:")
            for particle in self.df_primary['particle'].unique():
                count = len(self.df_primary[self.df_primary['particle'] == particle])
                report.append(f"  • {particle}: {count} шагов")

        if len(self.df_secondary) > 0:
            report.append("\nВТОРИЧНЫЕ частицы:")
            for particle in self.df_secondary['particle'].unique():
                count = len(self.df_secondary[self.df_secondary['particle'] == particle])
                report.append(f"  • {particle}: {count} шагов")

        report.append("")

        # Энергетический баланс
        verification = self.verify_results(parser)

        report.append("3. ЭНЕРГЕТИЧЕСКИЙ БАЛАНС")
        report.append("-" * 100)

        # Первичные частицы
        if 'primary' in verification and len(self.df_primary) > 0:
            prim_balance = verification['primary']['energy_balance']
            report.append("ПЕРВИЧНЫЕ частицы (Parent ID = 0):")
            report.append(f"  Начальная энергия: {prim_balance['initial_energy']:.6f} MeV")
            report.append(f"  Конечная энергия: {prim_balance['final_energy']:.6f} MeV")
            report.append(f"  Потерянная энергия: {prim_balance['energy_lost']:.6f} MeV")
            report.append(f"  Сумма всех dEStep: {prim_balance['total_de_step_all']:.6f} MeV")
            report.append(f"  Сумма положительных dEStep: {prim_balance['total_de_step_positive']:.6f} MeV")
            report.append("")

        # Вторичные частицы
        if 'secondary' in verification and len(self.df_secondary) > 0:
            sec_balance = verification['secondary']['energy_balance']
            report.append("ВТОРИЧНЫЕ частицы (Parent ID > 0):")
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
                report.append(f"Energy deposit (из лога): {comb_verif['energy_deposit_summary']:.6f} MeV")
                report.append(f"Сумма положительных dEStep (расчет): {comb_verif['energy_balance']['total_de_step_positive']:.6f} MeV")
                report.append(f"Абсолютная разница: {comb_verif['absolute_difference_de']:.6f} MeV")
                report.append(f"Относительная разница: {comb_verif['relative_difference_de']:.2f}%")
                report.append("")

                if comb_verif['relative_difference_de'] < 5:
                    report.append("✅ ОТЛИЧНО! Расхождение менее 5%")
                elif comb_verif['relative_difference_de'] < 20:
                    report.append("✅ ХОРОШО! Расхождение менее 20%")
                else:
                    report.append("⚠️ ВНИМАНИЕ! Расхождение более 20%")
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
        report.append("Первичные частицы (Parent ID = 0):")
        report.append("  • Это исходные частицы, запущенные в симуляции")
        report.append("  • Их энергетический баланс отражает прямые потери энергии")
        report.append("")
        report.append("Вторичные частицы (Parent ID > 0):")
        report.append("  • Это частицы, рожденные в процессе взаимодействия первичных")
        report.append("  • Они могут создаваться через процессы: ionIoni, eBrem, compt, и др.")
        report.append("  • Их энергия изначально взята из первичных частиц")
        report.append("")
        report.append("Возможные причины расхождений:")
        report.append("  1. Неполное логирование (не все шаги записываются)")
        report.append("  2. Энергетические пороги (частицы ниже порога не отслеживаются)")
        report.append("  3. Граничные эффекты (частицы покидают объем)")
        report.append("  4. Округления при конверсии единиц")
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
  python geant4_parser_primary_secondary.py -i simulation.log
  python geant4_parser_primary_secondary.py -i simulation.log -o results --export csv xlsx
  python geant4_parser_primary_secondary.py -i simulation.log --no-viz --debug
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
    print("GEANT4 LOG PARSER - ВЕРСИЯ С РАЗДЕЛЕНИЕМ НА ПЕРВИЧНЫЕ/ВТОРИЧНЫЕ ЧАСТИЦЫ")
    print("=" * 100)

    parser_obj = Geant4LogParser(args.input)
    parser_obj.parse_log(debug=args.debug)

    if not parser_obj.steps:
        print("ПРЕДУПРЕЖДЕНИЕ: Не найдено данных о шагах в логе!")
        print("Возможно, формат лога не соответствует ожидаемому.")
        print("\nПопробуйте запустить с флагом --debug для диагностики:")
        print(f"  python geant4_parser_primary_secondary.py -i {args.input} --debug")
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
            print(f"  Energy deposit (сводка): {comb_verif['energy_deposit_summary']:.6f} MeV")
            print(f"  Относительная разница: {comb_verif['relative_difference_de']:.4f}%")

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
