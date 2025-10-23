"""
Geant4 Log Parser - GUI версия
Программа для парсинга, анализа и визуализации логов симуляции Geant4
с графическим интерфейсом
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
import warnings
warnings.filterwarnings('ignore')

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import os
import subprocess
import sys

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

        print(f"Парсинг завершен. Найдено {len(self.steps)} шагов.")
        if debug and debug_lines_sample:
            print("\n[DEBUG] Примеры первых строк без Step#:")
            for i, line in debug_lines_sample[:10]:
                print(f"  Строка {i}: {line[:100]}")

    def _parse_step_line_simple(self, line: str, line_num: int, debug: bool = False) -> Optional[StepData]:
        """Упрощенный парсинг строки шага используя разделение по пробелам"""
        if not self.current_thread:
            return None

        # Убираем префикс потока
        content = re.sub(self.THREAD_PATTERN, '', line).strip()
        if not content or content == 'Step#':
            return None

        # Разделяем строку на части
        parts = content.split()
        if len(parts) < 10:
            return None

        try:
            # Пытаемся определить номер шага
            step_num_str = parts[0]
            if not step_num_str.replace('.', '').replace('-', '').isdigit():
                return None

            step_num = int(float(step_num_str))

            # Парсинг координат и единиц
            def parse_value_unit(val_str: str, unit_str: str, converter_func) -> float:
                """Парсит значение и единицу"""
                try:
                    value = float(val_str)
                    return converter_func(value, unit_str)
                except:
                    return 0.0

            # X, Y, Z с единицами
            x = parse_value_unit(parts[1], parts[2], self.convert_length)
            y = parse_value_unit(parts[3], parts[4], self.convert_length)
            z = parse_value_unit(parts[5], parts[6], self.convert_length)

            # Кинетическая энергия
            kine_e = parse_value_unit(parts[7], parts[8], self.convert_energy)

            # dEStep
            de_step = parse_value_unit(parts[9], parts[10], self.convert_energy)

            # StepLeng
            step_leng = parse_value_unit(parts[11], parts[12], self.convert_length)

            # TrakLeng
            trak_leng = parse_value_unit(parts[13], parts[14], self.convert_length)

            # Volume и Process
            volume = parts[15] if len(parts) > 15 else "Unknown"
            process = parts[16] if len(parts) > 16 else "Unknown"

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
                print(f"[DEBUG] Ошибка парсинга строки {line_num}: {str(e)}")
                print(f"  Содержимое: {content[:100]}")
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
                'x_mm': step.x,
                'y_mm': step.y,
                'z_mm': step.z,
                'kine_e_mev': step.kine_e,
                'de_step_mev': step.de_step,
                'step_leng_mm': step.step_leng,
                'trak_leng_mm': step.trak_leng,
                'volume': step.volume,
                'process': step.process,
                'track_id': step.track_id,
                'parent_id': step.parent_id,
                'particle': step.particle
            })

        return pd.DataFrame(data)

    def get_statistics(self) -> Dict:
        """Получение базовой статистики"""
        stats = {
            'total_steps': len(self.steps),
            'unique_threads': len(set(s.thread for s in self.steps)),
            'unique_particles': len(set(s.particle for s in self.steps)),
            'unique_processes': len(set(s.process for s in self.steps)),
            'unique_tracks': len(set(s.track_id for s in self.steps)),
        }
        return stats


class Geant4Analyzer:
    """Класс для анализа и визуализации данных"""

    def __init__(self, df: pd.DataFrame, summary: Dict, output_dir: str = 'output',
                 input_filename: str = None):
        self.df = df
        self.summary = summary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.input_filename = input_filename

    def aggregate_data(self) -> Dict:
        """Агрегация данных по частицам, трекам и потокам"""
        print("\nАгрегация данных:")

        # По типам частиц
        by_particle = self.df.groupby('particle').agg({
            'de_step_mev': ['sum', 'mean', 'std', 'count'],
            'kine_e_mev': ['mean', 'min', 'max'],
            'step_leng_mm': 'sum',
            'track_id': 'nunique'
        }).round(6)
        print(f"  - По типам частиц: {len(by_particle)} типов")

        # По процессам
        by_process = self.df.groupby('process').agg({
            'de_step_mev': ['sum', 'count'],
            'kine_e_mev': 'mean'
        }).round(6)
        print(f"  - По процессам: {len(by_process)} процессов")

        # По трекам
        by_track = self.df.groupby('track_id').agg({
            'particle': 'first',
            'parent_id': 'first',
            'de_step_mev': 'sum',
            'kine_e_mev': ['first', 'last'],
            'step_num': 'count',
            'trak_leng_mm': 'max'
        }).round(6)
        print(f"  - По трекам: {len(by_track)} треков")

        # По потокам
        by_thread = self.df.groupby('thread').agg({
            'de_step_mev': 'sum',
            'track_id': 'nunique',
            'step_num': 'count'
        }).round(6)
        print(f"  - По потокам: {len(by_thread)} потоков")

        return {
            'by_particle': by_particle,
            'by_process': by_process,
            'by_track': by_track,
            'by_thread': by_thread
        }

    def export_data(self, formats: List[str] = ['csv', 'xlsx']) -> None:
        """Экспорт данных в различные форматы"""
        print("\nЭкспорт данных:")

        # Основной DataFrame
        if 'csv' in formats:
            csv_path = self.output_dir / 'steps_data.csv'
            self.df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"  - CSV: {csv_path}")

        if 'xlsx' in formats:
            xlsx_path = self.output_dir / 'analysis_results.xlsx'
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name='Steps', index=False)

                # Агрегированные данные
                agg_results = self.aggregate_data()
                agg_results['by_particle'].to_excel(writer, sheet_name='By_Particle')
                agg_results['by_process'].to_excel(writer, sheet_name='By_Process')
                agg_results['by_track'].to_excel(writer, sheet_name='By_Track')
                agg_results['by_thread'].to_excel(writer, sheet_name='By_Thread')

            print(f"  - XLSX: {xlsx_path}")

    def create_visualizations(self, save_formats: List[str] = ['png']) -> List[str]:
        """Создание всех визуализаций"""
        print("\nСоздание визуализаций:")
        image_paths = []

        # 1. Распределение кинетической энергии по типам частиц
        image_paths.append(self._plot_energy_distribution(save_formats))

        # 2. Распределение потерь энергии (boxplot)
        image_paths.append(self._plot_energy_loss_distribution(save_formats))

        # 3. Частота процессов
        image_paths.append(self._plot_process_frequency(save_formats))

        # 4. Пространственное распределение (heatmap)
        image_paths.append(self._plot_spatial_distribution(save_formats))

        # 5. Сравнение с итоговой сводкой
        image_paths.append(self._plot_verification(save_formats))

        return image_paths

    def _plot_energy_distribution(self, save_formats: List[str]) -> str:
        """График распределения кинетической энергии"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Распределение кинетической энергии по типам частиц', fontsize=14, fontweight='bold')

        particles = self.df['particle'].unique()

        for idx, particle in enumerate(particles[:4]):  # Максимум 4 частицы
            ax = axes[idx // 2, idx % 2]
            particle_data = self.df[self.df['particle'] == particle]['kine_e_mev']

            ax.hist(particle_data, bins=50, alpha=0.7, color=f'C{idx}', edgecolor='black')
            ax.set_xlabel('Кинетическая энергия (MeV)', fontsize=10)
            ax.set_ylabel('Частота', fontsize=10)
            ax.set_title(f'{particle} (n={len(particle_data)})', fontsize=11, fontweight='bold')

            # Статистика
            stats_text = f'Min: {particle_data.min():.4f} MeV\nMax: {particle_data.max():.4f} MeV\nMean: {particle_data.mean():.4f} MeV'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
            ax.grid(False)

        plt.tight_layout()
        
        filename = self.output_dir / f'energy_distribution.{save_formats[0]}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename}")
        return str(filename)

    def _plot_energy_loss_distribution(self, save_formats: List[str]) -> str:
        """График распределения потерь энергии"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Фильтруем только ненулевые значения dE
        data_plot = []
        labels = []
        for particle in self.df['particle'].unique():
            de_data = self.df[self.df['particle'] == particle]['de_step_mev']
            de_nonzero = de_data[de_data != 0]
            if len(de_nonzero) > 0:
                data_plot.append(de_nonzero)
                labels.append(f'{particle}\n(n={len(de_nonzero)})')

        if data_plot:
            bp = ax.boxplot(data_plot, labels=labels, patch_artist=True, showfliers=False)

            # Цвета для каждого box
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_ylabel('Потеря энергии на шаг (MeV)', fontsize=11, fontweight='bold')
            ax.set_title('Распределение потерь энергии (dE) по типам частиц', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        filename = self.output_dir / f'energy_loss_distribution.{save_formats[0]}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename}")
        return str(filename)

    def _plot_process_frequency(self, save_formats: List[str]) -> str:
        """График частоты процессов"""
        fig, ax = plt.subplots(figsize=(12, 8))

        process_counts = self.df['process'].value_counts()
        process_counts = process_counts.sort_values(ascending=True)

        colors = plt.cm.viridis(np.linspace(0, 1, len(process_counts)))
        bars = ax.barh(range(len(process_counts)), process_counts.values, color=colors)

        ax.set_yticks(range(len(process_counts)))
        ax.set_yticklabels(process_counts.index, fontsize=9)
        ax.set_xlabel('Количество вызовов', fontsize=11, fontweight='bold')
        ax.set_title('Частота процессов (из парсинга)', fontsize=13, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Подписи значений
        for i, (bar, val) in enumerate(zip(bars, process_counts.values)):
            ax.text(val, i, f' {val}', va='center', fontsize=8)

        plt.tight_layout()
        
        filename = self.output_dir / f'process_frequency.{save_formats[0]}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename}")
        return str(filename)

    def _plot_spatial_distribution(self, save_formats: List[str]) -> str:
        """Пространственное распределение координат"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # XY проекция
        ax1 = axes[0, 0]
        h1 = ax1.hist2d(self.df['x_mm'], self.df['y_mm'], bins=50, cmap='YlOrRd')
        ax1.set_xlabel('X (mm)', fontsize=10)
        ax1.set_ylabel('Y (mm)', fontsize=10)
        ax1.set_title('XY проекция', fontsize=11, fontweight='bold')
        plt.colorbar(h1[3], ax=ax1, label='Плотность')
        ax1.grid(False)

        # XZ проекция
        ax2 = axes[0, 1]
        h2 = ax2.hist2d(self.df['x_mm'], self.df['z_mm'], bins=50, cmap='YlGnBu')
        ax2.set_xlabel('X (mm)', fontsize=10)
        ax2.set_ylabel('Z (mm)', fontsize=10)
        ax2.set_title('XZ проекция', fontsize=11, fontweight='bold')
        plt.colorbar(h2[3], ax=ax2, label='Плотность')
        ax2.grid(False)

        # YZ проекция
        ax3 = axes[1, 0]
        h3 = ax3.hist2d(self.df['y_mm'], self.df['z_mm'], bins=50, cmap='PuRd')
        ax3.set_xlabel('Y (mm)', fontsize=10)
        ax3.set_ylabel('Z (mm)', fontsize=10)
        ax3.set_title('YZ проекция', fontsize=11, fontweight='bold')
        plt.colorbar(h3[3], ax=ax3, label='Плотность')
        ax3.grid(False)

        # 3D scatter (выборка)
        ax4 = axes[1, 1]
        sample_size = min(1000, len(self.df))
        df_sample = self.df.sample(n=sample_size)
        scatter = ax4.scatter(df_sample['x_mm'], df_sample['y_mm'],
                            c=df_sample['z_mm'], cmap='coolwarm', s=10, alpha=0.6)
        ax4.set_xlabel('X (mm)', fontsize=10)
        ax4.set_ylabel('Y (mm)', fontsize=10)
        ax4.set_title(f'XY с цветом по Z (sample n={sample_size})', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Z (mm)')
        ax4.grid(False)

        fig.suptitle('Пространственное распределение координат', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f'spatial_distribution.{save_formats[0]}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename}")
        return str(filename)

    def _plot_verification(self, save_formats: List[str]) -> str:
        """График сравнения с итоговой сводкой"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Сравнение энергии
        total_de_positive = self.df[self.df['de_step_mev'] > 0]['de_step_mev'].sum()
        energy_deposit_summary = self.summary.get('energy_deposit', 0)

        categories = ['Сумма dEStep\n(положит.)', 'Energy deposit\n(сводка)']
        values = [total_de_positive, energy_deposit_summary]
        colors_bars = ['#3498db', '#e74c3c']

        bars1 = ax1.bar(categories, values, color=colors_bars, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Энергия (MeV)', fontsize=11, fontweight='bold')
        ax1.set_title('Сравнение энергетических депозитов', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)

        # Подписи значений
        for bar, val in zip(bars1, values):
            ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.6f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Разница
        diff = abs(total_de_positive - energy_deposit_summary)
        rel_diff = (diff / energy_deposit_summary * 100) if energy_deposit_summary > 0 else 0
        ax1.text(0.5, 0.95, f'Абс. разница: {diff:.6f} MeV\nОтн. разница: {rel_diff:.2f}%',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=10)

        # 2. Сравнение процессов
        process_counts_parsed = self.df['process'].value_counts().to_dict()
        process_counts_summary = self.summary.get('process_frequencies', {})

        all_processes = set(process_counts_parsed.keys()) | set(process_counts_summary.keys())
        all_processes = sorted(all_processes)[:10]  # Топ-10

        parsed_vals = [process_counts_parsed.get(p, 0) for p in all_processes]
        summary_vals = [process_counts_summary.get(p, 0) for p in all_processes]

        x = np.arange(len(all_processes))
        width = 0.35

        bars2 = ax2.bar(x - width/2, parsed_vals, width, label='Парсинг', color='#3498db', alpha=0.7)
        bars3 = ax2.bar(x + width/2, summary_vals, width, label='Сводка', color='#e74c3c', alpha=0.7)

        ax2.set_xlabel('Процесс', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Количество вызовов', fontsize=11, fontweight='bold')
        ax2.set_title('Сравнение частоты процессов (топ-10)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_processes, rotation=45, ha='right', fontsize=8)
        ax2.legend(fontsize=10)
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        
        filename = self.output_dir / f'verification_comparison.{save_formats[0]}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - {filename}")
        return str(filename)

    def verify_results(self, parser: Geant4LogParser) -> Dict:
        """Проверка результатов парсинга"""
        verification = {}

        # 1. Энергетический баланс
        primary_tracks = [tid for tid, pid in parser.track_particles.items()
                         if tid in self.df['track_id'].values and
                         self.df[self.df['track_id'] == tid]['parent_id'].iloc[0] == 0]

        initial_energy_primary = sum(parser.track_initial_energy.get(tid, 0) for tid in primary_tracks)
        final_energy_primary = sum(parser.track_final_energy.get(tid, 0) for tid in primary_tracks)
        energy_lost = initial_energy_primary - final_energy_primary

        total_de_step_all = self.df['de_step_mev'].sum()
        total_de_step_positive = self.df[self.df['de_step_mev'] > 0]['de_step_mev'].sum()

        energy_balance = {
            'initial_energy_primary': initial_energy_primary,
            'final_energy_primary': final_energy_primary,
            'energy_lost': energy_lost,
            'total_de_step_all': total_de_step_all,
            'total_de_step_positive': total_de_step_positive
        }

        # 2. Сравнение с Energy deposit
        energy_deposit_summary = self.summary.get('energy_deposit', 0)

        abs_diff_de = abs(total_de_step_positive - energy_deposit_summary)
        rel_diff_de = (abs_diff_de / energy_deposit_summary * 100) if energy_deposit_summary > 0 else 0

        abs_diff_calc = abs(energy_lost - energy_deposit_summary)
        rel_diff_calc = (abs_diff_calc / energy_deposit_summary * 100) if energy_deposit_summary > 0 else 0

        verification['energy_balance'] = energy_balance
        verification['energy_deposit_summary'] = energy_deposit_summary
        verification['absolute_difference_de'] = abs_diff_de
        verification['relative_difference_de'] = rel_diff_de
        verification['absolute_difference_calc'] = abs_diff_calc
        verification['relative_difference_calc'] = rel_diff_calc

        # 3. Сравнение процессов
        process_counts_parsed = self.df['process'].value_counts().to_dict()
        process_counts_summary = self.summary.get('process_frequencies', {})

        all_processes = set(process_counts_parsed.keys()) | set(process_counts_summary.keys())
        process_comparison = {}

        for process in all_processes:
            parsed_count = process_counts_parsed.get(process, 0)
            summary_count = process_counts_summary.get(process, 0)
            difference = parsed_count - summary_count

            process_comparison[process] = {
                'parsed': parsed_count,
                'summary': summary_count,
                'difference': difference,
                'abs_difference': abs(difference)
            }

        verification['process_comparison'] = process_comparison

        return verification

    def generate_report(self, parser: Geant4LogParser) -> str:
        """Генерация текстового отчета"""
        report = []
        report.append("=" * 100)
        report.append("ОТЧЕТ ПО АНАЛИЗУ ЛОГОВ GEANT4")
        report.append("=" * 100)
        report.append("")

        # Информация о входном файле
        if self.input_filename:
            report.append(f"Входной файл: {self.input_filename}")
        report.append(f"Выходная директория: {self.output_dir}")
        report.append("")

        # 1. Основная статистика
        report.append("1. ОСНОВНАЯ СТАТИСТИКА")
        report.append("-" * 100)
        stats = parser.get_statistics()
        report.append(f"Всего шагов найдено: {stats['total_steps']}")
        report.append(f"Уникальных потоков: {stats['unique_threads']}")
        report.append(f"Уникальных частиц: {stats['unique_particles']}")
        report.append(f"Уникальных процессов: {stats['unique_processes']}")
        report.append(f"Уникальных треков: {stats['unique_tracks']}")
        report.append("")

        # 2. Статистика по частицам
        report.append("2. СТАТИСТИКА ПО ТИПАМ ЧАСТИЦ")
        report.append("-" * 100)
        by_particle = self.df.groupby('particle').agg({
            'de_step_mev': ['sum', 'count'],
            'kine_e_mev': ['mean', 'min', 'max']
        }).round(6)

        for particle in by_particle.index:
            report.append(f"\nЧастица: {particle}")
            report.append(f"  Количество шагов: {by_particle.loc[particle, ('de_step_mev', 'count')]:.0f}")
            report.append(f"  Суммарная dEStep: {by_particle.loc[particle, ('de_step_mev', 'sum')]:.6f} MeV")
            report.append(f"  Средняя энергия: {by_particle.loc[particle, ('kine_e_mev', 'mean')]:.6f} MeV")
            report.append(f"  Мин энергия: {by_particle.loc[particle, ('kine_e_mev', 'min')]:.6f} MeV")
            report.append(f"  Макс энергия: {by_particle.loc[particle, ('kine_e_mev', 'max')]:.6f} MeV")
        report.append("")

        # 3. Верификация
        report.append("3. ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ")
        report.append("-" * 100)
        verification = self.verify_results(parser)

        # Энергетический баланс
        energy_balance = verification['energy_balance']
        report.append("Энергетический баланс:")
        report.append(f"  Начальная энергия первичных частиц: {energy_balance['initial_energy_primary']:.6f} MeV")
        report.append(f"  Конечная энергия первичных частиц: {energy_balance['final_energy_primary']:.6f} MeV")
        report.append(f"  Потерянная энергия (расчет): {energy_balance['energy_lost']:.6f} MeV")
        report.append(f"  Сумма положительных dEStep: {energy_balance['total_de_step_positive']:.6f} MeV")
        report.append(f"  Сумма всех dEStep: {energy_balance['total_de_step_all']:.6f} MeV")
        report.append("")

        # Сравнение с Energy deposit
        report.append("Сверка с итоговой сводкой:")
        report.append(f"  Energy deposit (сводка): {verification['energy_deposit_summary']:.6f} MeV")
        report.append(f"  Метод 1 (сумма положит. dEStep):")
        report.append(f"    - Абсолютная разница: {verification['absolute_difference_de']:.6f} MeV")
        report.append(f"    - Относительная разница: {verification['relative_difference_de']:.4f}%")
        report.append(f"  Метод 2 (баланс энергии):")
        report.append(f"    - Абсолютная разница: {verification['absolute_difference_calc']:.6f} MeV")
        report.append(f"    - Относительная разница: {verification['relative_difference_calc']:.4f}%")
        report.append("")

        # Определение типа лога
        log_type = "complete"
        balance_rel = verification['relative_difference_calc']

        if balance_rel < 5:
            log_type = "incomplete"
            report.append("4. ТИП ЛОГА И ИНТЕРПРЕТАЦИЯ")
            report.append("-" * 100)
            report.append("Тип лога: НЕПОЛНЫЙ (не все шаги записаны)")
            report.append("  ✅ Энергетический баланс первичных частиц сошёлся (<5%)")
            report.append("  Это нормально для неполных логов - главное, что баланс энергии первичных частиц сходится.")
        else:
            report.append("4. ТИП ЛОГА И ИНТЕРПРЕТАЦИЯ")
            report.append("-" * 100)
            report.append("Тип лога: ПОЛНЫЙ или ЧАСТИЧНЫЙ")
            report.append(f"  Относительное расхождение баланса: {balance_rel:.2f}%")

            if balance_rel > 10:
                report.append("  ⚠️  Есть расхождение в энергетическом балансе")
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
                diff_str = f"{data['difference']:+d}"
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
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчет сохранен: {report_path}")
        return str(report_path)


# ======================= GUI APPLICATION =======================

class Geant4ParserGUI:
    """Графический интерфейс для парсера Geant4"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Geant4 Log Parser - GUI")
        self.root.geometry("1200x800")
        
        # Переменные
        self.log_file_path = None
        self.output_dir = "output"
        self.parser_obj = None
        self.analyzer = None
        self.df = None
        self.report_path = None
        self.csv_path = None
        self.xlsx_path = None
        self.image_paths = []
        
        # Создание интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        """Создание виджетов интерфейса"""
        
        # Верхняя панель - загрузка файла
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Log файл:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(top_frame, text="Не выбран", foreground="red")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(top_frame, text="Загрузить Log", command=self.load_log_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Анализировать", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        
        # Прогресс-бар
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Вкладка 1: Консоль вывода
        console_frame = ttk.Frame(self.notebook)
        self.notebook.add(console_frame, text="📊 Консоль анализа")
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, 
                                                       font=('Courier', 9), bg='#1e1e1e', fg='#00ff00')
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка 2: Результаты отчета
        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="📄 Отчет")
        
        self.report_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD, 
                                                      font=('Courier', 9))
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка 3: Визуализации
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📈 Визуализации")
        
        # Canvas для прокрутки изображений
        canvas_frame = ttk.Frame(viz_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.viz_canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.viz_canvas.yview)
        self.viz_scrollable_frame = ttk.Frame(self.viz_canvas)
        
        self.viz_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))
        )
        
        self.viz_canvas.create_window((0, 0), window=self.viz_scrollable_frame, anchor="nw")
        self.viz_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.viz_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Нижняя панель - кнопки экспорта
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)
        
        ttk.Label(bottom_frame, text="Экспорт данных:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.btn_open_csv = ttk.Button(bottom_frame, text="📊 Открыть CSV", 
                                        command=self.open_csv, state=tk.DISABLED)
        self.btn_open_csv.pack(side=tk.LEFT, padx=5)
        
        self.btn_open_xlsx = ttk.Button(bottom_frame, text="📗 Открыть Excel", 
                                         command=self.open_xlsx, state=tk.DISABLED)
        self.btn_open_xlsx.pack(side=tk.LEFT, padx=5)
        
        self.btn_open_report = ttk.Button(bottom_frame, text="📄 Открыть отчет", 
                                           command=self.open_report, state=tk.DISABLED)
        self.btn_open_report.pack(side=tk.LEFT, padx=5)
        
        self.btn_open_folder = ttk.Button(bottom_frame, text="📁 Открыть папку", 
                                           command=self.open_output_folder, state=tk.DISABLED)
        self.btn_open_folder.pack(side=tk.LEFT, padx=5)
        
    def log_to_console(self, message):
        """Вывод сообщения в консоль"""
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.console_text.update()
        
    def load_log_file(self):
        """Загрузка log файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл лога Geant4",
            filetypes=[("Log files", "*.log *.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.log_file_path = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground="green")
            self.log_to_console(f"✅ Файл загружен: {file_path}")
            
    def start_analysis(self):
        """Запуск анализа в отдельном потоке"""
        if not self.log_file_path:
            messagebox.showerror("Ошибка", "Пожалуйста, сначала загрузите log файл!")
            return
            
        # Очистка предыдущих результатов
        self.console_text.delete(1.0, tk.END)
        self.report_text.delete(1.0, tk.END)
        for widget in self.viz_scrollable_frame.winfo_children():
            widget.destroy()
            
        # Отключение кнопок
        self.btn_open_csv.config(state=tk.DISABLED)
        self.btn_open_xlsx.config(state=tk.DISABLED)
        self.btn_open_report.config(state=tk.DISABLED)
        self.btn_open_folder.config(state=tk.DISABLED)
        
        # Запуск анализа в отдельном потоке
        self.progress.start()
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
        
    def run_analysis(self):
        """Выполнение анализа"""
        try:
            # Парсинг лога
            self.log_to_console("=" * 80)
            self.log_to_console("GEANT4 LOG PARSER - АНАЛИЗ")
            self.log_to_console("=" * 80)
            self.log_to_console(f"\n🔍 Парсинг файла: {self.log_file_path}")
            
            self.parser_obj = Geant4LogParser(self.log_file_path)
            self.parser_obj.parse_log(debug=False)
            
            if not self.parser_obj.steps:
                self.log_to_console("\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Не найдено данных о шагах в логе!")
                self.progress.stop()
                return
                
            self.log_to_console(f"✅ Парсинг завершен. Найдено {len(self.parser_obj.steps)} шагов.")
            
            # Конвертация в DataFrame
            self.df = self.parser_obj.to_dataframe()
            self.log_to_console(f"✅ Создан DataFrame с {len(self.df)} записями")
            
            # Создание анализатора
            self.analyzer = Geant4Analyzer(self.df, self.parser_obj.summary, 
                                          self.output_dir, self.log_file_path)
            
            # Агрегация данных
            self.log_to_console("\n📊 Агрегация данных...")
            agg_results = self.analyzer.aggregate_data()
            self.log_to_console("✅ Агрегация завершена")
            
            # Экспорт данных
            self.log_to_console("\n💾 Экспорт данных...")
            self.analyzer.export_data(formats=['csv', 'xlsx'])
            self.csv_path = str(Path(self.output_dir) / 'steps_data.csv')
            self.xlsx_path = str(Path(self.output_dir) / 'analysis_results.xlsx')
            self.log_to_console(f"✅ CSV сохранен: {self.csv_path}")
            self.log_to_console(f"✅ Excel сохранен: {self.xlsx_path}")
            
            # Визуализация
            self.log_to_console("\n📈 Создание визуализаций...")
            self.image_paths = self.analyzer.create_visualizations(save_formats=['png'])
            self.log_to_console("✅ Визуализации созданы")
            
            # Генерация отчета
            self.log_to_console("\n📄 Генерация отчета...")
            self.report_path = self.analyzer.save_report(self.parser_obj)
            self.log_to_console(f"✅ Отчет сохранен: {self.report_path}")
            
            # Вывод итоговой статистики
            self.log_to_console("\n" + "=" * 80)
            self.log_to_console("ИТОГОВАЯ СТАТИСТИКА")
            self.log_to_console("=" * 80)
            verification = self.analyzer.verify_results(self.parser_obj)
            
            energy_balance = verification['energy_balance']
            self.log_to_console(f"\nЭнергетический баланс:")
            self.log_to_console(f"  Начальная энергия: {energy_balance['initial_energy_primary']:.6f} MeV")
            self.log_to_console(f"  Потерянная энергия: {energy_balance['energy_lost']:.6f} MeV")
            self.log_to_console(f"  Сумма положит. dEStep: {energy_balance['total_de_step_positive']:.6f} MeV")
            
            self.log_to_console(f"\nСверка с итоговой сводкой:")
            self.log_to_console(f"  Energy deposit (сводка): {verification['energy_deposit_summary']:.6f} MeV")
            self.log_to_console(f"  Относительная разница: {verification['relative_difference_de']:.4f}%")
            
            self.log_to_console("\n" + "=" * 80)
            self.log_to_console(f"✅ Анализ завершен! Результаты в папке: {self.output_dir}")
            
            # Загрузка отчета
            self.load_report()
            
            # Загрузка визуализаций
            self.load_visualizations()
            
            # Включение кнопок
            self.root.after(0, self.enable_buttons)
            
        except Exception as e:
            self.log_to_console(f"\n❌ ОШИБКА: {str(e)}")
            import traceback
            self.log_to_console(traceback.format_exc())
        finally:
            self.progress.stop()
            
    def load_report(self):
        """Загрузка отчета в текстовое поле"""
        if self.report_path and os.path.exists(self.report_path):
            with open(self.report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report_content)
            
    def load_visualizations(self):
        """Загрузка изображений визуализаций"""
        for img_path in self.image_paths:
            if os.path.exists(img_path):
                try:
                    # Загрузка изображения
                    img = Image.open(img_path)
                    
                    # Изменение размера для отображения
                    display_width = 1100
                    aspect_ratio = img.height / img.width
                    display_height = int(display_width * aspect_ratio)
                    img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(img)
                    
                    # Создание фрейма для изображения
                    img_frame = ttk.Frame(self.viz_scrollable_frame)
                    img_frame.pack(pady=10)
                    
                    # Заголовок
                    title = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
                    ttk.Label(img_frame, text=title, font=('Arial', 11, 'bold')).pack()
                    
                    # Изображение
                    label = ttk.Label(img_frame, image=photo)
                    label.image = photo  # Сохраняем ссылку
                    label.pack()
                    
                except Exception as e:
                    self.log_to_console(f"⚠️ Ошибка загрузки изображения {img_path}: {e}")
                    
    def enable_buttons(self):
        """Включение кнопок экспорта"""
        self.btn_open_csv.config(state=tk.NORMAL)
        self.btn_open_xlsx.config(state=tk.NORMAL)
        self.btn_open_report.config(state=tk.NORMAL)
        self.btn_open_folder.config(state=tk.NORMAL)
        
    def open_csv(self):
        """Открытие CSV файла"""
        if self.csv_path and os.path.exists(self.csv_path):
            self.open_file(self.csv_path)
        else:
            messagebox.showerror("Ошибка", "CSV файл не найден!")
            
    def open_xlsx(self):
        """Открытие Excel файла"""
        if self.xlsx_path and os.path.exists(self.xlsx_path):
            self.open_file(self.xlsx_path)
        else:
            messagebox.showerror("Ошибка", "Excel файл не найден!")
            
    def open_report(self):
        """Открытие файла отчета"""
        if self.report_path and os.path.exists(self.report_path):
            self.open_file(self.report_path)
        else:
            messagebox.showerror("Ошибка", "Файл отчета не найден!")
            
    def open_output_folder(self):
        """Открытие папки с результатами"""
        if os.path.exists(self.output_dir):
            if sys.platform == 'win32':
                os.startfile(self.output_dir)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', self.output_dir])
            else:  # linux
                subprocess.Popen(['xdg-open', self.output_dir])
        else:
            messagebox.showerror("Ошибка", "Папка с результатами не найдена!")
            
    def open_file(self, filepath):
        """Открытие файла в системном приложении"""
        try:
            if sys.platform == 'win32':
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', filepath])
            else:  # linux
                subprocess.Popen(['xdg-open', filepath])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")


def main():
    """Запуск GUI приложения"""
    root = tk.Tk()
    app = Geant4ParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
