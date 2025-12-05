from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)


# ---------------------------------------------------------------------------
# Domain Model
# ---------------------------------------------------------------------------

class WorkoutType(Enum):
    SWIMMING = "swimming"
    CYCLING = "cycling"
    RUNNING = "running"

    @classmethod
    def from_menu_index(cls, index: int) -> "WorkoutType":
        mapping = {
            1: cls.SWIMMING,
            2: cls.CYCLING,
            3: cls.RUNNING,
        }
        return mapping[index]

    @classmethod
    def all(cls) -> List["WorkoutType"]:
        return [cls.SWIMMING, cls.CYCLING, cls.RUNNING]


@dataclass(frozen=True)
class TrainingRecord:
    """
    Represents a single workout session.
    """
    workout_type: WorkoutType
    distance_km: float
    time_min: float
    avg_pulse_bpm: float
    avg_speed_kmh: float

    @classmethod
    def from_raw(
        cls,
        workout_type: WorkoutType,
        distance_km: float,
        time_min: float,
        avg_pulse_bpm: float,
    ) -> "TrainingRecord":
        if time_min <= 0:
            raise ValueError("time_min must be > 0 to compute average speed.")
        avg_speed_kmh = distance_km / (time_min / 60.0)
        return cls(
            workout_type=workout_type,
            distance_km=distance_km,
            time_min=time_min,
            avg_pulse_bpm=avg_pulse_bpm,
            avg_speed_kmh=avg_speed_kmh,
        )


@dataclass(frozen=True)
class AggregateStats:
    """
    Aggregated statistics over a collection of TrainingRecords.
    """
    count: int
    total_distance_km: float
    avg_pulse_bpm: float
    avg_speed_kmh: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CsvConfig:
    """
    Configuration for CSV persistence.
    """
    file_path: Path
    newline: str = ""
    encoding: str = "utf-8"


@dataclass(frozen=True)
class AppConfig:
    """
    High-level application configuration.
    """
    csv: CsvConfig

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(csv=CsvConfig(file_path=Path("trainings.csv")))


# ---------------------------------------------------------------------------
# IO Abstractions (for testability / overengineering)
# ---------------------------------------------------------------------------

@runtime_checkable
class InputProvider(Protocol):
    def ask(self, prompt: str) -> str:
        ...


@runtime_checkable
class OutputProvider(Protocol):
    def write(self, text: str = "") -> None:
        ...


class ConsoleInputProvider:
    """
    Concrete implementation using built-in input().
    """

    def ask(self, prompt: str) -> str:
        return input(prompt)


class ConsoleOutputProvider:
    """
    Concrete implementation using built-in print().
    """

    def write(self, text: str = "") -> None:
        print(text)


# ---------------------------------------------------------------------------
# Repository Layer
# ---------------------------------------------------------------------------

class TrainingRepositoryError(Exception):
    """Base exception for repository-related errors."""


class TrainingRepository(Protocol):
    def append(self, record: TrainingRecord) -> None:
        ...

    def iter_all(self) -> Iterable[TrainingRecord]:
        ...


class CsvTrainingRepository:
    """
    CSV-based implementation of TrainingRepository.
    """

    def __init__(self, config: CsvConfig) -> None:
        self._config = config

    def append(self, record: TrainingRecord) -> None:
        try:
            with self._config.file_path.open(
                mode="a", newline=self._config.newline, encoding=self._config.encoding
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        record.workout_type.value,
                        f"{record.distance_km}",
                        f"{record.time_min}",
                        f"{record.avg_pulse_bpm}",
                        f"{record.avg_speed_kmh}",
                    ]
                )
        except OSError as exc:
            raise TrainingRepositoryError(
                f"Error while saving training data: {exc}"
            ) from exc

    def iter_all(self) -> Iterable[TrainingRecord]:
        path = self._config.file_path
        if not path.exists():
            # No file yet -> no trainings.
            return iter([])

        try:
            file = path.open(
                mode="r", newline=self._config.newline, encoding=self._config.encoding
            )
        except OSError as exc:
            raise TrainingRepositoryError(
                f"Error while loading training data: {exc}"
            ) from exc

        def generator() -> Iterable[TrainingRecord]:
            with file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 5:
                        # Skip malformed rows
                        continue
                    try:
                        workout_type_raw = row[0]
                        distance_raw = float(row[1])
                        time_raw = float(row[2])
                        pulse_raw = float(row[3])
                        speed_raw = float(row[4])
                        workout_type = WorkoutType(workout_type_raw)
                        yield TrainingRecord(
                            workout_type=workout_type,
                            distance_km=distance_raw,
                            time_min=time_raw,
                            avg_pulse_bpm=pulse_raw,
                            avg_speed_kmh=speed_raw,
                        )
                    except (ValueError, KeyError):
                        # Skip rows with invalid data
                        continue

        return generator()


# ---------------------------------------------------------------------------
# Domain Services
# ---------------------------------------------------------------------------

class StatsService:
    """
    Encapsulates statistics computations over trainings.
    """

    def compute_last_n(
        self,
        records: Iterable[TrainingRecord],
        workout_type: Optional[WorkoutType],
        n: int,
    ) -> Optional[AggregateStats]:
        """
        Compute aggregate stats for the last n records of the given workout_type.
        If workout_type is None, use all types.
        """
        if n <= 0:
            return None

        # Collect the last N matching records without loading everything into memory.
        from collections import deque

        relevant = deque(maxlen=n)
        for record in records:
            if workout_type is None or record.workout_type == workout_type:
                relevant.append(record)

        if not relevant:
            return None

        count = len(relevant)
        total_distance = sum(r.distance_km for r in relevant)
        avg_pulse = sum(r.avg_pulse_bpm for r in relevant) / count
        avg_speed = sum(r.avg_speed_kmh for r in relevant) / count

        return AggregateStats(
            count=count,
            total_distance_km=total_distance,
            avg_pulse_bpm=avg_pulse,
            avg_speed_kmh=avg_speed,
        )


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

class MainMenuOption(Enum):
    SHOW_STATS = auto()
    ADD_TRAINING = auto()
    EXIT = auto()


class UiStrings:
    """
    Central place to keep user-facing strings, to avoid scattering literals.
    """

    APP_WELCOME = "\nWelcome to TriaTrack, your tracker for all your triathletic workouts"
    SEPARATOR_MAIN = "-" * 70
    SEPARATOR_SUB = "-" * 40

    MAIN_MENU_HEADER = "Main Menu"
    MAIN_MENU_OPTIONS = [
        "Show stats",
        "Add new Training",
        "End Program",
    ]

    NEW_TRAINING_HEADER = "\nAdd new training"
    TRAINING_CHOICE_PROMPT = "Choose training: "

    STATS_MENU_HEADER = "\nWhich Stats would you like to show?"
    STATS_CHOICE_PROMPT = "Which Stats would you like to show?: "

    ANALYSER_HEADER = "\nWorkout Analyser"
    ANALYSER_PROMPT = "How many of your last workouts do you want to analyse: "

    MSG_INVALID_CHOICE = "Invalid choice"
    MSG_ENTER_NUMBER = "Please enter a number."
    MSG_ENTER_POSITIVE = "Please enter a correct number."
    MSG_PROGRAM_ENDED = "Program ended"
    MSG_PRESS_ENTER_CONTINUE = "Press Enter to continue..."
    MSG_PRESS_ENTER_CONTINUE_ALT = "Press ENTER to continue..."


# ---------------------------------------------------------------------------
# Input Validation / Parsing
# ---------------------------------------------------------------------------

class InputValidator:
    """
    Responsible for robust, reusable input parsing and validation.
    """

    def __init__(self, input_provider: InputProvider, output_provider: OutputProvider):
        self._in = input_provider
        self._out = output_provider

    def ask_int(
        self,
        prompt: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """
        Ask user for integer input with optional range constraints.
        """
        while True:
            raw = self._in.ask(prompt)
            try:
                value = int(raw)
            except ValueError:
                self._out.write(UiStrings.MSG_ENTER_NUMBER)
                continue

            if min_value is not None and value < min_value:
                self._out.write(UiStrings.MSG_INVALID_CHOICE)
                continue
            if max_value is not None and value > max_value:
                self._out.write(UiStrings.MSG_INVALID_CHOICE)
                continue
            return value

    def ask_positive_float(self, prompt: str) -> float:
        """
        Ask user for a strictly positive float.
        """
        while True:
            raw = self._in.ask(prompt)
            try:
                value = float(raw)
            except ValueError:
                self._out.write(UiStrings.MSG_ENTER_NUMBER)
                continue

            if value <= 0:
                self._out.write(UiStrings.MSG_ENTER_POSITIVE)
                continue
            return value


# ---------------------------------------------------------------------------
# CLI Application
# ---------------------------------------------------------------------------

class TriaTrackApp:
    """
    High-level orchestration of the application.
    """

    def __init__(
        self,
        repo: TrainingRepository,
        stats_service: StatsService,
        input_provider: InputProvider,
        output_provider: OutputProvider,
    ) -> None:
        self._repo = repo
        self._stats_service = stats_service
        self._in = input_provider
        self._out = output_provider
        self._validator = InputValidator(input_provider, output_provider)

    # ---------- Main Loop ----------

    def run(self) -> None:
        """
        Application entrypoint: shows and handles the main menu.
        """
        while True:
            self._print_main_menu()
            choice = self._ask_main_menu_choice()

            if choice is MainMenuOption.SHOW_STATS:
                self._handle_show_stats()
            elif choice is MainMenuOption.ADD_TRAINING:
                self._handle_add_training()
            elif choice is MainMenuOption.EXIT:
                break

        self._out.write(UiStrings.MSG_PROGRAM_ENDED)

    # ---------- Menu Rendering ----------

    def _print_main_menu(self) -> None:
        self._out.write(UiStrings.APP_WELCOME)
        self._out.write(UiStrings.SEPARATOR_MAIN)
        for idx, label in enumerate(UiStrings.MAIN_MENU_OPTIONS, start=1):
            self._out.write(f"{idx}) {label}")
        self._out.write(UiStrings.SEPARATOR_MAIN)

    def _ask_main_menu_choice(self) -> MainMenuOption:
        max_index = len(UiStrings.MAIN_MENU_OPTIONS)
        while True:
            try:
                choice_raw = self._validator.ask_int(
                    "Choose function: ", min_value=1, max_value=max_index
                )
                if choice_raw == 1:
                    return MainMenuOption.SHOW_STATS
                if choice_raw == 2:
                    return MainMenuOption.ADD_TRAINING
                if choice_raw == 3:
                    return MainMenuOption.EXIT
            except Exception:
                # Shouldn't happen due to validation, but we keep this just in case.
                self._out.write(UiStrings.MSG_INVALID_CHOICE)

    # ---------- Add Training Flow ----------

    def _handle_add_training(self) -> None:
        self._print_new_training_menu()

        # Present workout options based on WorkoutType enum:
        workout_types = WorkoutType.all()
        for idx, wt in enumerate(workout_types, start=1):
            self._out.write(f"{idx}) {wt.value}")

        choice_index = self._validator.ask_int(
            UiStrings.TRAINING_CHOICE_PROMPT,
            min_value=1,
            max_value=len(workout_types),
        )
        workout_type = WorkoutType.from_menu_index(choice_index)

        self._out.write(UiStrings.SEPARATOR_SUB)
        self._out.write(f"\nCreating new {workout_type.value} workout")
        self._out.write(UiStrings.SEPARATOR_SUB)

        record = self._input_training_data(workout_type)

        try:
            self._repo.append(record)
            self._out.write(
                f"\nSuccessfully added {workout_type.value}-training "
                f"of {record.distance_km:.2f} km."
            )
            self._out.write(UiStrings.SEPARATOR_MAIN)
        except TrainingRepositoryError as exc:
            self._out.write(str(exc))
            self._out.write(UiStrings.SEPARATOR_MAIN)

        self._wait_for_enter(UiStrings.MSG_PRESS_ENTER_CONTINUE_ALT)

    def _print_new_training_menu(self) -> None:
        self._out.write(UiStrings.SEPARATOR_SUB)
        self._out.write(UiStrings.NEW_TRAINING_HEADER)
        self._out.write(UiStrings.SEPARATOR_SUB)

    def _input_training_data(self, workout_type: WorkoutType) -> TrainingRecord:
        distance = self._validator.ask_positive_float(
            "Distance accomplished (in km): "
        )
        time_min = self._validator.ask_positive_float("Time needed in minutes: ")
        pulse = self._validator.ask_positive_float("Average pulse: ")

        return TrainingRecord.from_raw(
            workout_type=workout_type,
            distance_km=distance,
            time_min=time_min,
            avg_pulse_bpm=pulse,
        )

    # ---------- Stats Flow ----------

    def _handle_show_stats(self) -> None:
        workout_type = self._ask_stats_workout_type()
        self._out.write(UiStrings.SEPARATOR_MAIN)
        self._print_stats_menu_time_span()

        time_span = self._validator.ask_int(
            UiStrings.ANALYSER_PROMPT,
            min_value=1,
        )

        try:
            records = list(self._repo.iter_all())
        except TrainingRepositoryError as exc:
            self._out.write(str(exc))
            self._out.write(UiStrings.SEPARATOR_MAIN)
            self._wait_for_enter(UiStrings.MSG_PRESS_ENTER_CONTINUE)
            return

        stats = self._stats_service.compute_last_n(
            records=records, workout_type=workout_type, n=time_span
        )
        self._display_training_stats(workout_type, stats)
        self._wait_for_enter(UiStrings.MSG_PRESS_ENTER_CONTINUE)

    def _ask_stats_workout_type(self) -> Optional[WorkoutType]:
        """
        Ask which workout type's stats to show. Returns:
        - WorkoutType for specific discipline
        - None for 'all'
        """
        workout_types = WorkoutType.all()

        self._out.write(UiStrings.STATS_MENU_HEADER)
        self._out.write(UiStrings.SEPARATOR_MAIN)

        for idx, wt in enumerate(workout_types, start=1):
            self._out.write(f"{idx}) Show {wt.value} workouts")

        all_index = len(workout_types) + 1
        self._out.write(f"{all_index}) Show all workouts")
        self._out.write(UiStrings.SEPARATOR_MAIN)

        choice_index = self._validator.ask_int(
            UiStrings.STATS_CHOICE_PROMPT,
            min_value=1,
            max_value=all_index,
        )

        if choice_index == all_index:
            return None
        return WorkoutType.from_menu_index(choice_index)

    def _print_stats_menu_time_span(self) -> None:
        self._out.write(UiStrings.ANALYSER_HEADER)
        self._out.write(UiStrings.SEPARATOR_MAIN)

    def _display_training_stats(
        self,
        workout_type: Optional[WorkoutType],
        stats: Optional[AggregateStats],
    ) -> None:
        if stats is None:
            label = "all trainings" if workout_type is None else workout_type.value
            self._out.write(f"\nNo {label} trainings found.")
            self._out.write(UiStrings.SEPARATOR_MAIN)
            return

        if workout_type is None:
            label = "all"
        else:
            label = workout_type.value

        self._out.write(
            f"\nDisplaying your stats for the last "
            f"{stats.count} {label}-training(s)"
        )
        self._out.write(UiStrings.SEPARATOR_MAIN)
        self._out.write(
            f"Total distance: {stats.total_distance_km:.2f} KM"
        )
        self._out.write(
            f"Average pulse: {stats.avg_pulse_bpm:.2f} Beats per Minute"
        )
        self._out.write(
            f"Average speed: {stats.avg_speed_kmh:.2f} KM/H"
        )
        self._out.write(UiStrings.SEPARATOR_MAIN)

    # ---------- Misc Helpers ----------

    def _wait_for_enter(self, prompt: str) -> None:
        # Using input provider for consistency (helps testing)
        self._in.ask(prompt)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build_app(config: Optional[AppConfig] = None) -> TriaTrackApp:
    """
    Build the fully-wired application instance.
    """
    if config is None:
        config = AppConfig.default()

    repo = CsvTrainingRepository(config.csv)
    stats_service = StatsService()
    input_provider = ConsoleInputProvider()
    output_provider = ConsoleOutputProvider()

    return TriaTrackApp(
        repo=repo,
        stats_service=stats_service,
        input_provider=input_provider,
        output_provider=output_provider,
    )


def main() -> None:
    app = build_app()
    app.run()


if __name__ == "__main__":
    main()
