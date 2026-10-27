"""Streamlit interface for 100-point acoustic trajectory extraction."""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
import tgt


APP_TITLE = "100-point Acoustic Trajectory Extractor"
EXCEL_ROW_LIMIT = 1_048_576


@dataclass(frozen=True)
class ExtractionSettings:
    tier_name: str = "segment"
    n_points: int = 100
    time_step_s: float = 0.005
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 400.0
    max_formant_hz: float = 5500.0
    n_formants: int = 5
    skip_empty: bool = True

    def validate(self) -> None:
        if not self.tier_name.strip():
            raise ValueError("Tier name cannot be empty.")
        if self.n_points != 100:
            raise ValueError("This FPCA extractor requires exactly 100 points.")
        if self.time_step_s <= 0:
            raise ValueError("Time step must be positive.")
        if self.pitch_floor_hz <= 0:
            raise ValueError("Pitch floor must be positive.")
        if self.pitch_floor_hz >= self.pitch_ceiling_hz:
            raise ValueError("Pitch floor must be lower than pitch ceiling.")
        if self.max_formant_hz <= 0:
            raise ValueError("Maximum formant must be positive.")
        if not 3 <= self.n_formants <= 6:
            raise ValueError("Number of formants must be between 3 and 6.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_textgrid(path: Path) -> tgt.core.TextGrid:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return tgt.read_textgrid(str(path), encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not read TextGrid. " + " | ".join(errors))


def get_interval_tier(textgrid: tgt.core.TextGrid, tier_name: str) -> Any:
    wanted = tier_name.strip().casefold()
    for tier in textgrid.tiers:
        if tier.name.strip().casefold() == wanted:
            if not hasattr(tier, "intervals"):
                raise ValueError(f"Tier '{tier.name}' is not an interval tier.")
            return tier
    available = ", ".join(tier.name for tier in textgrid.tiers)
    raise ValueError(
        f"Tier '{tier_name}' not found. Available tiers: {available or '(none)'}"
    )


def optional_float(value: Any, *, positive: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if positive and number <= 0:
        return None
    return number


def extract_pair(
    audio_path: Path,
    textgrid_path: Path,
    settings: ExtractionSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings.validate()

    sound = parselmouth.Sound(str(audio_path))
    textgrid = read_textgrid(textgrid_path)
    tier = get_interval_tier(textgrid, settings.tier_name)

    pitch = sound.to_pitch_ac(
        time_step=settings.time_step_s,
        pitch_floor=settings.pitch_floor_hz,
        pitch_ceiling=settings.pitch_ceiling_hz,
    )
    intensity = sound.to_intensity(
        minimum_pitch=settings.pitch_floor_hz,
        time_step=settings.time_step_s,
    )
    formant = sound.to_formant_burg(
        time_step=settings.time_step_s,
        max_number_of_formants=settings.n_formants,
        maximum_formant=settings.max_formant_hz,
    )

    sound_duration_s = float(sound.duration)
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for interval_index, interval in enumerate(tier.intervals, start=1):
        label = interval.text.strip()
        if settings.skip_empty and not label:
            continue

        start_s = float(interval.start_time)
        end_s = float(interval.end_time)
        if start_s < -1e-6 or end_s > sound_duration_s + 1e-6:
            raise ValueError(
                f"Interval {interval_index} ({start_s:.6f}–{end_s:.6f}s) lies "
                f"outside the audio duration (0–{sound_duration_s:.6f}s)."
            )
        start_s = max(0.0, start_s)
        end_s = min(sound_duration_s, end_s)
        duration_s = end_s - start_s
        if duration_s <= 0:
            continue

        sample_times_s = np.linspace(start_s, end_s, settings.n_points)
        time_percent = np.linspace(0.0, 100.0, settings.n_points)
        valid_counts = {
            "f0_hz": 0,
            "intensity_db": 0,
            **{f"F{number}_hz": 0 for number in range(1, settings.n_formants + 1)},
        }

        for point_index, (time_s, percent) in enumerate(
            zip(sample_times_s, time_percent, strict=True), start=1
        ):
            f0_hz = optional_float(
                pitch.get_value_at_time(float(time_s)), positive=True
            )
            intensity_db = optional_float(intensity.get_value(float(time_s)))

            row: dict[str, Any] = {
                "file": audio_path.name,
                "textgrid": textgrid_path.name,
                "tier": tier.name,
                "interval_index": interval_index,
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "duration_ms": duration_s * 1000.0,
                "point_index": point_index,
                "time_percent": percent,
                "time_s": float(time_s),
                "time_ms": float(time_s) * 1000.0,
                "segment_ms": (float(time_s) - start_s) * 1000.0,
                "voiced": int(f0_hz is not None),
                "f0_hz": f0_hz,
                "intensity_db": intensity_db,
            }
            if f0_hz is not None:
                valid_counts["f0_hz"] += 1
            if intensity_db is not None:
                valid_counts["intensity_db"] += 1

            for formant_number in range(1, settings.n_formants + 1):
                column = f"F{formant_number}_hz"
                value = optional_float(
                    formant.get_value_at_time(formant_number, float(time_s)),
                    positive=True,
                )
                row[column] = value
                if value is not None:
                    valid_counts[column] += 1

            rows.append(row)

        qc_row: dict[str, Any] = {
            "file": audio_path.name,
            "textgrid": textgrid_path.name,
            "tier": tier.name,
            "interval_index": interval_index,
            "label": label,
            "duration_ms": duration_s * 1000.0,
            "n_points": settings.n_points,
            "voiced_points": valid_counts["f0_hz"],
            "voiced_proportion": valid_counts["f0_hz"] / settings.n_points,
            "valid_intensity_points": valid_counts["intensity_db"],
        }
        for formant_number in range(1, settings.n_formants + 1):
            column = f"F{formant_number}_hz"
            qc_row[f"valid_F{formant_number}_points"] = valid_counts[column]
        qc_rows.append(qc_row)

    return rows, qc_rows


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def build_upload_map(files: list[Any]) -> tuple[dict[str, Any], list[str]]:
    mapping: dict[str, Any] = {}
    duplicates: list[str] = []
    for uploaded in files:
        key = Path(uploaded.name).stem.casefold()
        if key in mapping:
            duplicates.append(Path(uploaded.name).stem)
        else:
            mapping[key] = uploaded
    return mapping, duplicates


def make_metadata(settings: ExtractionSettings) -> pd.DataFrame:
    values: dict[str, Any] = {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        **settings.as_dict(),
        "python_note": "F0 is missing (NaN) when unvoiced; voiced is coded 0/1.",
        "streamlit_version": package_version("streamlit"),
        "pandas_version": package_version("pandas"),
        "praat_parselmouth_version": package_version("praat-parselmouth"),
        "tgt_version": package_version("tgt"),
    }
    return pd.DataFrame(
        {"parameter": list(values.keys()), "value": list(values.values())}
    )


def make_excel(
    data: pd.DataFrame,
    qc: pd.DataFrame,
    metadata: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="trajectories", index=False)
        qc.to_excel(writer, sheet_name="interval_qc", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)
    return buffer.getvalue()


st.set_page_config(page_title=APP_TITLE, page_icon="〽️", layout="wide")
st.title(APP_TITLE)
st.caption("FPCA-ready F0, intensity, and formant trajectories — Seongwoo Kang")

st.markdown(
    "Upload paired audio and TextGrid files. Files are matched by filename "
    "(case-insensitive), and every interval is sampled at **100 normalised points**."
)
st.info(
    "Unvoiced F0 remains missing (NaN), while `voiced` is coded as 0/1. "
    "The app does not extrapolate F0 across unvoiced regions."
)

audio_files = st.file_uploader(
    "Audio files",
    type=["wav", "aiff", "aif", "flac"],
    accept_multiple_files=True,
)
textgrid_files = st.file_uploader(
    "TextGrid files",
    type=["textgrid"],
    accept_multiple_files=True,
)

st.sidebar.header("Extraction parameters")
tier_name = st.sidebar.text_input("Interval tier", value="segment")
pitch_floor = st.sidebar.number_input(
    "Pitch floor (Hz)", min_value=20.0, value=75.0, step=5.0
)
pitch_ceiling = st.sidebar.number_input(
    "Pitch ceiling (Hz)", min_value=50.0, value=400.0, step=10.0
)
max_formant = st.sidebar.number_input(
    "Maximum formant (Hz)", min_value=1000.0, value=5500.0, step=100.0
)
n_formants = st.sidebar.slider("Number of formants", 3, 6, 5, 1)
time_step_ms = st.sidebar.number_input(
    "Praat analysis step (ms)", min_value=1.0, max_value=20.0, value=5.0, step=1.0
)
keep_empty = st.sidebar.checkbox("Keep empty intervals", value=False)
st.sidebar.number_input(
    "Normalised points per interval",
    min_value=100,
    max_value=100,
    value=100,
    disabled=True,
)

settings = ExtractionSettings(
    tier_name=tier_name,
    n_points=100,
    time_step_s=float(time_step_ms) / 1000.0,
    pitch_floor_hz=float(pitch_floor),
    pitch_ceiling_hz=float(pitch_ceiling),
    max_formant_hz=float(max_formant),
    n_formants=int(n_formants),
    skip_empty=not keep_empty,
)

validation_error: str | None = None
try:
    settings.validate()
except ValueError as exc:
    validation_error = str(exc)
    st.error(validation_error)

audio_map, audio_duplicates = build_upload_map(audio_files or [])
textgrid_map, textgrid_duplicates = build_upload_map(textgrid_files or [])

if audio_duplicates or textgrid_duplicates:
    duplicate_names = sorted(set(audio_duplicates + textgrid_duplicates))
    st.error(
        "Duplicate filename stems are not allowed: " + ", ".join(duplicate_names)
    )

matched = sorted(set(audio_map) & set(textgrid_map))
audio_only = sorted(set(audio_map) - set(textgrid_map))
textgrid_only = sorted(set(textgrid_map) - set(audio_map))

if audio_files or textgrid_files:
    st.write(f"**{len(matched)} matched pair(s)**")
    if audio_only:
        st.warning("Audio without TextGrid: " + ", ".join(audio_only))
    if textgrid_only:
        st.warning("TextGrid without audio: " + ", ".join(textgrid_only))
    if matched:
        with st.expander("Matched pairs"):
            for stem in matched:
                st.write(
                    f"{audio_map[stem].name}  +  {textgrid_map[stem].name}"
                )

can_run = bool(
    matched
    and not audio_duplicates
    and not textgrid_duplicates
    and validation_error is None
)

if st.button("Run 100-point extraction", type="primary", disabled=not can_run):
    all_rows: list[dict[str, Any]] = []
    all_qc_rows: list[dict[str, Any]] = []
    logs: list[str] = []
    successful_pairs = 0
    progress = st.progress(0.0, text="Preparing extraction…")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for pair_index, stem in enumerate(matched, start=1):
            audio_upload = audio_map[stem]
            textgrid_upload = textgrid_map[stem]

            # Path.name prevents an uploaded filename from escaping the temp folder.
            audio_path = tmp / Path(audio_upload.name).name
            textgrid_path = tmp / Path(textgrid_upload.name).name
            audio_path.write_bytes(audio_upload.getvalue())
            textgrid_path.write_bytes(textgrid_upload.getvalue())

            try:
                rows, qc_rows = extract_pair(audio_path, textgrid_path, settings)
                if rows:
                    all_rows.extend(rows)
                    all_qc_rows.extend(qc_rows)
                    successful_pairs += 1
                    logs.append(
                        f"[ok] {audio_upload.name}: {len(qc_rows)} interval(s), "
                        f"{len(rows):,} rows"
                    )
                else:
                    logs.append(
                        f"[error] {audio_upload.name}: no intervals remained "
                        "after tier and empty-label filtering"
                    )
            except Exception as exc:
                logs.append(f"[error] {audio_upload.name}: {exc}")

            progress.progress(
                pair_index / len(matched),
                text=f"Processed {pair_index}/{len(matched)} pair(s)",
            )

    progress.empty()
    if all_rows:
        data = pd.DataFrame(all_rows)
        qc = pd.DataFrame(all_qc_rows)
        metadata = make_metadata(settings)
        csv_bytes = data.to_csv(index=False, na_rep="").encode("utf-8-sig")

        excel_bytes: bytes | None = None
        if len(data) + 1 <= EXCEL_ROW_LIMIT:
            excel_bytes = make_excel(data, qc, metadata)
        else:
            logs.append(
                "[warning] Excel disabled because the trajectories sheet exceeds "
                "Excel's row limit. Use CSV."
            )

        st.session_state["acoustic_result"] = {
            "data": data,
            "qc": qc,
            "metadata": metadata,
            "csv_bytes": csv_bytes,
            "excel_bytes": excel_bytes,
            "logs": logs,
            "successful_pairs": successful_pairs,
            "failed_pairs": len(matched) - successful_pairs,
        }
    else:
        st.session_state.pop("acoustic_result", None)
        st.error("No data were extracted. Open the extraction log below.")
        with st.expander("Extraction log", expanded=True):
            st.code("\n".join(logs), language=None)


result = st.session_state.get("acoustic_result")
if result:
    st.success(
        f"Extraction complete: {len(result['data']):,} rows from "
        f"{result['successful_pairs']} successful pair(s); "
        f"{result['failed_pairs']} failed."
    )

    with st.expander("Extraction log"):
        st.code("\n".join(result["logs"]), language=None)

    tab_preview, tab_qc, tab_metadata = st.tabs(
        ["Trajectory preview", "Interval QC", "Metadata"]
    )
    with tab_preview:
        st.dataframe(result["data"].head(2_000), use_container_width=True)
        if len(result["data"]) > 2_000:
            st.caption("Preview limited to the first 2,000 rows.")
    with tab_qc:
        st.dataframe(result["qc"], use_container_width=True)
    with tab_metadata:
        st.dataframe(result["metadata"], use_container_width=True)

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download CSV",
            data=result["csv_bytes"],
            file_name="acoustic_trajectories_100pt.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        if result["excel_bytes"] is not None:
            st.download_button(
                "Download Excel",
                data=result["excel_bytes"],
                file_name="acoustic_trajectories_100pt.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )
        else:
            st.warning("Excel unavailable: row limit exceeded. Download CSV.")
elif not audio_files or not textgrid_files:
    st.info("Upload at least one matching audio/TextGrid pair to begin.")
