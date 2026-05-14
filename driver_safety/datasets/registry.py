from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    name: str
    use_case: str
    access_url: str
    license_summary: str
    local_layout: str
    prepare_command: str
    media_policy: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str | list[str]]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="nitymed",
        name="NITYMED: Night-Time Yawning-Microsleep-Eyeblink-driver Distraction",
        use_case="Real in-car night driving video with yawning and microsleep clips.",
        access_url="https://datasets.esdalab.ece.uop.gr/",
        license_summary=(
            "Creative Commons Attribution. Access requires submitting name, affiliation, and "
            "business or academic email through the dataset page."
        ),
        local_layout="data/nitymed/{Yawning,Microsleep}/{Male,Female}/{HDTV720,FULL}/",
        prepare_command=(
            "ai-driver-safety datasets prepare-nitymed --input data/nitymed "
            "--out data/manifests/nitymed.json"
        ),
        media_policy=(
            "Use as the preferred README demo source after access approval. Commit only short "
            "derived assets with attribution, not the full raw dataset."
        ),
        notes=(
            "Best immediate fit for a real human driving/yawning demo.",
            "Includes 107 yawning videos and 21 microsleep videos according to the dataset page.",
        ),
    ),
    DatasetSpec(
        key="yawdd",
        name="YawDD: Yawning Detection Dataset",
        use_case="Real in-car human yawning, talking, singing, and silent-driving video.",
        access_url="https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset",
        license_summary=(
            "Open-access research dataset. Public display of screenshots is limited by the "
            "Participants Information file; do not redistribute full videos from this repo."
        ),
        local_layout="data/yawdd/{mirror-camera,dash-camera}/",
        prepare_command=(
            "ai-driver-safety datasets prepare-yawdd --input data/yawdd "
            "--participants-info data/yawdd/ParticipantsInformation.csv "
            "--out data/manifests/yawdd.json"
        ),
        media_policy=(
            "Use local videos for benchmarking. Commit only screenshots explicitly allowed by "
            "the participant metadata, and do not commit annotated full-video derivatives."
        ),
        notes=(
            "Primary validation track for yawn/mouth scoring.",
            "Filename heuristics identify normal, talking, and yawning clips when frame labels are absent.",
        ),
    ),
    DatasetSpec(
        key="dd-database",
        name="Drivers Drowsiness Database (DD-Database)",
        use_case="Physiological drowsiness signals collected in a driving simulator.",
        access_url="https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c",
        license_summary=(
            "Dryad-hosted public dataset. Keep raw EDF files under data/ and cite the Dryad DOI."
        ),
        local_layout="data/dd-database/raw/*.edf",
        prepare_command=(
            "ai-driver-safety datasets prepare-dd --input data/dd-database/raw "
            "--download-index data/dd-database/dryad-files.json "
            "--out data/manifests/dd-database.json"
        ),
        media_policy=(
            "Contains de-identified physiological EDF files rather than identifiable video; "
            "raw files still stay out of git."
        ),
        notes=(
            "Uses EEG, EOG, and ECG channels plus annotation EDF files.",
            "Install ai-driver-safety[datasets] to parse EDF annotations into sensor drowsiness events.",
        ),
    ),
    DatasetSpec(
        key="uah-driveset",
        name="UAH-DriveSet",
        use_case="Real driving smartphone, GPS, lane, vehicle, and behavior telemetry.",
        access_url="https://github.com/Eromera/uah_driveset_reader",
        license_summary=(
            "Creative Commons Attribution-NonCommercial 4.0 according to the official reader tool."
        ),
        local_layout="data/uah-driveset/**/{RAW_ACCELEROMETERS,RAW_GPS,PROC_LANE_DETECTION}.txt",
        prepare_command=(
            "ai-driver-safety datasets prepare-uah --input data/uah-driveset "
            "--out data/manifests/uah-driveset.json"
        ),
        media_policy=(
            "Use for local research and non-commercial benchmarking. Keep videos and raw telemetry out of git."
        ),
        notes=(
            "Primary validation track for car-sensor risk signals.",
            "Behavior labels are inferred from the dataset folder names: normal, drowsy, aggressive.",
        ),
    ),
)


def all_dataset_specs() -> tuple[DatasetSpec, ...]:
    return DATASETS


def get_dataset_spec(key: str) -> DatasetSpec:
    for spec in DATASETS:
        if spec.key == key:
            return spec
    available = ", ".join(spec.key for spec in DATASETS)
    raise KeyError(f"Unknown dataset '{key}'. Available datasets: {available}")
