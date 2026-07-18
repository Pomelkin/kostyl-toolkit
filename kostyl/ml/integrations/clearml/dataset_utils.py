from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from clearml import Dataset as ClearMLDataset


def collect_clearml_datasets(
    datasets_mapping: dict[str, str],
) -> dict[str, ClearMLDataset]:
    """
    Collect ClearML datasets by dataset ID.

    Args:
        datasets_mapping: Mapping where keys are human-readable names and values
            are ClearML dataset IDs.

    Returns:
        A mapping of dataset names to fetched `ClearMLDataset` instances.

    """
    datasets_list = {}
    for name, dataset_id in datasets_mapping.items():
        clearml_dataset = ClearMLDataset.get(dataset_id, alias=name)
        datasets_list[name] = clearml_dataset
    return datasets_list


def download_clearml_datasets(datasets: Collection[ClearMLDataset]) -> None:
    """
    Download all ClearML datasets in parallel.

    Args:
        datasets: Collection of initialized `ClearMLDataset` instances to download.

    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ds.get_local_copy) for ds in datasets]
        for future in as_completed(futures):
            future.result()
    return


def get_datasets_paths(datasets_mapping: dict[str, ClearMLDataset]) -> dict[str, Path]:
    """
    Return local filesystem paths for ClearML datasets.

    Args:
        datasets_mapping: Mapping of dataset names to initialized
            `ClearMLDataset` instances.

    Returns:
        Mapping of dataset names to local `Path` objects pointing to the
        downloaded dataset copies.

    """
    return {name: Path(ds.get_local_copy()) for name, ds in datasets_mapping.items()}
