import os
import shutil
import argparse
from pathlib import Path


def flatten_adni_dataset(raw_dir, clean_dir):
  raw_path = Path(raw_dir)
  clean_path = Path(clean_dir)

  # Validate input directory
  if not raw_path.exists() or not raw_path.is_dir():
    print(f"Error: The raw directory '{raw_dir}' does not exist.")
    return

  # Create the clean base directory if it doesn't exist
  clean_path.mkdir(parents=True, exist_ok=True)

  # Get all top-level folders (These should be your Subject IDs like 116_S_4855)
  subject_folders = [f for f in raw_path.iterdir() if f.is_dir()]

  print(
    f"Found {len(subject_folders)} subject folders. Beginning extraction...\n")

  processed_count = 0

  for subject_folder in subject_folders:
    subject_id = subject_folder.name

    # Auto-discover all NIfTI files deeply nested inside this subject's folder
    # We look for both uncompressed (.nii) and compressed (.nii.gz)
    nifti_files = list(subject_folder.rglob('*.nii')) + list(
      subject_folder.rglob('*.nii.gz'))

    if not nifti_files:
      continue  # Skip folders that don't have any converted NIfTI files yet

    # Create the new, clean subject directory: e.g., <clean_dataset>/116_S_4855/
    target_subject_dir = clean_path / subject_id
    target_subject_dir.mkdir(parents=True, exist_ok=True)

    mri_found = False
    pet_found = False

    for nifti in nifti_files:
      path_str = str(nifti).lower()

      # Preserve the original extension (either .nii or .nii.gz)
      # nifti.suffixes returns ['.nii', '.gz'] for compressed files
      extension = "".join(nifti.suffixes)

      # Detect PET scans (ADNI uses 'av45', 'fdg', or 'pet' in the protocol/filename)
      if ('av45' in path_str or 'pet' in path_str or 'fbb' in path_str) and not pet_found:
        target_file = target_subject_dir / f"pet{extension}"
        shutil.copy2(nifti, target_file)
        pet_found = True

      # Detect MRI scans (ADNI uses 'mt1', 'mprage', 'n3m', or '_mr_' in the protocol/filename)
      elif (
          'mt1' in path_str or 'mprage' in path_str or '_mr_' in path_str) and not mri_found:
        target_file = target_subject_dir / f"mri{extension}"
        shutil.copy2(nifti, target_file)
        mri_found = True

    # Log output for visibility
    if mri_found or pet_found:
      processed_count += 1
      status = []
      if mri_found: status.append("MRI")
      if pet_found: status.append("PET")
      print(f"Extracted {subject_id} -> [{', '.join(status)}]")

  print(
    f"\nCleanup Complete! Flattened {processed_count} subjects into: {clean_path.absolute()}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Flatten deeply nested ADNI NIfTI files into a clean <Subject>/[mri|pet].nii structure.")
  parser.add_argument("raw_dataset_dir", type=str,
                      help="Path to the messy, nested ADNI dataset")
  parser.add_argument("clean_dataset_dir", type=str,
                      help="Path where the clean dataset will be generated")

  args = parser.parse_args()

  flatten_adni_dataset(args.raw_dataset_dir, args.clean_dataset_dir)